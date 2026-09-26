"""A KV cache whose rows can be rolled back independently, in O(1).

Why this exists
---------------
Speculative decoding writes k+1 tokens per round and then keeps only the
accepted prefix, so every round ends with a partial rollback. With batch > 1 the
number of accepted tokens differs per row, so the cache becomes ragged: row 0
might keep 4 tokens while row 1 keeps 1.

`DynamicCache.crop()` cannot express that -- it takes a single scalar length and
slices the whole batch. The obvious workarounds are both bad: re-prefilling the
accepted prefix throws away the work speculative decoding just did, and padding
every row to the max accepted length inflates the cache by ~2x, which would
corrupt the GPU-memory numbers we report.

So: one preallocated [B, H_kv, C, D] buffer per layer plus a per-row `lengths`
vector. Writes scatter to each row's own offset; rollback is `lengths -= n`.
Stale entries past a row's length are never attended (the mask is built from
`lengths`) and get overwritten by the next round. No allocation, no copy, no
synchronisation.

Correctness rests on one invariant, checked by tests/test_kv_rollback.py:

    after rollback, the cache is indistinguishable from having only ever
    processed the accepted prefix.

Mask convention
---------------
Row b's tokens occupy slots [0, lengths[b]) contiguously, so causality in slot
space is just `s <= lengths[b] + j` for query j. That single expression also
excludes the stale tail, because the stale tail lives at slots >= lengths[b] + q.
Right-padded prefill needs one extra term, `s < valid[b]`, to hide the pad.
"""
import torch
from transformers.cache_utils import Cache, CacheLayerMixin

__all__ = ["RaggedCache"]


class _RaggedLayer(CacheLayerMixin):
    """One layer's preallocated buffer. Offsets come from the shared parent."""

    is_sliding = False
    is_compileable = False

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.keys = None
        self.values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor):
        p = self.parent
        b, h, _, d = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.zeros(b, h, p.capacity, d, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(b, h, p.capacity, d, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        p = self.parent
        b, h, q, d = key_states.shape
        # Offsets are the lengths as they were BEFORE this forward. The parent
        # advances them once, after all layers have written -- never here, or
        # layer 1 would write at layer 0's post-write offset.
        idx = p.lengths.view(b, 1) + torch.arange(q, device=key_states.device).view(1, q)
        idx4 = idx.view(b, 1, q, 1).expand(b, h, q, d)
        self.keys.scatter_(2, idx4, key_states)
        self.values.scatter_(2, idx4, value_states)
        kv_len = p.kv_len_for(q)
        return self.keys[:, :, :kv_len, :], self.values[:, :, :kv_len, :]

    def get_mask_sizes(self, cache_position):
        q = cache_position.shape[0]
        return self.parent.kv_len_for(q), 0

    def get_seq_length(self):
        return int(self.parent.lengths.max().item()) if self.is_initialized else 0

    def get_max_cache_shape(self):
        return self.parent.capacity


class RaggedCache(Cache):
    """Per-row-length KV cache with O(1) rollback.

    Usage per forward:
        pos  = cache.positions(q)                  # [B, q] RoPE positions
        mask = cache.attn_mask(q, dtype, valid)    # [B, 1, q, kv_len]
        out  = model(input_ids, position_ids=pos, attention_mask=mask,
                     past_key_values=cache, cache_position=cache.cache_position(q))
        cache.advance(q)                           # exactly once per forward
    """

    def __init__(self, n_layers, batch_size, capacity, device):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.lengths = torch.zeros(self.batch_size, dtype=torch.long, device=device)
        self.device = device
        super().__init__(layers=[_RaggedLayer(self) for _ in range(n_layers)])

    # ---- geometry ----------------------------------------------------------
    def kv_len_for(self, q):
        """Width of the key/value window this forward will read."""
        return int(self.lengths.max().item()) + q

    def positions(self, q):
        """RoPE positions [B, q]: each row continues from its own length."""
        ar = torch.arange(q, device=self.device).view(1, q)
        return self.lengths.view(-1, 1) + ar

    def cache_position(self, q):
        """Shared 1-D cache_position. Only used by mask builders we bypass, but
        transformers wants it present and self-consistent."""
        base = int(self.lengths.max().item())
        return torch.arange(base, base + q, device=self.device)

    def attn_mask(self, q, dtype, valid=None):
        """Additive mask [B, 1, q, kv_len]; 0 where attention is allowed.

        valid: optional [B] cap on readable slots, for right-padded prefill.
        Additive float (not bool) and finfo.min (not -inf) to match what
        transformers itself produces -- a fully masked row then yields 0 rather
        than NaN.
        """
        kv_len = self.kv_len_for(q)
        s = torch.arange(kv_len, device=self.device).view(1, 1, kv_len)
        j = torch.arange(q, device=self.device).view(1, q, 1)
        allowed = s <= (self.lengths.view(-1, 1, 1) + j)
        if valid is not None:
            allowed = allowed & (s < valid.view(-1, 1, 1))
        neg = torch.finfo(dtype).min
        return torch.where(allowed, 0.0, neg).to(dtype).unsqueeze(1)

    # ---- mutation ----------------------------------------------------------
    def advance(self, q):
        """Call once per forward, after the model has run."""
        self.lengths += q

    def set_lengths(self, lengths):
        self.lengths.copy_(lengths.to(self.lengths.device, self.lengths.dtype))

    def rollback(self, n):
        """Drop the last n[b] tokens of row b. O(1): no copy, no allocation."""
        if not torch.is_tensor(n):
            n = torch.full_like(self.lengths, int(n))
        self.lengths -= n.to(self.lengths.device, self.lengths.dtype)
        if bool((self.lengths < 0).any()):
            raise RuntimeError("RaggedCache.rollback below zero")

    # ---- Cache interface ---------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0):
        return int(self.lengths.max().item())

    def reset(self):
        self.lengths.zero_()
        for layer in self.layers:
            if layer.is_initialized:
                layer.keys.zero_()
                layer.values.zero_()
