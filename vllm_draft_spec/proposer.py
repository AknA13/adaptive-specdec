"""DraftModelProposer: speculative decoding with a real draft model in vLLM V1.

It subclasses EagleProposer, which sounds odd until you look at what EAGLE's
`propose()` actually does. Strip away the hidden-state plumbing and what is
left is: shift the target's token stream by one, build drafting attention
metadata, run a small model k times, and after each step bump positions,
seq_lens and the slot mapping by one. That machinery is exactly what a plain
draft model needs, and it is the part that is fiddly to get right.

Two properties of Qwen3 make the inherited parts correct rather than merely
convenient:

* The shift-by-one input layout (the drafter's KV at position p encodes token
  p+1) is a no-op for a model with pure RoPE and no sliding window, because
  shifting every position uniformly does not change relative distances. The
  only cost is that the drafter never sees token 0 of the prompt.

* Target and draft are both 8 KV heads x 128 head_dim x bf16, so their
  per-layer FullAttentionSpec is identical and all 36 + 28 = 64 layers land in
  ONE KV cache group sharing one block table. No KV-manager changes at all.
  Budget for it though: the same gpu_memory_utilization now buys ~44% fewer
  KV blocks, because a block spans 64 layers instead of 36.

There is also no separate draft prefill. `propose()` runs on every scheduled
token of every step, so the drafter rides the target's token stream through
chunked prefill, preemption and recompute for free.

Dynamic k
---------
`num_speculative_tokens` sizes buffers and the scheduler's lookahead at
startup, so we set it to k_max and vary k <= k_max per step. No captured
cudagraph shape depends on k -- the captured shapes are pad_for_cudagraph of
num_tokens and of batch_size -- so varying it costs nothing.
"""
import copy
import os
import time

import torch

from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
# PADDING_SLOT_ID lives in eagle.py itself (eagle.py:54), not in the
# attention backends utils where you would expect it.
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, EagleProposer

logger = init_logger(__name__)

# Must contain no digits: models/utils.py:extract_layer_index asserts a layer
# name holds exactly one integer, and "draft_0" would give it two.
DRAFT_PREFIX = "draft_model"


class DraftModelProposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        self.k_max = int(self.speculative_config.num_speculative_tokens)
        self.k = self.k_max
        self._pending_counts = None      # GPU tensor: 1 + accepted, previous step
        self._pending_drafted = None     # list[int]: drafted per request
        # Cost-ratio telemetry. Without this the controller optimises against
        # its c_init and picks k for a draft that is cheaper than the real one:
        # measured on an H200, leaving c at 0.15 made the adaptive policy track
        # fixed k=8 (0.32x AR) when the true c ~ 1.6 calls for k=1.
        self._t_prev_enter = None
        self._t_draft_prev = None
        self._k_prev = None
        self.controller = self._make_controller()
        logger.info("adaptive-specdec: DraftModelProposer k_max=%d controller=%s",
                    self.k_max, getattr(self.controller, "name", "fixed"))

    # ---- controller --------------------------------------------------------
    def _make_controller(self):
        """SPECDEC_CONTROLLER: 'fixed' (default, k=k_max) | 'ewma' | 'ewma+exit' | 'ucb'.

        Env rather than a config field because SpeculativeConfig is a pydantic
        model and adding a field to it would mean a much more invasive patch
        than this package is willing to make.
        """
        spec = os.environ.get("SPECDEC_CONTROLLER", "fixed").strip().lower()
        if spec in ("", "fixed", "none", "off"):
            return None
        try:
            from specdec.controller import make_controller
            # extra_fwd=0: vLLM's drafter rides the target's token stream and
            # needs no commit-only forward, unlike the from-scratch engine.
            return make_controller(spec, k_max=self.k_max, extra_fwd=0)
        except Exception as e:
            logger.warning("adaptive-specdec: controller %r unavailable (%s); "
                           "falling back to fixed k=%d", spec, e, self.k_max)
            return None

    def _select_k(self, t_draft_fwd=None, t_target_fwd=None):
        """Choose this step's k, folding in last step's acceptance and cost."""
        if self.controller is None:
            return self.k_max
        if self._pending_counts is not None and self._pending_drafted is not None:
            # Read the PREVIOUS step's counts. They were produced by a kernel
            # that has long since finished, so the D2H copy does not stall the
            # pipeline the way reading the current step's would.
            try:
                counts = self._pending_counts.tolist()
                drafted = self._pending_drafted
                n_acc = [max(0, int(c) - 1) for c in counts[:len(drafted)]]
                kk = max(1, max(drafted)) if drafted else self.k
                self.controller.update(kk, n_acc, dt=1.0,
                                       t_draft_fwd=t_draft_fwd,
                                       t_target_fwd=t_target_fwd)
            except Exception as e:
                logger.warning("adaptive-specdec: telemetry read failed: %s", e)
            self._pending_counts = None
            self._pending_drafted = None
        self.k = max(1, min(self.k_max, int(self.controller.propose_k())))
        return self.k

    def _step_costs(self, now):
        """Per-forward draft and target cost, from wall time between steps.

        The gap between consecutive propose() entries is one whole engine step:
        the previous step's drafting, then the target forward, sampling and
        scheduling. Subtracting the drafting we timed ourselves leaves the
        target side. Wall clock rather than CUDA events because the drafter is
        launch-bound here, so launch time is the cost that matters -- and
        because reading an event would sync the very pipeline we are measuring.
        """
        if self._t_prev_enter is None or self._t_draft_prev is None:
            return None, None
        period = now - self._t_prev_enter
        # An idle gap between requests is not a step; it would read as an
        # enormous target cost and drive k to the ceiling.
        if period <= 0 or period > 1.0:
            return None, None
        t_target = max(1e-5, period - self._t_draft_prev)
        t_draft_fwd = self._t_draft_prev / max(1, self._k_prev or 1)
        return t_draft_fwd, t_target

    # ---- telemetry hooks ---------------------------------------------------
    def prepare_inputs_padded(self, common_attn_metadata, spec_decode_metadata,
                              valid_sampled_tokens_count):
        self._pending_counts = valid_sampled_tokens_count
        self._pending_drafted = list(spec_decode_metadata.num_draft_tokens)
        return super().prepare_inputs_padded(
            common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count)

    def prepare_next_token_ids_cpu(self, sampled_token_ids, *a, **kw):
        # Non-padded path (disable_padded_drafter_batch=True). Accepted count is
        # len(sampled)-1 and it is already on the CPU, so this is free.
        if self.controller is not None and sampled_token_ids:
            n_acc = [max(0, len(s) - 1) for s in sampled_token_ids]
            t_d, t_t = self._step_costs(time.perf_counter())
            self.controller.update(max(1, self.k), n_acc, dt=1.0,
                                   t_draft_fwd=t_d, t_target_fwd=t_t)
        return super().prepare_next_token_ids_cpu(sampled_token_ids, *a, **kw)

    # ---- model loading -----------------------------------------------------
    def load_model(self, target_model) -> None:
        """Build the draft model under its own prefix and its own model config.

        Two things are load-bearing:

        * prefix=DRAFT_PREFIX. A stock Qwen3 built with prefix="" registers
          "model.layers.0.self_attn.attn" into the shared static_forward_context
          and collides with the target, which raises
          "Duplicate layer name" at attention/layer.py:276.

        * a shallow-copied VllmConfig whose model_config is the DRAFT config.
          Qwen3ForCausalLM.__init__ reads vllm_config.model_config.hf_config
          (qwen3.py:269), so without this it would build 36 layers of width 4096
          and then fail to load 0.6B weights into them. The copy is shallow on
          purpose: compilation_config -- and therefore static_forward_context --
          must stay shared, or the draft's attention layers never get KV cache.
        """
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.model_executor.model_loader.utils import (
            initialize_model, process_weights_after_loading)
        from vllm.utils.torch_utils import set_default_torch_dtype

        before = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase))

        dcfg = self.draft_model_config
        draft_vllm_config = copy.copy(self.vllm_config)
        draft_vllm_config.model_config = dcfg

        loader = get_model_loader(self.vllm_config.load_config)
        with set_model_tag(DRAFT_PREFIX):
            with set_default_torch_dtype(dcfg.dtype):
                with torch.device(self.device):
                    model = initialize_model(vllm_config=draft_vllm_config,
                                             model_config=dcfg, prefix=DRAFT_PREFIX)
                loader.load_weights(model, dcfg)
                process_weights_after_loading(model, dcfg, self.device)
        self.model = model.eval()

        after = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase))
        self.attn_layer_names = sorted(after - before)
        self.indexer_layer_names = []
        self.draft_indexer_metadata_builder = None
        logger.info("adaptive-specdec: draft model %s loaded, %d attention layers "
                    "registered under %r", dcfg.model, len(self.attn_layer_names),
                    DRAFT_PREFIX)
        if not self.attn_layer_names:
            raise RuntimeError(
                "draft model registered no attention layers -- the prefix is "
                "probably colliding with the target's")

    # ---- drafting ----------------------------------------------------------
    def propose(self, target_token_ids, target_positions, target_hidden_states,
                next_token_ids, last_token_indices, common_attn_metadata,
                sampling_metadata, mm_embed_inputs=None):
        """Mirrors EagleProposer.propose, minus hidden states, plus dynamic k.

        target_hidden_states is accepted and ignored: the runner computes and
        passes it for the EAGLE path, and a plain draft model conditions on
        token ids alone.
        """
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        _t_enter = time.perf_counter()
        _t_draft_fwd, _t_target_fwd = self._step_costs(_t_enter)
        k = self._select_k(_t_draft_fwd, _t_target_fwd)
        self._t_prev_enter = _t_enter

        # Shift the input ids by one, then overwrite each sequence's last slot
        # with its freshly sampled token:
        #   [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3] -> [a2, ...]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None
        builder = (self.attn_metadata_builder if self.attn_metadata_builder is not None
                   else self._get_attention_metadata_builder())
        attn_metadata = builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0)
        per_layer_attn_metadata = {ln: attn_metadata for ln in self.attn_layer_names}

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens)
        cudagraph_runtime_mode = CUDAGraphMode.NONE
        if (self.use_cuda_graph
                and num_tokens_dp_padded <= self.compilation_config.max_cudagraph_capture_size):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens_dp_padded)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        self._set_positions(num_tokens, target_positions)
        with set_forward_context(per_layer_attn_metadata, self.vllm_config,
                                 num_tokens=num_input_tokens,
                                 num_tokens_across_dp=num_tokens_across_dp,
                                 cudagraph_runtime_mode=cudagraph_runtime_mode):
            hidden_states = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens))
        logits = self.model.compute_logits(hidden_states[last_token_indices])
        draft_token_ids = logits.argmax(dim=-1)
        if k == 1:
            self._t_draft_prev = time.perf_counter() - _t_enter
            self._k_prev = 1
            return draft_token_ids.view(-1, 1)

        positions = target_positions[last_token_indices]
        draft_token_ids_list = [draft_token_ids]

        batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=batch_size, num_tokens_padded=batch_size)
        if (self.use_cuda_graph
                and batch_size_dp_padded <= self.compilation_config.max_cudagraph_capture_size):
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size_dp_padded)
            cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        else:
            input_batch_size = batch_size_dp_padded
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        if batch_size_across_dp is not None:
            batch_size_across_dp[self.dp_rank] = input_batch_size

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]).clone()

        for token_index in range(k - 1):
            # int32 matters: argmax returns int64 and the compiled draft model
            # is traced for int32 input ids.
            input_ids = draft_token_ids_list[-1].int()

            positions += 1
            # Requests that would run past the model length are kept in the
            # batch (removing them mid-step is messy) but parked at position 0
            # with seq_len 1 and a padding slot, and their drafts are ignored.
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu + 1
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            common_attn_metadata.num_computed_tokens_cpu = (
                common_attn_metadata.seq_lens_cpu - 1)

            block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)).view(-1)
            common_attn_metadata.slot_mapping = (
                block_ids * self.block_size + clamped_positions % self.block_size)
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID)

            attn_metadata = builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=token_index + 1)
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)

            with set_forward_context(per_layer_attn_metadata, self.vllm_config,
                                     num_tokens=input_batch_size,
                                     num_tokens_across_dp=batch_size_across_dp,
                                     cudagraph_runtime_mode=cudagraph_runtime_mode):
                hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self._get_positions(input_batch_size))
            logits = self.model.compute_logits(hidden_states[:batch_size])
            draft_token_ids_list.append(logits.argmax(dim=-1))

        self._t_draft_prev = time.perf_counter() - _t_enter
        self._k_prev = k
        return torch.stack(draft_token_ids_list, dim=1)

    # ---- warmup / capture --------------------------------------------------
    def dummy_run(self, num_tokens, use_cudagraphs=True, is_graph_capturing=False):
        """Same as the parent, without hidden_states/inputs_embeds.

        During capture only one forward is issued -- re-issuing the same shape
        would just re-capture the same graph. During warmup we issue k_max so
        the batch-shaped path of the drafting loop is compiled too.
        """
        cudagraphs_enabled = use_cudagraphs and self.use_cuda_graph
        num_input_tokens = num_tokens
        num_tokens_across_dp = None
        for fwd_idx in range(self.k_max if not is_graph_capturing else 1):
            if fwd_idx <= 1:
                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens)
                if (cudagraphs_enabled
                        and num_tokens_dp_padded
                        <= self.compilation_config.max_cudagraph_capture_size):
                    num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens_dp_padded)
                else:
                    num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

            with set_forward_context(
                    None, self.vllm_config, num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=(CUDAGraphMode.PIECEWISE if cudagraphs_enabled
                                            else CUDAGraphMode.NONE)):
                self.model(input_ids=self.input_ids[:num_input_tokens],
                           positions=self._get_positions(num_input_tokens))
