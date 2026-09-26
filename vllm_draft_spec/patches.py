"""The four patches that let vLLM V1 reach a draft-model proposer.

Each is deliberately as small as possible, and each is annotated with the exact
upstream line it works around, because those line numbers are the first thing
that will rot on a vLLM upgrade.

  1. SpeculativeConfig.__post_init__   config/speculative.py:377-384
     Swallow the NotImplementedError and replay the config tail it skipped.
  2. SpeculativeConfig.use_eagle       config/speculative.py:636-637
     Two lines, and the highest-leverage change in the package -- see below.
  3. EngineArgs._check_feature_supported  engine/arg_utils.py:1774-1779
     The same refusal, raised earlier, when method is passed explicitly.
  4. GPUModelRunner.__init__           v1/worker/gpu_model_runner.py:374-396
     Rebind self.drafter to our proposer after the runner builds its own.
"""
import ast

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False


def _patch_speculative_config():
    from vllm.config.speculative import SpeculativeConfig

    orig = SpeculativeConfig.__post_init__

    def __post_init__(self):
        try:
            return orig(self)
        except NotImplementedError as e:
            if "draft model" not in str(e).lower():
                raise
        # The raise fires at speculative.py:379, after draft_model_config has
        # been built (:322) but before the tail at :404-460. Replay that tail
        # verbatim; skipping it leaves speculative_token_tree and
        # draft_parallel_config unset and the engine dies much later with a
        # far less obvious error.
        hf = self.draft_model_config.hf_config
        if self.num_speculative_tokens is not None and hasattr(hf, "num_lookahead_tokens"):
            hf.num_lookahead_tokens = self.num_speculative_tokens

        n_predict = getattr(hf, "n_predict", None)
        if n_predict is not None:
            if self.num_speculative_tokens is None:
                self.num_speculative_tokens = n_predict
            elif (self.num_speculative_tokens > n_predict
                  and self.num_speculative_tokens % n_predict != 0):
                raise ValueError(
                    f"num_speculative_tokens:{self.num_speculative_tokens}"
                    f" must be divisible by {n_predict=}")

        if self.speculative_token_tree is None:
            self.speculative_token_tree = str(
                [(i + 1) * (0,) for i in range(self.num_speculative_tokens)])
        else:
            tree_choices = ast.literal_eval(self.speculative_token_tree)
            self.speculative_token_tree = str(
                sorted(tree_choices, key=lambda t: (len(t), t)))

        self.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(
            self.target_parallel_config, self.draft_tensor_parallel_size, hf)
        self.draft_model_config.max_model_len = (
            SpeculativeConfig._maybe_override_draft_max_model_len(
                self.max_model_len, self.draft_model_config.max_model_len,
                self.target_model_config.max_model_len))
        self.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            self.target_parallel_config, self.draft_tensor_parallel_size)
        logger.info("adaptive-specdec: enabled draft_model speculative decoding "
                    "(draft=%s, num_speculative_tokens=%s)",
                    self.draft_model_config.model, self.num_speculative_tokens)
        return self

    SpeculativeConfig.__post_init__ = __post_init__


def _patch_use_eagle():
    """Make the stack treat draft_model the way it treats EAGLE.

    This is two lines and it buys, for free and correctly:
      * Scheduler.num_lookahead_tokens = k_max, so allocate_slots reserves the
        speculative slots (scheduler.py:184-187, :310)
      * KVCacheManager(use_eagle=True), which drops the last matched block on a
        prefix-cache hit -- REQUIRED for us, because the drafter's KV at
        position p is computed from token p+1, so that block is stale
        (single_type_kv_cache_manager.py:342-344)
      * the padded-drafter-batch path, the drafter's profile/capture hook, and
        the whole EAGLE branch of propose_draft_token_ids
        (gpu_model_runner.py:3134-3175, 4084-4101, 3342-3444)

    It is sound because a plain draft model sits in exactly the same structural
    position as an EAGLE head: a separate model, its own attention layers, its
    own KV, drafting k tokens per step off the target's token stream.
    """
    from vllm.config.speculative import SpeculativeConfig

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp", "draft_model")

    SpeculativeConfig.use_eagle = use_eagle


def _patch_engine_args():
    from vllm.engine.arg_utils import EngineArgs

    orig = EngineArgs._check_feature_supported

    def _check_feature_supported(self, *a, **kw):
        try:
            return orig(self, *a, **kw)
        except NotImplementedError as e:
            if "draft model" not in str(e).lower():
                raise
            return None

    EngineArgs._check_feature_supported = _check_feature_supported


def _patch_model_runner():
    """Swap in our proposer after the runner has built its own.

    The runner constructs an EagleProposer at gpu_model_runner.py:383 for any
    method where use_eagle() is true, which now includes us. That object is a
    few buffers and no weights (load_model has not run yet), so replacing it
    afterwards is cheap and avoids having to reimplement __init__'s ordering.

    Because DraftModelProposer subclasses EagleProposer, every
    isinstance(self.drafter, EagleProposer) check in the runner (:1640, :3157,
    :3343, :4085, :5259) keeps passing. That is the whole reason to subclass
    rather than write a proposer from scratch: zero further runner patches.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        sc = self.vllm_config.speculative_config
        if sc is not None and sc.method == "draft_model" and hasattr(self, "drafter"):
            from vllm_draft_spec.proposer import DraftModelProposer
            self.drafter = DraftModelProposer(self.vllm_config, self.device, self)
            logger.info("adaptive-specdec: installed DraftModelProposer")

    GPUModelRunner.__init__ = __init__


def apply_all():
    global _APPLIED
    if _APPLIED:
        return
    _patch_speculative_config()
    _patch_use_eagle()
    _patch_engine_args()
    _patch_model_runner()
    _APPLIED = True
    logger.info("adaptive-specdec: patches applied")
