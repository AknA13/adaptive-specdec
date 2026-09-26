# Adaptive speculative decoding. `make help` for the stage list.
SHELL := /bin/bash
PY ?= $(shell . ./env.sh 2>/dev/null; echo $${SPEC_PY:-python})

.PHONY: help test test-fast gates traces filter train bench-engine bench-vllm profile all clean

help:
	@echo "CPU (run anywhere, no GPU needed):"
	@echo "  make test         all correctness tests incl. gates G1 and G2-logic"
	@echo "  make test-fast    integrity + controller only (~2s)"
	@echo ""
	@echo "GPU stages (submit with slurm/submit.sh; each is idempotent):"
	@echo "  make traces       stage 1: Qwen3-8B reasoning traces + teacher top-k"
	@echo "  make filter       stage 2: verify, dedupe, decontaminate"
	@echo "  make train        stage 3: FSDP2 draft training (sft and sft_kd)"
	@echo "  make bench-engine stage 4: from-scratch engine benchmark grid"
	@echo "  make bench-vllm   stage 5: vLLM serving benchmark"
	@echo "  make profile      stage 6: torch.profiler traces + CUDA-graph delta"
	@echo "  make all          submit the whole chain with --dependency=afterok"

test:
	@set -e; rc=0; \
	for t in tests/test_repo_integrity.py tests/test_controller.py \
	         tests/test_kv_rollback.py tests/test_greedy_equivalence.py \
	         tests/test_losses.py tests/test_vllm_patches.py \
	         tests/test_rejection_sampling.py; do \
	  echo; echo "===== $$t ====="; \
	  $(PY) $$t || rc=1; \
	done; \
	echo; if [ $$rc -ne 0 ]; then echo "SOME TESTS FAILED"; else echo "ALL TESTS PASSED"; fi; \
	exit $$rc

test-fast:
	@$(PY) tests/test_repo_integrity.py && $(PY) tests/test_controller.py

traces:       ; bash scripts/01_gen_traces.sh
filter:       ; bash scripts/02_filter.sh
train:        ; bash scripts/03_train_draft.sh
bench-engine: ; bash scripts/04_bench_engine.sh
bench-vllm:   ; bash scripts/05_bench_vllm.sh
profile:      ; bash scripts/06_profile.sh
all:          ; bash scripts/run_pipeline.sh

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
