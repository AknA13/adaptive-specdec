#!/bin/bash
# Submit the whole pipeline as a SLURM chain. Each stage is idempotent, so a
# preempted stage just re-runs and skips what it already produced.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
step "submitting pipeline"
J1=$(slurm/submit.sh --job traces --gpus 1 --time 4:00:00 --requeue -- scripts/01_gen_traces.sh | awk '/submitted job/{print $3}')
info "traces  = $J1"
J2=$(slurm/submit.sh --job filter --gpus 1 --time 1:00:00 --after "$J1" -- scripts/02_filter.sh | awk '/submitted job/{print $3}')
info "filter  = $J2"
J3=$(slurm/submit.sh --job train --gpus 2 --time 4:00:00 --requeue --after "$J2" -- scripts/03_train_draft.sh | awk '/submitted job/{print $3}')
info "train   = $J3"
J4=$(slurm/submit.sh --job bench --gpus 1 --time 4:00:00 --after "$J3" -- scripts/04_bench_engine.sh | awk '/submitted job/{print $3}')
info "engine  = $J4"
J5=$(slurm/submit.sh --job vllm --gpus 1 --time 4:00:00 --after "$J4" -- scripts/05_bench_vllm.sh | awk '/submitted job/{print $3}')
info "serving = $J5"
J6=$(slurm/submit.sh --job prof --gpus 1 --time 2:00:00 --after "$J4" -- scripts/06_profile.sh | awk '/submitted job/{print $3}')
info "profile = $J6"
echo
echo "watch:  squeue -u $USER"
echo "logs :  tail -f $REPO/logs/*.out"
