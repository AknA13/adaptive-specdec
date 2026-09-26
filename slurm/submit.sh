#!/bin/bash
# Generic SLURM submitter: wraps any repo command in a job script and submits it.
#
#   slurm/submit.sh --job traces --gpus 1 --requeue -- scripts/01_gen_traces.sh
#   slurm/submit.sh --job train  --gpus 2 --time 4:00:00 -- scripts/03_train_draft.sh --name sft_kd
#   slurm/submit.sh --job bench  --gpus 1 --after 12345  -- scripts/04_bench_engine.sh
#   slurm/submit.sh --print --job x --gpus 1 -- ...      # show the job script, submit nothing
#
# The generated script is kept at logs/<job>.job.sh and its output at logs/<job>.out,
# so you can re-submit or inspect exactly what ran.
#
# Cluster settings come from env.sh: SPEC_SLURM_{PARTITION,QOS,NODELIST,CPUS,TIME}.
SPEC_QUIET_ENV=1 source "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib.sh"

JOB=job; GPUS=1; TIME="${SPEC_SLURM_TIME:-8:00:00}"; AFTER=""; REQUEUE=0; PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --job)     JOB="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --time)    TIME="$2"; shift 2 ;;
    --after)   AFTER="$2"; shift 2 ;;     # chain: start only after this job id succeeds
    --requeue) REQUEUE=1; shift ;;        # survive preemption (stages are idempotent)
    --print)   PRINT=1; shift ;;
    --)        shift; break ;;
    *) die "unknown flag: $1 (did you forget the -- before the command?)" ;;
  esac
done
[ $# -gt 0 ] || die "no command given -- put it after --"

# ---- etiquette guard: never hold more than SPEC_MAX_GPUS across all my jobs ----
# Copied from antidistill/jobs/submit.sh. The cluster is shared and every
# partition preempts; hogging just means my own jobs get requeued.
CAP="${SPEC_MAX_GPUS:-4}"
if command -v squeue >/dev/null 2>&1; then
  HELD=$(squeue -u "$USER" -h -t RUNNING,PENDING -O "tres-alloc" 2>/dev/null \
         | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2 | paste -sd+ - | bc 2>/dev/null)
  HELD="${HELD:-0}"
  if [ $((HELD + GPUS)) -gt "$CAP" ]; then
    die "GPU cap: already hold ${HELD} GPU(s), asking for ${GPUS}, cap is ${CAP}.
     Raise it with SPEC_MAX_GPUS=N if you really mean to."
  fi
  info "gpu budget: ${HELD} held + ${GPUS} requested <= ${CAP}"
fi

PART="${SPEC_SLURM_PARTITION:-}"; QOS="${SPEC_SLURM_QOS:-}"
NODELIST="${SPEC_SLURM_NODELIST:-}"; CPUS="${SPEC_SLURM_CPUS:-32}"
mkdir -p "$REPO/logs"
JS="$REPO/logs/${JOB}.job.sh"

{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=spec_${JOB}"
  echo "#SBATCH --output=$REPO/logs/${JOB}.out"
  echo "#SBATCH --open-mode=append"
  [ -n "$PART" ]     && echo "#SBATCH --partition=$PART"
  [ -n "$QOS" ]      && echo "#SBATCH --qos=$QOS"
  [ -n "$NODELIST" ] && echo "#SBATCH --nodelist=$NODELIST"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --gres=gpu:${GPUS}"
  echo "#SBATCH --cpus-per-task=${CPUS}"
  echo "#SBATCH --time=${TIME}"
  [ "$REQUEUE" = 1 ] && echo "#SBATCH --requeue"
  echo
  echo "set -uo pipefail"
  echo "cd \"$REPO\""
  echo "echo \"[${JOB}] host=\$(hostname) gpus=\${CUDA_VISIBLE_DEVICES:-?} start=\$(date -Is)\""
  # Preemption guard: if fewer GPUs are visible than we asked for, the allocation
  # is broken -- requeue rather than silently training on the wrong world size.
  echo "NVIS=\$(echo \"\${CUDA_VISIBLE_DEVICES:-}\" | tr ',' '\\n' | grep -c .)"
  echo "if [ \"\$NVIS\" -lt ${GPUS} ]; then"
  echo "  echo \"[${JOB}] only \$NVIS/${GPUS} GPUs visible -- requeueing\" >&2"
  echo "  scontrol requeue \$SLURM_JOB_ID; sleep 60; exit 1"
  echo "fi"
  # Per-job compile caches: two jobs sharing an inductor/triton cache on the same
  # node race and corrupt it. NFS also hangs vLLM/inductor, so keep these local.
  echo "CROOT=\${TMPDIR:-/tmp}/spec-cache-${JOB}-\$SLURM_JOB_ID"
  echo "mkdir -p \"\$CROOT\"/{inductor,triton,xdg,tmp,vllm}"
  echo "export TORCHINDUCTOR_CACHE_DIR=\"\$CROOT/inductor\" TRITON_CACHE_DIR=\"\$CROOT/triton\""
  echo "export XDG_CACHE_HOME=\"\$CROOT/xdg\" TMPDIR=\"\$CROOT/tmp\" VLLM_CACHE_ROOT=\"\$CROOT/vllm\""
  echo "export VLLM_LOGGING_LEVEL=\${VLLM_LOGGING_LEVEL:-WARNING}"
  echo
  printf 'bash'; printf ' %q' "$@"; echo
  echo "rc=\$?"
  echo "echo \"[${JOB}] DONE rc=\$rc end=\$(date -Is)\""
  echo "rm -rf \"\$CROOT\""
  echo "exit \$rc"
} > "$JS"
chmod +x "$JS"

if [ "$PRINT" = 1 ]; then cat "$JS"; exit 0; fi
need_cmd sbatch
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency=afterok:"$AFTER")
ID=$(sbatch "${DEP[@]}" --parsable "$JS") || die "sbatch failed"
echo "submitted job $ID  ($JS)"
echo "  tail -f $REPO/logs/${JOB}.out"
