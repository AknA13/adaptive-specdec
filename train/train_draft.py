"""Stage 3: train the Qwen3-0.6B draft on filtered Qwen3-8B traces, with FSDP2.

  torchrun --nproc_per_node=2 -m train.train_draft --name sft_kd --lam 0.5

A note on FSDP at this scale, since it would be dishonest to imply otherwise:
0.6B in bf16 with AdamW fits comfortably on one H200, and DDP would train this
model perfectly well. FSDP2 is used because it shards optimizer state and
parameters the same way the pipeline would need for a larger draft, and because
`fully_shard` is what the rest of this cluster's training code has moved to. The
sharding is doing real work on the optimizer state, not on the parameters.

Preemption is the operational constraint here -- every partition on this cluster
is PreemptMode=REQUEUE -- so the run checkpoints on a wall-clock budget
(--stop-after-sec) and exits cleanly for the scheduler to requeue.
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def log(m, main=True):
    if main:
        print(f"[train] {m}", flush=True)


def get_layers(model):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. sft or sft_kd")
    ap.add_argument("--model", default=C.DRAFT_ID)
    ap.add_argument("--data", default=None, help="filtered jsonl (default: TRACE_DIR/filtered.jsonl)")
    ap.add_argument("--lam", type=float, default=C.KD_LAMBDA, help="0 = pure SFT")
    ap.add_argument("--kd-temperature", type=float, default=C.KD_TEMPERATURE)
    ap.add_argument("--epochs", type=float, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=4, help="per rank")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--topk", type=int, default=C.GEN_TOP_LOGPROBS)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--reduce-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--stop-after-sec", type=int, default=0,
                    help="checkpoint and exit before the SLURM time limit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    from train.data import TraceDataset, make_collate
    from train.losses import combined

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    rank, world = dist.get_rank(), dist.get_world_size()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    out_dir = C.CKPT_DIR / args.name
    if out_dir.exists() and (out_dir / "config.json").exists() and not args.overwrite:
        log(f"checkpoint already exists at {out_dir} -- nothing to do "
            f"(pass --overwrite to retrain)", is_main)
        dist.destroy_process_group()
        return 0

    data_path = Path(args.data) if args.data else (C.TRACE_DIR / "filtered.jsonl")
    if not data_path.exists():
        log(f"FATAL missing {data_path}; run stage 2 first", is_main)
        dist.destroy_process_group()
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = TraceDataset(data_path, tok, args.max_seq_len, args.topk, args.limit)
    log(f"{len(ds)} sequences ({ds.skipped_too_long} dropped over {args.max_seq_len} tokens) "
        f"world={world} lam={args.lam}", is_main)
    if len(ds) == 0:
        log("FATAL empty dataset", is_main)
        dist.destroy_process_group()
        return 1

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=2,
                    collate_fn=make_collate(pad_id, args.topk), drop_last=True, pin_memory=True)

    cfg = AutoConfig.from_pretrained(args.model)
    cfg.use_cache = False
    # flash-attn is NOT installed on this cluster; asking for it hard-fails.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32 if args.reduce_dtype == "fp32" else torch.bfloat16)
    mesh = init_device_mesh("cuda", (world,))
    for layer in get_layers(model):
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)      # root: embeddings / head / norm
    torch.cuda.empty_cache()

    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95), fused=True)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return args.min_lr_frac + (1 - args.min_lr_frac) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log(f"{total_steps} optimizer steps ({steps_per_epoch}/epoch, warmup {warmup})", is_main)

    metrics_path = C.LOG_DIR / f"train_{args.name}.jsonl"
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    mf = open(metrics_path, "a") if is_main else None

    step = 0
    micro = 0
    stop = False
    t0 = time.time()
    for epoch in range(math.ceil(args.epochs)):
        if stop:
            break
        sampler.set_epoch(epoch)
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, parts = combined(out.logits, batch, lam=args.lam,
                                   temperature=args.kd_temperature)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if is_main and (step % args.log_every == 0 or step == 1):
                el = time.time() - t0
                rec = {"step": step, "total": total_steps, "loss": float(loss.detach()),
                       "ce": parts["ce"], "kd": parts["kd"], "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(el, 1)}
                log(f"step {step}/{total_steps} loss={rec['loss']:.4f} "
                    f"ce={rec['ce']:.4f} kd={rec['kd']:.4f} gn={rec['grad_norm']:.2f} "
                    f"lr={rec['lr']:.2e} {el/60:.1f}min", True)
                mf.write(json.dumps(rec) + "\n")
                mf.flush()

            if step >= total_steps:
                stop = True
                break
            if args.stop_after_sec and (time.time() - t_start) > args.stop_after_sec:
                log(f"wall-clock budget reached at step {step}; checkpointing and exiting "
                    f"for requeue", is_main)
                stop = True
                break

    # ---- save a standard HF checkpoint that vLLM can load directly ----------
    dist.barrier()
    sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Rebuild on CPU and use save_pretrained so tied embeddings, the config
        # and the index file all come out in the exact layout vLLM expects.
        cpu_cfg = AutoConfig.from_pretrained(args.model)
        cpu_cfg.use_cache = True
        shell = AutoModelForCausalLM.from_config(cpu_cfg)
        missing, unexpected = shell.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
        tied = getattr(cpu_cfg, "tie_word_embeddings", False)
        bad = [m for m in missing if not (tied and m.endswith("lm_head.weight"))]
        if bad or unexpected:
            log(f"WARNING state dict mismatch missing={bad[:4]} unexpected={list(unexpected)[:4]}")
        shell = shell.to(torch.bfloat16)
        shell.save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
        meta = {"name": args.name, "base": args.model, "lam": args.lam,
                "steps": step, "total_steps": total_steps, "epochs": args.epochs,
                "lr": args.lr, "world": world, "seqs": len(ds),
                "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "wall_s": round(time.time() - t_start, 1), "out": str(out_dir)}
        (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
        C.publish_result(f"stage3_train_{args.name}", meta)
        log(f"saved {out_dir} after {step} steps ({(time.time()-t_start)/60:.1f} min)")
        mf.close()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
