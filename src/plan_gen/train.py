"""Training CLI: small GPT on serialized plan tokens (CPU-friendly).

usage: python src/plan_gen/train.py --data out_lib/plans/procthor_plans.jsonl \
           --out-dir out/plan_gen_smoke --steps 300 --batch 32

Next-token accuracy is reported for reference only; the honest layout
metrics live in metrics.py and are computed on sampled plans (see smoke.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_gen import data as D  # noqa: E402
from plan_gen.model import GPT, GPTConfig  # noqa: E402


def build_model() -> GPT:
    return GPT(GPTConfig(vocab=D.VOCAB, seq_len=D.SEQ_LEN))


@torch.no_grad()
def evaluate(model: GPT, seqs: torch.Tensor, batch: int = 256) -> dict:
    """Next-token accuracy over real (non-PAD) targets, split by slot role."""
    model.eval()
    correct = total = 0
    per = {"type": [0, 0], "coord": [0, 0]}
    for i in range(0, seqs.shape[0], batch):
        s = seqs[i:i + batch]
        x, y = s[:, :-1], s[:, 1:]
        pred, _ = model(x)
        pred = pred.argmax(-1)
        mask = y != D.PAD
        hit = (pred == y) & mask
        correct += int(hit.sum())
        total += int(mask.sum())
        pos = torch.arange(1, s.shape[1])[None].expand_as(y)
        r = (pos - 1) % 5
        is_coord = (r >= 1) & (r <= 4) & mask
        is_type = (r == 0) & mask & (y != D.EOS)
        for name, m in (("type", is_type), ("coord", is_coord)):
            per[name][0] += int((hit & m).sum())
            per[name][1] += int(m.sum())
    return {"overall": correct / max(total, 1),
            "type": per["type"][0] / max(per["type"][1], 1),
            "coord": per["coord"][0] / max(per["coord"][1], 1),
            "n_targets": total}


def lr_at(step: int, steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(steps - warmup, 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def train_loop(model: GPT, train_seqs: torch.Tensor, steps: int, batch: int,
               base_lr: float = 1e-3, warmup: int = 30, seed: int = 0,
               log_every: int = 25, log_fn=print):
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            betas=(0.9, 0.95), weight_decay=0.05)
    losses = []
    t0 = time.time()
    step = 0
    epoch = 0
    while step < steps:
        for x, y in D.batches(train_seqs, batch, seed=seed + epoch):
            lr = lr_at(step, steps, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            model.train()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            if step % log_every == 0 or step == steps - 1:
                el = time.time() - t0
                log_fn(f"[train] step {step:4d}/{steps} loss {losses[-1]:.4f} "
                       f"lr {lr:.2e} elapsed {el:.0f}s "
                       f"({el / (step + 1):.2f}s/step)")
            step += 1
            if step >= steps:
                break
        epoch += 1
    return {"losses": losses, "loss_first": losses[0],
            "loss_last": losses[-1], "loss_min": min(losses),
            "seconds": time.time() - t0,
            "sec_per_step": (time.time() - t0) / steps}


def load_or_build_splits(data_path: str, train_target: int = 9000,
                         seed: int = 42):
    houses = D.load_plans(data_path)
    train_idx, test_idx, info = D.make_split(houses, train_target, seed)
    return houses, train_idx, test_idx, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_lib/plans/procthor_plans.jsonl")
    ap.add_argument("--out-dir", default="out/plan_gen_smoke")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    os.makedirs(args.out_dir, exist_ok=True)

    houses, train_idx, test_idx, info = load_or_build_splits(
        args.data, seed=args.split_seed)
    print(f"[data] {info}")
    train_seqs = D.PlanDataset.from_houses(houses, train_idx).seqs
    test_seqs = D.PlanDataset.from_houses(houses, test_idx).seqs
    print(f"[data] train seqs {tuple(train_seqs.shape)} "
          f"test seqs {tuple(test_seqs.shape)}")

    model = build_model()
    print(f"[model] params {model.num_params():,} cfg {model.config_dict()}")
    stats = train_loop(model, train_seqs, args.steps, args.batch,
                       base_lr=args.lr, warmup=args.warmup, seed=args.seed)
    acc = evaluate(model, test_seqs)
    print(f"[eval] test next-token acc (reference only): {acc}")

    ckpt = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({"model": model.state_dict(), "config": model.config_dict(),
                "train_stats": {k: v for k, v in stats.items()
                                if k != "losses"},
                "test_acc": acc, "split": info,
                "data": args.data}, ckpt)
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump({"losses": stats["losses"],
                   "train_stats": {k: v for k, v in stats.items()
                                   if k != "losses"},
                   "test_acc": acc, "split": info,
                   "args": vars(args)}, f, indent=1)
    print(f"[train] saved {ckpt}")


if __name__ == "__main__":
    main()
