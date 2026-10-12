"""End-to-end smoke run for the program->plan generator.

Chain: data load -> group split -> metrics self-test -> short GPT training
-> test next-token accuracy -> type-conditioned sampling (greedy + temp 1.0)
-> honest layout metrics (degenerate / overlap / out-of-bounds, with the
ground-truth test set as baseline) -> 2D plan PNGs -> skeleton DSL export
-> deterministic compile + cutaway render.

usage (inside the cpu-limited container):
  python src/plan_gen/smoke.py --steps 300 --batch 32 --samples 48

Smoke only: a few hundred steps. Real training belongs to a bigger run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_gen import data as D  # noqa: E402
from plan_gen import metrics as M  # noqa: E402
from plan_gen.export_dsl import export_and_compile  # noqa: E402
from plan_gen.render_plan import render_grid  # noqa: E402
from plan_gen.sample import sample_plan  # noqa: E402
from plan_gen.train import (build_model, evaluate, load_or_build_splits,  # noqa: E402
                            train_loop)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_lib/plans/procthor_plans.jsonl")
    ap.add_argument("--out-dir", default="out/plan_gen_smoke")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out = args.out_dir
    t0 = time.time()

    # ---- 1. data + leak-free split
    houses, train_idx, test_idx, split_info = load_or_build_splits(args.data)
    log(f"[smoke] split {split_info}")
    train_seqs = D.PlanDataset.from_houses(houses, train_idx).seqs
    test_seqs = D.PlanDataset.from_houses(houses, test_idx).seqs
    log(f"[smoke] seq tensors train {tuple(train_seqs.shape)} "
        f"test {tuple(test_seqs.shape)} vocab {D.VOCAB} seqlen {D.SEQ_LEN}")

    # ---- 2. metrics self-test (guard against constant-returning metrics)
    st = M.self_test()
    log(f"[smoke] metrics self-test pass={st['pass']} checks={st['checks']}")

    # ---- 3. short training run
    model = build_model()
    log(f"[smoke] model params {model.num_params():,} "
        f"cfg {model.config_dict()}")
    stats = train_loop(model, train_seqs, args.steps, args.batch,
                       base_lr=args.lr, seed=args.seed, log_every=25,
                       log_fn=log)
    log(f"[smoke] train loss first {stats['loss_first']:.4f} "
        f"last {stats['loss_last']:.4f} min {stats['loss_min']:.4f} "
        f"({stats['sec_per_step']:.2f}s/step)")
    ckpt_path = os.path.join(out, "ckpt.pt")
    torch.save({"model": model.state_dict(), "config": model.config_dict(),
                "train_stats": {k: v for k, v in stats.items()
                                if k != "losses"},
                "test_acc": None, "split": split_info, "data": args.data},
               ckpt_path)
    log(f"[smoke] saved checkpoint {ckpt_path}")

    # ---- 4. reference metric
    acc = evaluate(model, test_seqs)
    log(f"[smoke] test next-token acc (reference only) {acc}")

    # ---- 5. conditional sampling + honest metrics
    n_samp = min(args.samples, len(test_idx))
    programs = [[r["type"] for r in houses[hi]["rooms"]]
                for hi in test_idx[:n_samp]]
    gt_plans = [D.normalize_house(houses[hi]) for hi in test_idx[:n_samp]]
    gt_counts = [M.layout_counts(p) for p in gt_plans]
    metrics_out = {
        "config": vars(args),
        "split": split_info,
        "metrics_self_test": st,
        "train": {k: v for k, v in stats.items() if k != "losses"},
        "test_next_token_acc_reference": acc,
        "ground_truth_test_baseline": M.aggregate(gt_counts),
    }

    gen_plans = {}
    for mode, temp in (("greedy", 1.0), ("temp1.0", 1.0)):
        plans, infos = [], []
        for i, prog in enumerate(programs):
            rooms, info = sample_plan(
                model, prog,
                mode="greedy" if mode == "greedy" else "sample",
                temp=temp, top_k=0, seed=args.seed + i)
            plans.append(rooms)
            infos.append(info)
        counts = [M.layout_counts(p) for p in plans]
        agg = M.aggregate(counts)
        agg["rooms_requested"] = sum(len(p) for p in programs)
        agg["rooms_generated"] = sum(len(p) for p in plans)
        agg["eos_agreement"] = sum(i["eos_predicted"] for i in infos) / len(infos)
        agg["type_slots_match"] = sum(i["types_match"] for i in infos) / len(infos)
        metrics_out[f"generated_{mode}"] = agg
        gen_plans[mode] = plans
        log(f"[smoke] metrics[{mode}] {agg}")

    # ---- 6. 2D plan renders
    for mode, plans in gen_plans.items():
        p = render_grid(plans[:12], os.path.join(out, f"gen_grid_{mode}.png"),
                        cols=4, cell=220,
                        title=f"generated ({mode}, 12 plans)")
        log(f"[smoke] render {p}")
    p = render_grid([list(g) for g in gt_plans[:12]],
                    os.path.join(out, "gt_grid.png"), cols=4, cell=220,
                    title="ground truth test plans (12)")
    log(f"[smoke] render {p}")

    # ---- 7. DSL export -> skeleton compile -> voxel shell (end-to-end)
    dsl_summs = []
    for mode in ("temp1.0", "greedy"):
        rooms = gen_plans[mode][0]
        summ = export_and_compile(
            rooms, os.path.join(out, "dsl"), f"gen0_{mode}", size=12)
        dsl_summs.append({"mode": mode, **summ})
        log(f"[smoke] dsl[{mode}] {summ}")
    metrics_out["dsl_export"] = dsl_summs

    metrics_out["wall_seconds"] = time.time() - t0
    mf = os.path.join(out, "metrics.json")
    with open(mf, "w") as f:
        json.dump(metrics_out, f, indent=1)
    log(f"[smoke] wrote {mf}")
    log(f"[smoke] SMOKE OK in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
