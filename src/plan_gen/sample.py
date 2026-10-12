"""Type-conditioned plan sampling.

usage: python src/plan_gen/sample.py --ckpt out/plan_gen_smoke/ckpt.pt \
           --types Bedroom,Bathroom,Kitchen,LivingRoom --out-dir out/plan_gen_smoke/samples

Conditioning: the room slot ORDER is fixed by the same rule as training
(stable sort by (type rank, program index)) and the type token of every slot
is forced to the requested type; only the four coordinate tokens per room are
sampled from the model. The trailing EOS slot is sampled too and reported,
but decoding does not depend on it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_gen import data as D  # noqa: E402
from plan_gen import metrics as M  # noqa: E402
from plan_gen.model import GPT, GPTConfig  # noqa: E402
from plan_gen.render_plan import render_plan  # noqa: E402


@torch.no_grad()
def sample_plan(model: GPT, type_list: list[str], mode: str = "sample",
                temp: float = 1.0, top_k: int = 0,
                seed: int = 0) -> tuple[list, dict]:
    """Generate one plan for a program (list of room types).

    Returns (rooms, info); rooms = [(type, x0, z0, x1, z1)] normalized.
    mode 'greedy' takes argmax at every coordinate; 'sample' draws from the
    temperature/top-k filtered distribution.
    """
    model.eval()
    order = D.plan_order(type_list)
    ordered_types = [type_list[i] for i in order]
    gen = torch.Generator().manual_seed(seed)
    toks = [D.BOS]
    for typ in ordered_types:
        toks.append(D.TYPE_BASE + D.TYPE_RANK[typ])  # forced condition
        for _ in range(4):
            logits = model.next_logits(
                torch.tensor([toks], dtype=torch.long))[0]
            if mode == "greedy":
                t = int(torch.argmax(logits))
            else:
                logits = logits / max(temp, 1e-6)
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits = logits.masked_fill(logits < v[-1], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                t = int(torch.multinomial(probs, 1, generator=gen))
            toks.append(t)
    eos_logits = model.next_logits(torch.tensor([toks], dtype=torch.long))[0]
    eos_tok = int(torch.argmax(eos_logits))
    toks.append(eos_tok)
    rooms = D.decode(toks[1:])
    info = {"requested_rooms": len(type_list),
            "generated_rooms": len(rooms),
            "eos_predicted": eos_tok == D.EOS,
            "requested_types_sorted": ordered_types,
            "generated_types": [r[0] for r in rooms],
            "types_match": [r[0] for r in rooms] == ordered_types,
            "mode": mode, "temp": temp, "top_k": top_k, "seed": seed,
            "tokens": toks}
    return rooms, info


def load_ckpt(path: str) -> GPT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def parse_types(s: str) -> list[str]:
    out = [t.strip() for t in s.split(",") if t.strip()]
    for t in out:
        if t not in D.TYPE_RANK:
            raise SystemExit(f"unknown room type {t!r}; "
                             f"choose from {D.TYPES}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--types", required=True,
                    help="comma-separated room types, e.g. "
                         "Bedroom,Bathroom,Kitchen,LivingRoom")
    ap.add_argument("--out-dir", default="out/plan_gen_smoke/samples")
    ap.add_argument("--mode", choices=["greedy", "sample"], default="sample")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dsl", action="store_true",
                    help="also export+compile the skeleton DSL shell")
    args = ap.parse_args()

    type_list = parse_types(args.types)
    model = load_ckpt(args.ckpt)
    rooms, info = sample_plan(model, type_list, args.mode, args.temp,
                              args.top_k, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"plan_{args.mode}_{args.seed}"
    layout = {"program": type_list,
              "rooms": [{"type": r[0], "rect": [r[1], r[2], r[3], r[4]]}
                        for r in rooms],
              "info": {k: v for k, v in info.items() if k != "tokens"}}
    jf = os.path.join(args.out_dir, f"{tag}.json")
    json.dump(layout, open(jf, "w"), indent=1)
    render_plan(rooms, os.path.join(args.out_dir, f"{tag}.png"),
                title=f"{args.mode} {'/'.join(type_list)}")
    counts = M.layout_counts(rooms)
    print(f"[sample] {tag}: {len(rooms)} rooms {info['generated_types']} "
          f"eos={info['eos_predicted']} counts={counts}")
    if args.dsl:
        from plan_gen.export_dsl import export_and_compile
        summ = export_and_compile(rooms, os.path.join(args.out_dir, tag),
                                  tag)
        print(f"[sample] dsl export: {summ}")
    print(f"[sample] wrote {jf}")


if __name__ == "__main__":
    main()
