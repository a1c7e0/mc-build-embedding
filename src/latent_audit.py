"""Latent audit: where does the VQ-VAE spend its codes, and are they local?

Experiments (the representation audit):
  1. co-occurrence: which codes appear next to which (8^3 grid adjacency)
  2. position heat: where in the building does each code family live
  3. perturbation: swap ONE token of a real building, decode, measure
     change locality (good: local change; bad: whole-house collapse)

Outputs numbers to stdout and perturbation renders to out_lib/atlas/.

usage: python src/latent_audit.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402
from train_vq import VQVAE  # noqa: E402

OUT = "out_lib/atlas"
CLASS_TO_ID = {0: 0, 1: 1, 2: 5, 3: 20, 4: 18, 5: 35}


def to_grid(x: np.ndarray) -> np.ndarray:
    g = np.zeros(x.shape, dtype=np.uint16)
    for c, i in CLASS_TO_ID.items():
        g[x == c] = i
    return g


def decode_codes(vq, cb, codes):  # noqa: ANN001
    with torch.no_grad():
        q = cb[codes].permute(0, 4, 1, 2, 3)
        return vq.dec(q).argmax(1)[0].numpy()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dev = "cpu"
    vq = VQVAE().to(dev)
    sd = torch.load("out_lib/vq_ckpt.pt", map_location=dev, weights_only=True)
    vq.load_state_dict(sd["model"])
    vq.eval()
    cb = getattr(getattr(vq.vq, "_codebook", None), "embed", None)
    if cb is None:
        cb = vq.vq.codebook.embed
    if cb.dim() == 3:
        cb = cb[0]

    d = np.load("out_lib/vq_tokens.npz")
    tokens, styles, idx = d["tokens"], d["styles"], d["idx"]
    vocab = int(tokens.max()) + 1
    grid = tokens.reshape(-1, 8, 8, 8)

    # 1. adjacency co-occurrence (x-direction, symmetric)
    co = np.zeros((vocab, vocab), dtype=np.int64)
    for ax in (0, 1, 2):
        a = np.moveaxis(grid, ax, 0)
        np.add.at(co, (a[:-1].reshape(-1), a[1:].reshape(-1)), 1)
    co = co + co.T
    tot = co.sum()
    same = float(np.trace(co) / max(tot, 1))
    top = np.argsort(-co.sum(axis=1))[:12]
    print(f"[audit] adjacency mass {tot}; top codes by link-degree: "
          f"{top.tolist()}", flush=True)
    # are links local in code space? (same-code dominance)
    same = int(np.trace(co)) / max(tot, 1)
    print(f"[audit] same-code adjacency fraction: {same:.3f}", flush=True)

    # 2. position heat: z-slices of usage for the 6 most common codes
    counts = np.bincount(tokens.reshape(-1), minlength=vocab)
    for code in np.argsort(-counts)[:6]:
        pos = np.argwhere(grid == code)
        if not len(pos):
            continue
        mz = pos[:, 3].mean() - 3.5
        mx = pos[:, 1].mean() - 3.5
        my = pos[:, 2].mean() - 3.5
        print(f"[audit] code {int(code)} n={int(counts[code])} "
              f"mean-pos dx={mx:+.2f} dy={my:+.2f} dz={mz:+.2f} "
              f"(0=centre of 8^3 grid)", flush=True)

    # 3. single-token perturbation locality
    meta = {int(r["idx"]): r for r in (json.loads(l) for l in
                                       open("out_lib/meta.jsonl",
                                            encoding="utf-8"))} \
        if False else None  # noqa: F841  (kept simple: use first item)
    import json as _json
    rng = np.random.default_rng(3)
    picks = rng.choice(len(grid), 3, replace=False)
    rows = []
    for pi in picks:
        orig = grid[pi:pi + 1].copy()
        rec0 = decode_codes(vq, cb, torch.from_numpy(orig).to(dev))
        flat = orig.reshape(-1)
        trials = []
        for _ in range(6):
            t = flat.copy()
            pos = int(rng.integers(0, len(t)))
            old = int(t[pos])
            new = int(rng.integers(0, vocab))
            while new == old:
                new = int(rng.integers(0, vocab))
            t[pos] = new
            rec1 = decode_codes(vq, cb,
                                torch.from_numpy(
                                    t.reshape(1, 8, 8, 8)).to(dev))
            diff = (rec0 != rec1)
            zm, xm, ym = np.unravel_index(int(np.argmax(diff * 0)), (32, 32, 32)) \
                if diff.any() else (0, 0, 0)
            trials.append(float(diff.mean()))
        rows.append((int(pi), float(np.mean(trials)), float(np.max(trials))))
        print(f"[audit] perturb item {int(pi)}: changed-voxel fraction "
              f"mean {np.mean(trials):.4f} max {np.max(trials):.4f}",
              flush=True)
    # render one perturbation pair for eyes
    pi = int(picks[0])
    orig = grid[pi:pi + 1].copy()
    t = orig.reshape(-1).copy()
    pos = int(rng.integers(0, len(t)))
    t[pos] = (int(t[pos]) + 137) % vocab
    rec0 = decode_codes(vq, cb, torch.from_numpy(orig).to(dev))
    rec1 = decode_codes(vq, cb, torch.from_numpy(
        t.reshape(1, 8, 8, 8)).to(dev))
    ims = [render2(to_grid(rec0), max_px=300), render2(to_grid(rec1), max_px=300)]
    h = max(im.height for im in ims)
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 8, h),
                      (255, 255, 255))
    sheet.paste(ims[0], (0, 0))
    sheet.paste(ims[1], (ims[0].width + 8, 0))
    sheet.save(f"{OUT}/perturb_{pi}.png")
    print(f"[audit] -> {OUT}/perturb_{pi}.png (left=orig right=1-token swap)",
          flush=True)


if __name__ == "__main__":
    main()
