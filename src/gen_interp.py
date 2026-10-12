"""Generative interpolation: decode points along latent lines.

The AE has a decoder, so we can finally ask the real question from the
design discussion: what does the model GENERATE between two clusters?

For each cluster pair: z(t) = (1-t) * centroid_A + t * centroid_B, decode,
threshold, render. Also decodes 8 random latent samples to check whether the
decoder is generative at all (or only reconstructs).

Outputs: out/ae/gen_interp_{A}_{B}.png, out/ae/gen_random.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402
from train_ae import AE, SIDE  # noqa: E402


def to_grid(flat: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (flat.reshape(SIDE, SIDE, SIDE) > thr).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--pairs", default="0-6,2-3,4-1")
    args = ap.parse_args()
    adir = os.path.join(args.out, "ae")

    Z = np.load(os.path.join(adir, "embeddings.npz"))["Z"]
    cid = np.load(os.path.join(adir, "embeddings.npz"))["cid"]
    model = AE(SIDE ** 3, Z.shape[1])
    model.load_state_dict(torch.load(os.path.join(args.out, "ae_model.pt")))
    model.eval()

    ctrs = {}
    for c in sorted(set(cid.tolist())):
        ctrs[c] = Z[cid == c].mean(axis=0)

    for pair in args.pairs.split(","):
        a, b = (int(v) for v in pair.split("-"))
        tiles, caps = [], []
        with torch.no_grad():
            for t in np.linspace(0, 1, 9):
                z = (1 - t) * torch.from_numpy(ctrs[a]) + \
                    t * torch.from_numpy(ctrs[b])
                flat = model.dec(z.unsqueeze(0)).numpy()[0]
                tiles.append(render(to_grid(flat), max_px=260))
                caps.append(f"t={t:.2f}")
        sheet(tiles, caps).save(
            os.path.join(adir, f"gen_interp_{a}_{b}.png"))
        print(f"[gen] decoded c{a}->c{b}", flush=True)

    g = np.random.default_rng(7)
    with torch.no_grad():
        zs = torch.from_numpy(g.normal(0, 1, (8, Z.shape[1])).astype(np.float32))
        zs = zs * Z.std(axis=0) + Z.mean(axis=0)
        tiles = [render(to_grid(model.dec(z.unsqueeze(0)).numpy()[0]),
                        max_px=260) for z in zs]
    sheet(tiles, [f"rnd{j + 1}" for j in range(8)]).save(
        os.path.join(adir, "gen_random.png"))
    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
