"""Multi-token perturbation: how far does a run of swapped codes reach?

For real buildings, replace a contiguous run of k codes (k = 2/4/8) with
random codes, decode, and measure the changed-voxel fraction. Complements
the single-token test: if locality is real, change should scale ~k x local;
if the decoder is merely insensitive, change stays tiny at every k.

usage: python src/perturb_multi.py
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
    vocab = cb.shape[0]

    d = np.load("out_lib/vq_tokens.npz")
    grid = d["tokens"].reshape(-1, 8, 8, 8)

    def decode(g8):  # noqa: ANN001
        with torch.no_grad():
            q = cb[torch.from_numpy(g8).to(dev)].permute(0, 4, 1, 2, 3)
            return vq.dec(q).argmax(1)[0].numpy()

    rng = np.random.default_rng(7)
    picks = rng.choice(len(grid), 3, replace=False)
    for k in (2, 4, 8):
        fr = []
        for pi in picks:
            t = grid[pi:pi + 1].copy()
            rec0 = decode(t)
            for _ in range(4):
                tt = t.copy()
                flat = tt.reshape(-1)
                pos = int(rng.integers(0, len(flat) - k))
                flat[pos:pos + k] = rng.integers(0, vocab, k)
                rec1 = decode(tt)
                fr.append(float((rec0 != rec1).mean()))
        print(f"[perturb-multi] k={k}: changed-voxel fraction "
              f"mean {np.mean(fr):.4f} max {np.max(fr):.4f}", flush=True)

    # render one k=4 pair
    pi = int(picks[0])
    t = grid[pi:pi + 1].copy()
    flat = t.reshape(-1)
    pos = int(rng.integers(0, len(flat) - 4))
    flat[pos:pos + 4] = rng.integers(0, vocab, 4)
    rec0 = decode(grid[pi:pi + 1])
    rec1 = decode(t)
    ims = [render2(to_grid(rec0), max_px=300), render2(to_grid(rec1), max_px=300)]
    h = max(im.height for im in ims)
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 8, h), (255, 255, 255))
    sheet.paste(ims[0], (0, 0))
    sheet.paste(ims[1], (ims[0].width + 8, 0))
    sheet.save(f"{OUT}/perturb_multi_{pi}.png")
    print(f"[perturb-multi] -> {OUT}/perturb_multi_{pi}.png", flush=True)


if __name__ == "__main__":
    main()
