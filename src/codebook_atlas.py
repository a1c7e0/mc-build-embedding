"""Codebook atlas: decode every VQ code in isolation and tile the renders.

De-mystifies the VQ-VAE: what does each of the 1024 "words" look like?
Codes are sorted by usage (vq_tokens.npz), so the atlas front-loads what
the prior actually speaks with. Outputs atlas sheets to out_lib/atlas/.

usage: python src/codebook_atlas.py
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

    tok = np.load("out_lib/vq_tokens.npz")["tokens"]
    counts = np.bincount(tok.reshape(-1), minlength=vocab)
    order = np.argsort(-counts)          # most used first

    thumbs = []
    with torch.no_grad():
        for k in range(vocab):
            code = int(order[k])
            vol = torch.full((1, 8, 8, 8), code, dtype=torch.long, device=dev)
            q = cb[vol].permute(0, 4, 1, 2, 3)
            rec = vq.dec(q).argmax(1)[0].numpy()
            im = render2(to_grid(rec), max_px=72)
            if counts[code] == 0:
                im = im.convert("L").convert("RGB")    # grey = unused code
            thumbs.append(im)
            if k % 128 == 0:
                print(f"[atlas] {k}/{vocab}", flush=True)

    cell = 72 + 4
    per = 32                              # 32 x 32 = 1024 per sheet... one sheet
    sheet = Image.new("RGB", (cell * per, cell * per), (255, 255, 255))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % per) * cell, (k // per) * cell))
    sheet.save(os.path.join(OUT, "atlas_all.png"))
    print(f"[atlas] -> {OUT}/atlas_all.png "
          f"(row-major, sorted by usage; grey = never used)", flush=True)
    top = ", ".join(f"{int(order[i])}:{int(counts[order[i]])}"
                    for i in range(12))
    print(f"[atlas] top codes by usage: {top}", flush=True)


if __name__ == "__main__":
    main()
