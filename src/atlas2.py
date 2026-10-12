"""V0.2 judgment tool: encode a dataset, audit codebook usage, render atlas.

argv: ckpt data.npz outdir
Builds tokens from the data (VQ encoder), reports usage/perplexity, and
renders the codebook atlas (each code decoded alone in a sea of itself).
Same question as codebook_atlas: do codes look like building PARTS yet?

usage: python src/atlas2.py out_lib/vq_ckpt2.pt out_lib/vq_data2.npz out_lib/atlas2
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

CLASS_TO_ID = {0: 0, 1: 1, 2: 5, 3: 20, 4: 18, 5: 35}


def to_grid(x: np.ndarray) -> np.ndarray:
    g = np.zeros(x.shape, dtype=np.uint16)
    for c, i in CLASS_TO_ID.items():
        g[x == c] = i
    return g


def main() -> None:
    ckpt_f, data_f, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    dev = "cpu"
    m = VQVAE().to(dev)
    sd = torch.load(ckpt_f, map_location=dev, weights_only=True)
    m.load_state_dict(sd["model"])
    m.eval()
    cb = getattr(getattr(m.vq, "_codebook", None), "embed", None)
    if cb is None:
        cb = m.vq.codebook.embed
    if cb.dim() == 3:
        cb = cb[0]
    vocab = cb.shape[0]

    X = np.load(data_f)["X"]
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 8):
            xb = torch.from_numpy(np.moveaxis(
                np.eye(6, dtype=np.float32)[X[i:i + 8]], -1, 1).copy()
            ).to(dev)
            z = m.enc(xb)
            zl = z.permute(0, 2, 3, 4, 1).reshape(z.shape[0], -1, z.shape[1])
            _, codes, _ = m.vq(zl)
            outs.append(codes.cpu().numpy())
    tokens = np.concatenate(outs)
    lat = int(round(len(tokens[0]) ** (1 / 3)))
    np.savez_compressed(os.path.join(out, "tokens.npz"), tokens=tokens)

    c = np.bincount(tokens.reshape(-1), minlength=vocab)
    used = int((c > 0).sum())
    p = c[c > 0] / c.sum()
    perplexity = float(np.exp(-(p * np.log(p)).sum()))
    print(f"[atlas2] tokens {tokens.shape} lat {lat}^3 vocab {vocab} "
          f"used {used} perplexity {perplexity:.1f}", flush=True)

    order = np.argsort(-c)
    thumbs = []
    with torch.no_grad():
        for k in range(vocab):
            code = int(order[k])
            vol = torch.full((1, lat, lat, lat), code, dtype=torch.long,
                             device=dev)
            q = cb[vol].permute(0, 4, 1, 2, 3)
            rec = m.dec(q).argmax(1)[0].numpy()
            thumbs.append(render2(to_grid(rec), max_px=72))
            if k % 256 == 0:
                print(f"[atlas2] {k}/{vocab}", flush=True)
    cell = 76
    per = 32
    sheet = Image.new("RGB", (cell * per, cell * per), (255, 255, 255))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % per) * cell, (k // per) * cell))
    sheet.save(os.path.join(out, "atlas_v2.png"))
    print(f"[atlas2] -> {out}/atlas_v2.png (usage-sorted)", flush=True)


if __name__ == "__main__":
    main()
