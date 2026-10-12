"""VQ-VAE evaluation: reconstruction, codebook usage, sampling, interpolation.

Renders evaluation sheets into out_lib/vq_eval/ for vision QA:
  recon_a.png / recon_b.png   original vs reconstruction (2 views each pair)
  interp_<pair>.png           latent walks (5 steps between two buildings)
  samples.png                 8 decodes of random code sequences

usage: python src/vq_eval.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402
from train_vq import VQVAE  # noqa: E402

CLASS_TO_ID = {0: 0, 1: 1, 2: 5, 3: 20, 4: 18, 5: 35}
OUT = "out_lib/vq_eval"


def to_grid(x: np.ndarray) -> np.ndarray:
    """class-index volume (32^3) -> legacy-id grid for the renderer."""
    g = np.zeros(x.shape, dtype=np.uint16)
    for c, i in CLASS_TO_ID.items():
        g[x == c] = i
    return g


def sheet(images, path, gap=8):  # noqa: ANN001
    """Paste PIL images horizontally with a gap."""
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    from PIL import Image
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, (h - im.height) // 2))
        x += im.width + gap
    out.save(path)
    print(f"[eval] -> {path}", flush=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    dev = "cpu"
    data = np.load("out_lib/vq_data.npz")
    X = data["X"]
    idx = data["idx"]
    ck = torch.load("out_lib/vq_ckpt.pt", map_location=dev, weights_only=True)
    model = VQVAE().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    rng = np.random.default_rng(7)

    def enc_dec(vol):  # noqa: ANN001
        xb = torch.from_numpy(np.moveaxis(
            np.eye(6, dtype=np.float32)[vol][None], -1, 1).copy()).to(dev)
        with torch.no_grad():
            rec, _ = model(xb)
        return rec.argmax(1)[0].numpy()

    # --- reconstruction on random items -------------------------------
    picks = rng.choice(len(X), 2, replace=False)
    accs = []
    for pi in picks:
        rec = enc_dec(X[pi])
        accs.append((rec == X[pi]).mean())
        sheet([render2(to_grid(X[pi]), max_px=300),
               render2(to_grid(rec), max_px=300)],
              os.path.join(OUT, f"recon_{int(pi)}.png"))
    print(f"[eval] recon voxel acc: {np.mean(accs):.3f}", flush=True)

    # --- codebook usage ----------------------------------------------
    with torch.no_grad():
        z = model.enc(torch.from_numpy(np.moveaxis(
            np.eye(6, dtype=np.float32)[X[:64]], -1, 1).copy()).to(dev))
        _, indices, _ = model.vq(z.reshape(64, -1, z.shape[1]))
    used = len(torch.unique(indices))
    print(f"[eval] codebook used: {used}/1024 on 64 items", flush=True)

    # --- interpolation between two buildings -------------------------
    a, b = int(picks[0]), int(picks[1])
    za = model.enc(torch.from_numpy(np.moveaxis(
        np.eye(6, dtype=np.float32)[X[a:a + 1]], -1, 1).copy()).to(dev))
    zb = model.enc(torch.from_numpy(np.moveaxis(
        np.eye(6, dtype=np.float32)[X[b:b + 1]], -1, 1).copy()).to(dev))
    imgs = []
    for t in np.linspace(0, 1, 5):
        zt = (1 - t) * za + t * zb
        with torch.no_grad():
            rec = model.dec(zt).argmax(1)[0].numpy()
        imgs.append(render2(to_grid(rec), max_px=260))
    sheet(imgs, os.path.join(OUT, f"interp_{a}_{b}.png"))

    # --- random codebook walks (prior-free sanity) --------------------
    imgs = []
    with torch.no_grad():
        for k in range(8):
            zr = torch.randn(1, 64, 8, 8, 8, device=dev)
            zr = zr.permute(0, 2, 3, 4, 1).reshape(1, -1, 64)
            q, _, _ = model.vq(zr)
            q = q.reshape(1, 8, 8, 8, 64).permute(0, 4, 1, 2, 3)
            rec = model.dec(q).argmax(1)[0].numpy()
            imgs.append(render2(to_grid(rec), max_px=200))
    sheet(imgs, os.path.join(OUT, "samples.png"))
    print("[eval] done", flush=True)


if __name__ == "__main__":
    main()
