"""3D VQ-VAE for Minecraft buildings (categorical 32^3 voxels, 6 classes).

Quantizer: lucidrains' vector-quantize-pytorch (reused, not hand-rolled).
Trains on out_lib/vq_data.npz; checkpoints to out_lib/vq_ckpt.pt.

usage: python src/train_vq.py [steps]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize

CKPT = os.environ.get("VQ_CKPT", "out_lib/vq_ckpt.pt")
LOG = os.environ.get("VQ_LOG", "docs/vq_train.log")


class ResBlock(nn.Module):
    def __init__(self, ch: int):  # noqa: ANN001
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.ReLU(), nn.Conv3d(ch, ch, 3, padding=1))

    def forward(self, x):  # noqa: ANN001
        return x + self.net(x)


class VQVAE(nn.Module):
    def __init__(self, ch: int = 64, codebook: int = 1024):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(6, ch, 4, stride=2, padding=1), ResBlock(ch),
            nn.Conv3d(ch, ch, 4, stride=2, padding=1), ResBlock(ch))
        self.vq = VectorQuantize(dim=ch, codebook_size=codebook,
                                 decay=0.99, commitment_weight=0.25)
        self.dec = nn.Sequential(
            ResBlock(ch),
            nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1), ResBlock(ch),
            nn.ConvTranspose3d(ch, ch * 2, 4, stride=2, padding=1),
            nn.ReLU(), nn.Conv3d(ch * 2, 6, 3, padding=1))

    def forward(self, x):  # noqa: ANN001
        z = self.enc(x)                       # (B, 64, s, s, s)
        b, c = z.shape[:2]
        sp = z.shape[2:]
        zl = z.permute(0, 2, 3, 4, 1).reshape(b, -1, c)   # (B, s^3, 64)
        q, _, loss = self.vq(zl)              # channel_last sequence form
        q = q.reshape(b, *sp, c).permute(0, 4, 1, 2, 3)
        return self.dec(q), loss


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    dev = "cpu"
    data = np.load(os.environ.get("VQ_DATA", "out_lib/vq_data.npz"))
    X = data["X"]
    N = len(X)
    model = VQVAE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=dev, weights_only=True)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start = sd["step"]
        print(f"[vq] resumed from step {start}", flush=True)
    else:
        start = 0
    log = open(LOG, "a", encoding="utf-8")
    t0 = time.time()
    rng = np.random.default_rng(0)
    model.train()
    for step in range(start, steps):
        bi = rng.integers(0, N, 8)
        # one-hot (N,32,32,32,6) -> channel axis second
        xb = torch.from_numpy(
            np.moveaxis(np.eye(6, dtype=np.float32)[X[bi]], -1, 1).copy()
        ).to(dev)
        recon, vq_loss = model(xb)
        target = torch.from_numpy(X[bi].astype(np.int64)).to(dev)
        rec_loss = F.cross_entropy(recon, target)
        loss = rec_loss + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                pred = recon.argmax(1).numpy()
                iou = ((pred > 0) == (X[bi] > 0)).mean()
            msg = (f"[vq] step {step} loss {loss.item():.3f} "
                   f"rec {rec_loss.item():.3f} vq {vq_loss.item():.3f} "
                   f"vox_acc {iou:.3f} ({time.time()-t0:.0f}s)")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if step % 200 == 0 and step > start:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1}, CKPT)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": start + steps}, CKPT)
    print(f"[vq] done, saved {CKPT}", flush=True)


if __name__ == "__main__":
    main()
