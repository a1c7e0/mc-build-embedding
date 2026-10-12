"""Small learned encoder baseline: MLP autoencoder over 24^3 occupancy.

Question: does a learned 256-D latent beat the handcrafted+PCA space in
Finding 1-4? CPU-only, tiny on purpose.

Emits: out/ae_latent.npz (Z256 train+val), out/ae_loss.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

SIDE = 24


def load_grids(out_dir: str) -> tuple[torch.Tensor, list[dict]]:
    meta = [json.loads(l) for l in
            open(os.path.join(out_dir, "meta.jsonl"), encoding="utf-8")]
    xs = []
    for rec in meta:
        g = np.load(os.path.join(out_dir, rec["file"]))["grid"]
        occ = (g != 0).astype(np.float32)
        dx, dy, dz = occ.shape
        idx = [np.minimum(((np.arange(SIDE) + 0.5) * d / SIDE).astype(int),
                          d - 1) for d in (dx, dy, dz)]
        xs.append(occ[np.ix_(*idx)].ravel())
    return torch.from_numpy(np.vstack(xs)), meta


class AE(nn.Module):
    def __init__(self, dim: int, latent: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, dim), nn.Sigmoid())

    def forward(self, x):  # noqa: ANN001
        return self.dec(self.enc(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(42)
    X, meta = load_grids(args.out)
    n = X.shape[0]
    n_val = max(64, n // 10)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"[ae] n={n} dim={X.shape[1]} train={len(tr_idx)} val={len(val_idx)}",
          flush=True)

    model = AE(X.shape[1], args.latent)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    t0 = time.time()
    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(args.epochs):
        model.train()
        tr = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(tr), args.batch):
            xb = X[tr[i:i + args.batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(X[val_idx]), X[val_idx]))
        trl = tot / len(tr_idx)
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[ae] ep{ep + 1:03d} train={trl:.4f} val={vl:.4f} "
                  f"({(ep + 1) / (time.time() - t0):.2f} ep/s)", flush=True)
        if bad >= 8:
            print(f"[ae] early stop at ep{ep + 1} (best val={best:.4f})",
                  flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Z = model.enc(X).numpy()
    np.savez_compressed(os.path.join(args.out, "ae_latent.npz"), Z256=Z)
    torch.save(model.state_dict(), os.path.join(args.out, "ae_model.pt"))
    with open(os.path.join(args.out, "ae_loss.json"), "w") as fh:
        json.dump({"best_val_bce": best, "epochs": args.epochs,
                   "seconds": round(time.time() - t0)}, fh, indent=2)
    print(f"[ae] done best_val={best:.4f} "
          f"elapsed={time.time() - t0:.0f}s -> out/ae_latent.npz", flush=True)


if __name__ == "__main__":
    main()
