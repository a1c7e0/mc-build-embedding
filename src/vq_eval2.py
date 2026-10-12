"""VQ evaluation v2: honest metrics (accuracy is banned — air dominates).

Metrics per batch:
  occupied IoU        intersection/union of (pred>0) vs (true>0)
  non-air accuracy    class accuracy over occupied cells only
  per-class IoU       IoU of each of the 6 classes
  CC delta            |connected components pred - true| on occupied masks
  wall-face F1        F1 on boundary faces (solid<->air) position match

usage: python src/vq_eval2.py [ckpt] [data.npz] [n]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(__file__))
from train_vq import VQVAE  # noqa: E402


def occ_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def faces(mask: np.ndarray) -> np.ndarray:
    f = np.zeros(mask.shape, dtype=bool)
    for ax in (0, 1, 2):
        shifted = np.roll(mask, 1, axis=ax)
        shifted[0 if ax == 0 else slice(None), 0 if ax == 1 else slice(None),
                0 if ax == 2 else slice(None)] = False
        f |= mask ^ shifted
    return f


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "out_lib/vq_ckpt.pt"
    dataf = sys.argv[2] if len(sys.argv) > 2 else "out_lib/vq_data2.npz"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    dev = "cpu"
    m = VQVAE().to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    m.load_state_dict(sd["model"])
    m.eval()
    X = np.load(dataf)["X"][:n]
    ious, naccs, ccds, ff1 = [], [], [], []
    cls_iou = [[] for _ in range(6)]
    with torch.no_grad():
        for i in range(0, len(X), 8):
            xb = torch.from_numpy(np.moveaxis(
                np.eye(6, dtype=np.float32)[X[i:i + 8]], -1, 1).copy()
            ).to(dev)
            pred = m(xb)[0].argmax(1).numpy()
            for p, t in zip(pred, X[i:i + 8]):
                po, to = p > 0, t > 0
                ious.append(occ_iou(po, to))
                occ = po | to
                if occ.any():
                    naccs.append(float((p[occ] == t[occ]).mean()))
                ccds.append(abs(label(po)[1] - label(to)[1]))
                pf, tf = faces(po), faces(to)
                tp = float((pf & tf).sum())
                prec = tp / max(pf.sum(), 1)
                rec = tp / max(tf.sum(), 1)
                ff1.append(2 * prec * rec / max(prec + rec, 1e-9))
                for c in range(6):
                    cls_iou[c].append(occ_iou(p == c, t == c))
    print(f"[eval2] {dataf} ckpt={ckpt} n={len(X)}")
    print(f"  occupied IoU   {np.mean(ious):.3f}")
    print(f"  non-air acc    {np.mean(naccs):.3f}")
    print(f"  wall-face F1   {np.mean(ff1):.3f}")
    print(f"  CC |delta|     {np.mean(ccds):.2f}")
    for c in range(6):
        print(f"  class{c} IoU    {np.mean(cls_iou[c]):.3f}")


if __name__ == "__main__":
    main()
