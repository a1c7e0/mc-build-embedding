"""Prepare categorical voxel tensors for VQ-VAE training.

Crops every library building to its bounding box, rescales (nearest) to fit
30^3, centres in 32^3, and maps legacy block ids to 6 material classes:
air / mineral / wood / glass / organic / colourful.

Output: out_lib/vq_data.npz  X uint8 (N, 32, 32, 32) class indices + idx list.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

SIZE = 32
FIT = 30

WOOD = {5, 17, 58, 85, 96, 107, 126, 134, 143, 162, 163, 164, 183, 184, 185,
        186, 187, 188, 189, 190, 191, 192, 200, 219}
GLASS = {20, 95, 102, 160, 171, 231, 241}
ORGANIC = {6, 18, 31, 37, 38, 39, 40, 59, 78, 83, 86, 106, 111, 175, 161, 106}
MINERAL = {1, 2, 3, 4, 7, 12, 13, 14, 15, 16, 22, 24, 35, 41, 42, 43, 44, 45,
           48, 49, 56, 57, 79, 80, 82, 87, 89, 91, 97, 98, 99, 101, 112, 113,
           121, 155, 156, 159, 168, 169, 172, 179, 201, 202, 206, 215, 216,
           232, 251}

CLASS_OF = {}
for _i in MINERAL:
    CLASS_OF[_i] = 1
for _i in WOOD:
    CLASS_OF[_i] = 2
for _i in GLASS:
    CLASS_OF[_i] = 3
for _i in ORGANIC:
    CLASS_OF[_i] = 4


def classify(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i in np.unique(arr):
        if i == 0:
            continue
        out[arr == i] = CLASS_OF.get(int(i), 1 if i < 35 else 5)
    return out


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    xs, keep = [], []
    for r in meta:
        g = np.load(os.path.join(lib, r["file"]))["grid"]
        g = classify(g)
        nz = np.nonzero(g)
        if not len(nz[0]):
            continue
        x0 = [int(a.min()) for a in nz]
        x1 = [int(a.max()) + 1 for a in nz]
        crop = g[x0[0]:x1[0], x0[1]:x1[1], x0[2]:x1[2]]
        f = [FIT / max(s, 1) for s in crop.shape]
        rz = zoom(crop, f, order=0)
        out = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        o = [(SIZE - s) // 2 for s in rz.shape]
        out[o[0]:o[0] + rz.shape[0],
            o[1]:o[1] + rz.shape[1],
            o[2]:o[2] + rz.shape[2]] = rz
        xs.append(out)
        keep.append(r["idx"])
    X = np.stack(xs)
    np.savez_compressed(os.path.join(lib, "vq_data.npz"), X=X,
                        idx=np.array(keep))
    frac = (X > 0).mean()
    print(f"[prep] {X.shape} saved -> {lib}/vq_data.npz "
          f"fill={frac:.3f} classes={sorted(set(np.unique(X).tolist()))}",
          flush=True)


if __name__ == "__main__":
    main()
