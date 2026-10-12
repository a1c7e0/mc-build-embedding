"""Convert stylegen outputs to vq_data format with exact style labels.

Reads out_lib/stylegen/<style>/<style>_v*.npz (compiled voxel shells),
maps legacy block IDs back to 6-class, saves as a vq_data-compatible
npz with styles array (0=modern 1=medieval 2=asian ... per STYLES order).

usage: python src/stylegen_to_vq.py [stylegen_dir] [out.npz]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import zoom

STYLE_TO_IDX = {"chinese_temple": 2, "japanese_house": 2, "gothic_church": 1,
                "nautical_dock": 5}  # asian=2, medieval=1, nautical=5
LEGACY_TO_CLASS = {0: 0, 1: 1, 5: 2, 18: 4, 20: 3, 35: 5}  # mineral/wood/glass/organic/colourful
SIZE = 32
FIT = 30


def resize_to_32(g: np.ndarray) -> np.ndarray:
    """Crop to bbox, aspect-preserving scale to fit 30^3, centre-pad to 32^3."""
    nz = np.nonzero(g)
    if not len(nz[0]):
        return np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
    crop = g[nz[0].min():nz[0].max() + 1,
             nz[1].min():nz[1].max() + 1,
             nz[2].min():nz[2].max() + 1]
    f = FIT / max(crop.shape)
    rz = zoom(crop, (f, f, f), order=0)
    if any(s > SIZE for s in rz.shape):
        rz = rz[:SIZE, :SIZE, :SIZE]
    out = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
    o = [(SIZE - s) // 2 for s in rz.shape]
    out[o[0]:o[0] + rz.shape[0],
        o[1]:o[1] + rz.shape[1],
        o[2]:o[2] + rz.shape[2]] = rz
    return out


def main() -> None:
    sdir = sys.argv[1] if len(sys.argv) > 1 else "out_lib/stylegen"
    out = sys.argv[2] if len(sys.argv) > 2 else "out_lib/vq_data_stylegen.npz"
    xs, styles, idxs = [], [], []
    for style_name in sorted(os.listdir(sdir)):
        style_dir = os.path.join(sdir, style_name)
        if not os.path.isdir(style_dir):
            continue
        si = STYLE_TO_IDX.get(style_name, 7)
        for fn in sorted(os.listdir(style_dir)):
            if not fn.endswith(".npz"):
                continue
            g = np.load(os.path.join(style_dir, fn))["grid"]
            cls = np.zeros(g.shape, dtype=np.uint8)
            for lid, cid in LEGACY_TO_CLASS.items():
                cls[g == lid] = cid
            xs.append(resize_to_32(cls))
            styles.append(si)
            idxs.append(f"{style_name}:{fn}")
    X = np.stack(xs)
    np.savez_compressed(out, X=X, styles=np.array(styles, dtype=np.int32),
                        idx=np.array(idxs))
    print(f"[stylegen2vq] {X.shape} -> {out} "
          f"style_counts={np.bincount(styles, minlength=8).tolist()}")


if __name__ == "__main__":
    main()
