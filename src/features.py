"""Interpretable per-building feature vectors from voxel grids.

Feature groups:
  geometry   - bbox dims, aspect, fill ratio, footprint, column height stats
  material   - block-class histogram (wood/stone/plant/glass/colourful/...)
  profile    - vertical mass distribution (5 bands)
  symmetry   - mirror symmetry along x and z
  colour     - mean colour + 8-bin hue histogram over block colours
Output: out/features.npz (X float32, names list in out/feature_names.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import block_color  # noqa: E402


def _add(cls_map: dict[int, str], cls: str, ids: list[int]) -> None:
    for i in ids:
        cls_map[i] = cls


ID_CLASS: dict[int, str] = {}
_add(ID_CLASS, "liquid", [8, 9, 10, 11])
_add(ID_CLASS, "plant", [6, 18, 31, 32, 37, 38, 39, 40, 59, 81, 83, 86, 103,
                         104, 105, 106, 111, 115, 141, 142, 161, 170, 175])
_add(ID_CLASS, "glass", [20, 95, 102, 160])
_add(ID_CLASS, "wood", [5, 17, 25, 47, 54, 58, 64, 66, 68, 72, 84, 85, 96,
                        107, 125, 126, 130, 134, 135, 136, 143, 146, 162,
                        163, 164, 183, 184, 185, 186, 187, 188, 189, 190,
                        191, 192, 193, 194, 195, 196, 219])
_add(ID_CLASS, "colourful", [35, 171, 46, 41, 57, 133, 152, 22, 21, 24])
_add(ID_CLASS, "metal", [26, 27, 28, 29, 33, 34, 42, 69, 71, 76, 101, 117,
                         145, 147, 148, 154, 157, 158, 167, 216])
CLASSES = ["wood", "metal", "plant", "glass", "colourful", "liquid"]


def class_of(bid: int) -> str:
    return ID_CLASS.get(bid, "mineral")


def hue_of(rgb: tuple[int, int, int]) -> float | None:
    r, g, b = [c / 255.0 for c in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.04:
        return None  # achromatic
    if mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    return h / 6.0


def shape_block(grid: np.ndarray, t: int = 12) -> np.ndarray:
    """Coarse occupancy signature: t^3 point-sampled voxel occupancy plus the
    three orthographic silhouette projections (t*t each)."""
    occ = (grid != 0).astype(np.float32)
    dx, dy, dz = occ.shape
    xs = np.minimum(((np.arange(t) + 0.5) * dx / t).astype(int), dx - 1)
    ys = np.minimum(((np.arange(t) + 0.5) * dy / t).astype(int), dy - 1)
    zs = np.minimum(((np.arange(t) + 0.5) * dz / t).astype(int), dz - 1)
    small = occ[np.ix_(xs, ys, zs)]
    proj_xy = small.mean(axis=2).ravel()
    proj_zy = small.mean(axis=0).ravel()
    proj_xz = small.mean(axis=1).ravel()
    return np.concatenate([small.ravel(), proj_xy, proj_zy, proj_xz])


def features_for(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    dx, dy, dz = grid.shape
    occ = grid != 0
    n = int(occ.sum())
    vol = dx * dy * dz

    ids = grid[occ]
    cls_hist = np.zeros(len(CLASSES) + 1)  # +1 = mineral
    for bid, cnt in zip(*np.unique(ids, return_counts=True)):
        c = class_of(int(bid))
        idx = CLASSES.index(c) if c in CLASSES else len(CLASSES)
        cls_hist[idx] += cnt
    cls_frac = cls_hist / max(n, 1)

    # vertical mass distribution in 5 bands
    ys = np.nonzero(occ)[1] if n else np.zeros(1)
    band = np.histogram(ys, bins=5, range=(0, dy))[0].astype(float)
    band /= max(n, 1)

    # column heights over the footprint
    col_h = occ.sum(axis=1).astype(float)  # [x, z]
    col_occ = (col_h > 0).astype(float)
    ch_mean = float(col_h.mean()) / dy
    ch_std = float(col_h.std()) / dy
    ch_max = float(col_h.max()) / dy
    ch_med = float(np.median(col_h)) / dy
    footprint = float(col_occ.mean())
    top_var = float(col_h[col_h > 0].std()) / dy if col_occ.any() else 0.0

    sym_x = float((occ == occ[::-1, :, :]).mean())
    sym_z = float((occ == occ[:, :, ::-1]).mean())

    # colour stats over block palette (cached)
    cols = np.array([block_color(int(b)) for b in ids], dtype=float) / 255.0
    rgb_mean = cols.mean(axis=0) if len(cols) else np.zeros(3)
    rgb_std = cols.std(axis=0) if len(cols) else np.zeros(3)
    hue_hist = np.zeros(8)
    sat_mean = 0.0
    for row in cols:
        mx, mn = float(row.max()), float(row.min())
        sat_mean += (mx - mn)
        h = hue_of(tuple(row))
        if h is not None:
            hue_hist[min(int(h * 8), 7)] += 1
    hue_hist /= max(n, 1)
    sat_mean /= max(n, 1)

    # surface ratio
    pad = np.zeros((dx + 2, dy + 2, dz + 2), dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = occ
    empty = ~pad
    surface = occ & (
        empty[:-2, 1:-1, 1:-1] | empty[2:, 1:-1, 1:-1] |
        empty[1:-1, :-2, 1:-1] | empty[1:-1, 2:, 1:-1] |
        empty[1:-1, 1:-1, :-2] | empty[1:-1, 1:-1, 2:])
    surface_ratio = float(surface.sum()) / max(n, 1)

    values = list(cls_frac) + [
        np.log1p(dx), np.log1p(dy), np.log1p(dz),
        dx / max(dz, 1), dy / max(max(dx, dz), 1),
        np.log1p(n), n / max(vol, 1),
        ch_mean, ch_std, ch_max, ch_med, footprint, top_var,
        sym_x, sym_z, surface_ratio,
        *band,
        *rgb_mean, *rgb_std, sat_mean, *hue_hist,
    ]
    names = [f"frac_{c}" for c in CLASSES + ["mineral"]] + [
        "log_dx", "log_dy", "log_dz", "aspect_xz", "aspect_y",
        "log_n", "fill", "col_mean", "col_std", "col_max", "col_med",
        "footprint", "top_var", "sym_x", "sym_z", "surface",
        *[f"band_{i}" for i in range(5)],
        "rgb_r", "rgb_g", "rgb_b", "rgb_r_std", "rgb_g_std", "rgb_b_std",
        "sat", *[f"hue_{i}" for i in range(8)],
    ]
    assert len(values) == len(names), (len(values), len(names))
    shape = shape_block(grid)
    names += [f"shape_{i}" for i in range(len(shape))]
    return np.concatenate([np.asarray(values, dtype=np.float32), shape]), names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    meta = [json.loads(l) for l in
            open(os.path.join(args.out, "meta.jsonl"), encoding="utf-8")]
    rows, names = [], None
    t0 = time.time()
    for i, rec in enumerate(meta):
        z = np.load(os.path.join(args.out, rec["file"]))
        f, names = features_for(z["grid"])
        rows.append(f)
        if (i + 1) % 200 == 0:
            print(f"[features] {i + 1}/{len(meta)} "
                  f"({(i + 1) / (time.time() - t0):.1f}/s)", flush=True)
    X = np.vstack(rows)
    np.savez_compressed(os.path.join(args.out, "features.npz"), X=X)
    with open(os.path.join(args.out, "feature_names.json"), "w") as fh:
        json.dump(names, fh, indent=2)
    print(f"[features] done: X={X.shape} elapsed={time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
