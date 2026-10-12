"""Representation v0.2 prep: aspect-preserving, buildings only, deduped.

Changes vs prep_vq_data.py (audit findings):
  1. keep only vision_types type == "building" (52% of the library)
  2. uniform scale to fit 30^3 KEEPING ASPECT (v1 stretched each axis
     independently, destroying all proportions and scale)
  3. drop exact-duplicate tensors and degenerate (dim <= 1) sources
Output: out_lib/vq_data2.npz  X uint8 (N, 32, 32, 32) 6 classes + idx.

usage: python src/prep_vq_data2.py [out_lib]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(__file__))
from prep_vq_data import classify  # noqa: E402  (same 6-class fold)

SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 32
FIT = SIZE - 2


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                       encoding="utf-8")]
    vtypes = json.load(open(os.path.join(lib, "vision_types.json"),
                           encoding="utf-8"))
    xs, keep, seen = [], [], set()
    n_type = n_degen = n_dup = 0
    for r in meta:
        vt = vtypes.get(str(r["idx"]))
        if not vt or vt.get("type") != "building":
            n_type += 1
            continue
        if min(r["dims"]) <= 1:
            n_degen += 1
            continue
        g = np.load(os.path.join(lib, r["file"]))["grid"]
        g = classify(g)
        nz = np.nonzero(g)
        if not len(nz[0]):
            continue
        x0 = [int(a.min()) for a in nz]
        x1 = [int(a.max()) + 1 for a in nz]
        crop = g[x0[0]:x1[0], x0[1]:x1[1], x0[2]:x1[2]]
        f = FIT / max(crop.shape)              # ONE factor for all axes
        rz = zoom(crop, (f, f, f), order=0)
        out = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        o = [(SIZE - s) // 2 for s in rz.shape]
        out[o[0]:o[0] + rz.shape[0],
            o[1]:o[1] + rz.shape[1],
            o[2]:o[2] + rz.shape[2]] = rz
        hsh = out.tobytes()
        if hsh in seen:
            n_dup += 1
            continue
        seen.add(hsh)
        xs.append(out)
        keep.append(r["idx"])
    X = np.stack(xs)
    out_name = sys.argv[3] if len(sys.argv) > 3 else \
        f"out_lib/vq_data{'2' if SIZE == 32 else str(SIZE)}.npz"
    np.savez_compressed(out_name, X=X, idx=np.array(keep))
    frac = (X > 0).mean()
    print(f"[prep2] {X.shape} saved -> {lib}/vq_data2.npz "
          f"fill={frac:.3f} dropped: type={n_type} degen={n_degen} "
          f"dup={n_dup}", flush=True)


if __name__ == "__main__":
    main()
