"""Architectural-skeleton metrics on the building library.

For every library building (raw voxel grid) compute:
  enclosure     interior air pockets (not reachable from outside) / inside air
                -> high = closed rooms (houses), low = open structures
  wall_gap      perimeter columns with no solid block in the middle band
                -> high = missing walls / broken enclosure
  roof_cover    top-band coverage over the footprint
  base_cover    bottom-band coverage (grounded?)

Output: out_lib/skeleton.json  {idx: {...}}
usage: python src/skeleton_metrics.py [lib]
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np


def flood_outside(air: np.ndarray) -> np.ndarray:
    """Mark air voxels reachable from the bbox boundary (6-connectivity)."""
    nx, ny, nz = air.shape
    seen = np.zeros_like(air, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for x in range(nx):
        for z in range(nz):
            for y in (0, ny - 1):
                if air[x, y, z] and not seen[x, y, z]:
                    seen[x, y, z] = True
                    q.append((x, y, z))
    for x in range(nx):
        for y in range(ny):
            for z in (0, nz - 1):
                if air[x, y, z] and not seen[x, y, z]:
                    seen[x, y, z] = True
                    q.append((x, y, z))
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = x + dx, y + dy, z + dz
            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz \
                    and air[a, b, c] and not seen[a, b, c]:
                seen[a, b, c] = True
                q.append((a, b, c))
    return seen


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    out = {}
    for k, r in enumerate(meta):
        g = np.load(os.path.join(lib, r["file"]))["grid"]
        solid = g > 0
        if not solid.any():
            continue
        # close doors/windows by dilation before flood: interior of a house
        # must not leak to the outside through its openings
        solid_c = solid.copy()
        for ax in (0, 1, 2):
            solid_c |= np.roll(solid_c, 1, axis=ax) | np.roll(solid_c, -1, axis=ax)
        air = ~solid_c
        outside = flood_outside(air)
        # v3: the pocket is RAW air sealed by the dilated shell (rooms with
        # doors), measured against RAW walls — roof+floor alone score low,
        # pillared pavilions leak, real rooms close. (v2 measured the eroded
        # core against the shell: degenerate, everything = 1.0)
        interior = (~solid) & ~outside
        air_total = int(air.sum())
        # enclosure v2: boundary-closure ratio — of the interior's boundary
        # faces, how many touch real walls (roofs+floor alone must not count)
        if interior.any():
            neigh = closed_f = 0
            for ax in (0, 1, 2):
                for s in (1, -1):
                    sh = np.zeros_like(solid_c)
                    sh_i = np.zeros_like(interior)
                    if s == 1:
                        sh[1:] = solid_c[:-1]
                        sh_i[1:] = interior[:-1]
                    else:
                        sh[:-1] = solid_c[1:]
                        sh_i[:-1] = interior[1:]
                    face = interior & ~sh_i
                    neigh += int(face.sum())
                    closed_f += int((face & sh).sum())
            enclosure = closed_f / max(neigh, 1)
        else:
            enclosure = 0.0
        # wall continuity: perimeter footprint columns, mid height band
        foot = solid.any(axis=1)                       # (x, z)
        ys = np.nonzero(solid.any(axis=(0, 2)))[0]
        y0, y1 = int(ys.min()), int(ys.max())
        band = solid[:, y0 + max((y1 - y0) // 4, 1): y0 + max(
            (y1 - y0) * 3 // 4, 2), :].any(axis=1)     # (x, z)
        per = foot.copy()
        per[1:-1, 1:-1] = False
        wall_gap = float((per & ~band).sum() / max(int(per.sum()), 1))
        top_band = solid[:, max(y1 - 1, 0):, :].any(axis=1)
        roof_cover = float((top_band & foot).sum() / max(int(foot.sum()), 1))
        bot_band = solid[:, :y0 + 2, :].any(axis=1)
        base_cover = float((bot_band & foot).sum() / max(int(foot.sum()), 1))
        out[str(r["idx"])] = {
            "enclosure": round(enclosure, 3),
            "wall_gap": round(wall_gap, 3),
            "roof_cover": round(roof_cover, 3),
            "base_cover": round(base_cover, 3),
            "interior_voxels": int(interior.sum())}
        if k % 500 == 0:
            print(f"[skeleton] {k}/{len(meta)}", flush=True)
    json.dump(out, open(os.path.join(lib, "skeleton.json"), "w"))
    enc = np.array([v["enclosure"] for v in out.values()])
    gap = np.array([v["wall_gap"] for v in out.values()])
    print(f"[skeleton] done {len(out)} items | enclosure "
          f"med {np.median(enc):.3f} | wall_gap med {np.median(gap):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
