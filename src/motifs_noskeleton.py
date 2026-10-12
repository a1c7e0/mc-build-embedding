"""Extract decoration motifs by REMOVING the structural skeleton first.

Strategy:
1. Detect the building shell (walls, floor, roof) using flood fill from outside
2. Remove shell blocks → only interior decoration remains
3. Extract 3x3x3 motifs from remaining decoration blocks

This gives furniture-like patterns instead of wall/floor patterns.

usage: python src/motifs_noskeleton.py [n_buildings] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

AIR = 0
WATER = 9
MOTIF = 3


def detect_shell(grid):
    """Return boolean mask of structural shell blocks (walls+floor+roof).
    Shell = blocks reachable from outside via 6-connectivity."""
    W, H, D = grid.shape
    solid = grid > 0
    shell = np.zeros_like(solid)

    # Flood fill from outside: mark all air reachable from boundary
    from collections import deque
    outside = np.zeros_like(solid)
    q = deque()
    # Start from all boundary air
    for x in range(W):
        for z in range(D):
            for y in (0, H - 1):
                if not solid[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    q.append((x, y, z))
    for x in range(W):
        for y in range(H):
            for z in (0, D - 1):
                if not solid[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    q.append((x, y, z))
    for y in range(H):
        for z in range(D):
            for x in (0, W - 1):
                if not solid[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    q.append((x, y, z))

    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                if not solid[nx, ny, nz] and not outside[nx, ny, nz]:
                    outside[nx, ny, nz] = True
                    q.append((nx, ny, nz))

    # Shell = solid blocks adjacent to outside air (outer walls, floor, roof)
    for x in range(W):
        for y in range(H):
            for z in range(D):
                if not solid[x, y, z]:
                    continue
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                        if outside[nx, ny, nz]:
                            shell[x, y, z] = True
                            break
                    else:
                        shell[x, y, z] = True  # boundary = shell
                        break

    return shell, outside


def extract_motifs_noskeleton(grid, shell, outside):
    """Extract motifs from blocks that are NOT shell (interior decoration).
    Also include motifs from "surface" positions (shell blocks that face interior air)."""
    W, H, D = grid.shape
    solid = grid > 0

    # Interior decoration = solid blocks NOT in shell (embedded in walls, furniture)
    # OR solid blocks in shell but adjacent to interior air (wall decorations, trim)
    decoration = solid & ~shell

    # Also get "wall decorations" = shell blocks adjacent to interior air
    interior_air = solid == 0
    # Interior air = air NOT reachable from outside
    interior_air = interior_air & ~outside

    wall_deco = np.zeros_like(solid)
    for x in range(W):
        for y in range(H):
            for z in range(D):
                if not shell[x, y, z] or not solid[x, y, z]:
                    continue
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                        if interior_air[nx, ny, nz]:
                            wall_deco[x, y, z] = True
                            break

    # Extract 3x3x3 motifs centered on decoration blocks
    motifs = Counter()
    STRIDE = 2

    for x in range(0, W - MOTIF + 1, STRIDE):
        for y in range(0, H - MOTIF + 1, STRIDE):
            for z in range(0, D - MOTIF + 1, STRIDE):
                patch = grid[x:x+MOTIF, y:y+MOTIF, z:z+MOTIF]
                deco_patch = (decoration | wall_deco)[x:x+MOTIF, y:y+MOTIF, z:z+MOTIF]

                # At least 2 decoration blocks in the patch
                if int(deco_patch.sum()) < 2:
                    continue
                # At least 1 non-air block
                n_solid = int((patch > 0).sum())
                if n_solid == 0 or n_solid == MOTIF ** 3:
                    continue

                mk = tuple(patch.flatten().tolist())
                motifs[mk] += 1

    return motifs


def main():
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_json = sys.argv[2] if len(sys.argv) > 2 else "out_lib/deco_motifs_noskel.json"

    lib_dir = "out_lib"
    meta = [json.loads(l) for l in open(f"{lib_dir}/meta.jsonl", encoding="utf-8")]
    vtypes = json.load(open(f"{lib_dir}/vision_types.json", encoding="utf-8"))

    global_motifs = Counter()
    style_motifs = defaultdict(Counter)
    processed = 0

    for i, r in enumerate(meta):
        if processed >= n_limit:
            break
        idx = str(r["idx"])
        vt = vtypes.get(idx, {})
        if vt.get("type") != "building":
            continue
        style = vt.get("style", "other")
        if style not in ("modern", "medieval", "asian", "scifi", "western", "nautical"):
            style = "other"
        gpath = os.path.join(lib_dir, r["file"])
        if not os.path.exists(gpath):
            continue
        g = np.load(gpath)["grid"]
        W, H, D = g.shape
        if W < 5 or H < 5 or D < 5:
            continue

        shell, outside = detect_shell(g)
        motifs = extract_motifs_noskeleton(g, shell, outside)
        processed += 1

        for mk, c in motifs.items():
            global_motifs[mk] += c
            style_motifs[style][mk] += c

        if processed % 20 == 0:
            print(f"[motif] {processed}/{n_limit}", flush=True)

    print(f"\nprocessed {processed} buildings, {len(global_motifs)} unique motifs")

    # Top furniture-like motifs: 2+ distinct non-air blocks, frequent
    print(f"\nTop 25 decoration motifs (skeleton removed):")
    for mk, c in global_motifs.most_common(25):
        blocks = Counter(mk)
        n_types = len(set(mk) - {0})
        bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
        marker = " [FURN]" if n_types >= 2 and c >= 20 else ""
        print(f"  {c:4d}x [{n_types}t]{marker}  {bs}")

    print(f"\nFurniture-like (>=2 non-air types, count>=20):")
    furn = [(mk, c) for mk, c in global_motifs.most_common(500)
            if len(set(mk) - {0}) >= 2 and c >= 20]
    print(f"  {len(furn)} found")
    for mk, c in furn[:20]:
        blocks = Counter(mk)
        bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
        print(f"  {c:4d}x  {bs}")

    # Save
    top = []
    for mk, c in global_motifs.most_common(500):
        n_types = len(set(mk) - {0})
        if n_types >= 2 and c >= 10:
            best_style = "other"
            best_c = 0
            for s, counter in style_motifs.items():
                if mk in counter and counter[mk] > best_c:
                    best_c = counter[mk]
                    best_style = s
            top.append({
                "pattern": list(mk),
                "count": c,
                "n_types": n_types,
                "style": best_style,
                "blocks": dict(Counter(mk)),
            })
    json.dump({"n_buildings": processed, "total_unique": len(global_motifs),
               "motifs": top}, open(out_json, "w"))
    print(f"\nsaved {out_json} ({len(top)} motifs)")


if __name__ == "__main__":
    main()
