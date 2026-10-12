"""Motif-based decoration: apply multi-block furniture patterns to skeleton shells.

Uses deco_motifs_noskel.json as template library. Places complete 3x3x3
furniture motifs (not single blocks) at interior/decoration positions.

usage: python src/apply_motifs.py <grid.npz> <style> <seed> <out.npz>
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

AIR = 0
MOTIF = 3


def load_motifs(path="out_lib/deco_motifs_noskel.json"):
    data = json.load(open(path, encoding="utf-8"))
    motifs = data.get("motifs", [])
    # Group by style
    by_style = {}
    for m in motifs:
        s = m.get("style", "other")
        if s not in by_style:
            by_style[s] = []
        by_style[s].append(m["pattern"])
    # Also keep all mixed-type motifs
    all_furn = [m["pattern"] for m in motifs if m.get("n_types", 0) >= 2]
    return by_style, all_furn


def find_interior_air(grid):
    """Find interior air positions (enclosed by solid on most sides)."""
    W, H, D = grid.shape
    solid = grid > 0
    # Simple: air with >= 3 solid neighbors = interior
    interior = []
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            for z in range(1, D - 1):
                if grid[x, y, z] != 0:
                    continue
                neighbors = sum(1 for dx, dy, dz in
                    [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                    if grid[x+dx, y+dy, z+dz] > 0)
                if neighbors >= 3:
                    interior.append((x, y, z))
    return interior


def place_motif(grid, motif_pattern, x, y, z):
    """Place a 3x3x3 motif at position (x,y,z) = lower corner.
    Only overwrite air blocks. Returns True if placed."""
    W, H, D = grid.shape
    if x + MOTIF > W or y + MOTIF > H or z + MOTIF > D:
        return False

    # Check target area is at least 50% air (so we're placing furniture in space)
    target = grid[x:x+MOTIF, y:y+MOTIF, z:z+MOTIF]
    n_air = int((target == 0).sum())
    if n_air < MOTIF**3 * 0.3:  # need at least 30% air
        return False

    # Place: overwrite air with non-air motif blocks
    placed = 0
    pattern = np.array(motif_pattern, dtype=np.uint16).reshape(MOTIF, MOTIF, MOTIF)
    for dx in range(MOTIF):
        for dy in range(MOTIF):
            for dz in range(MOTIF):
                if pattern[dx, dy, dz] != 0 and grid[x+dx, y+dy, z+dz] == 0:
                    grid[x+dx, y+dy, z+dz] = pattern[dx, dy, dz]
                    placed += 1
    return placed > 0


def main():
    if len(sys.argv) < 5:
        print("usage: python src/apply_motifs.py <grid.npz> <style> <seed> <out.npz>")
        sys.exit(1)

    grid_file = sys.argv[1]
    style = sys.argv[2]
    seed = int(sys.argv[3])
    out_file = sys.argv[4]

    rng = random.Random(seed)
    npz = np.load(grid_file)
    grid = (npz["world"] if "world" in npz else npz["grid"]).copy()
    g_orig = grid.copy()

    # Colorful/visible blocks that make good furniture
    COLORFUL_BLOCKS = {35, 41, 42, 47, 50, 54, 58, 85, 86, 89, 96,
                       102, 155, 159, 171, 173, 20, 5, 17, 18}
    STRUCTURAL_BORING = {1, 3, 4, 12, 13, 24, 43, 45, 98}

    by_style, all_furn = load_motifs()
    # FILTER: prefer motifs with colorful/visible blocks, not just gray stone
    def is_colorful(pattern):
        blocks = set(pattern) - {0}
        has_color = bool(blocks & COLORFUL_BLOCKS)
        has_boring = bool(blocks & STRUCTURAL_BORING)
        return has_color and not (has_boring and len(blocks) < 3)

    all_furn = [m for m in all_furn if is_colorful(m)]
    for s in by_style:
        by_style[s] = [m for m in by_style[s] if is_colorful(m)]
    motifs = by_style.get(style, all_furn)
    if not motifs:
        motifs = all_furn

    print(f"[motif_deco] grid {grid.shape}, style={style}, "
          f"{len(motifs)} motifs available")

    interior = find_interior_air(grid)
    print(f"[motif_deco] {len(interior)} interior air positions")

    placed_count = 0
    # Try to place motifs at interior positions
    for x, y, z in interior:
        if placed_count >= 20:  # limit to avoid clutter (increased)
            break
        if rng.random() > 0.6:  # 60% chance per position (increased visibility)
            continue
        motif = motifs[rng.randint(0, len(motifs) - 1)]
        # Position motif so its center is near this interior point
        mx = max(0, min(x - MOTIF // 2, grid.shape[0] - MOTIF))
        my = max(0, min(y - MOTIF // 2, grid.shape[1] - MOTIF))
        mz = max(0, min(z - MOTIF // 2, grid.shape[2] - MOTIF))
        if place_motif(grid, motif, mx, my, mz):
            placed_count += 1

    print(f"[motif_deco] placed {placed_count} furniture motifs")
    np.savez_compressed(out_file, grid=grid)

    # Comparison render
    from PIL import Image
    img0 = render2(g_orig, max_px=280, cutaway=True)
    img1 = render2(grid, max_px=280, cutaway=True)
    w = img0.width + img1.width + 10
    h = max(img0.height, img1.height)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    sheet.paste(img0, (0, 0))
    sheet.paste(img1, (img0.width + 10, 0))
    cmp = out_file.replace(".npz", "_compare.png")
    sheet.save(cmp)
    print(f"[motif_deco] comparison: {cmp}")


if __name__ == "__main__":
    main()
