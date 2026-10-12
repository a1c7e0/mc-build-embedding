"""Style-aware decoration applicator.

Takes a compiled skeleton shell and fills decoration blocks
(exterior trim, interior furnishing, roof details) according
to style+position distributions learned from MC corpus.

usage: python src/apply_decoration.py <skeleton.json> <grid.npz> <style> <out.npz>
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
WATER = 9

# Decoration block palette per style (block_id → weight)
# Derived from MC corpus analysis: surface/exterior decorations
EXTERIOR_DECO = {
    "modern":    {35: 0.35, 42: 0.15, 20: 0.15, 102: 0.10, 89: 0.08,
                  85: 0.07, 41: 0.05, 55: 0.03, 50: 0.02},
    "medieval":  {4: 0.25, 98: 0.20, 1: 0.15, 5: 0.12, 17: 0.10,
                  35: 0.05, 85: 0.05, 89: 0.03, 50: 0.03, 45: 0.02},
    "asian":     {5: 0.25, 17: 0.15, 24: 0.15, 98: 0.10, 35: 0.08,
                  18: 0.08, 85: 0.07, 89: 0.05, 2: 0.04, 12: 0.03},
    "scifi":     {42: 0.25, 35: 0.15, 89: 0.15, 55: 0.10, 20: 0.10,
                  101: 0.08, 41: 0.07, 102: 0.05, 43: 0.03, 50: 0.02},
    "western":   {24: 0.25, 4: 0.18, 159: 0.15, 12: 0.10, 1: 0.08,
                  35: 0.06, 85: 0.06, 43: 0.05, 89: 0.04, 2: 0.03},
    "nautical":  {5: 0.25, 17: 0.15, 43: 0.12, 42: 0.10, 35: 0.08,
                  85: 0.08, 89: 0.07, 41: 0.05, 20: 0.05, 50: 0.05},
    "other":     {35: 0.20, 1: 0.15, 42: 0.10, 89: 0.10, 20: 0.10,
                  85: 0.08, 41: 0.07, 102: 0.07, 5: 0.08, 50: 0.05},
}

# Interior decoration (furniture-ish blocks)
INTERIOR_DECO = {
    "modern":    {35: 0.20, 42: 0.15, 20: 0.12, 5: 0.12, 89: 0.10,
                  47: 0.08, 54: 0.06, 61: 0.05, 171: 0.06, 102: 0.06},
    "medieval":  {5: 0.22, 47: 0.15, 17: 0.12, 4: 0.10, 35: 0.08,
                  85: 0.08, 89: 0.07, 58: 0.06, 54: 0.06, 50: 0.06},
    "asian":     {5: 0.20, 17: 0.15, 35: 0.12, 47: 0.10, 85: 0.10,
                  89: 0.08, 18: 0.08, 50: 0.07, 54: 0.06, 58: 0.04},
    "scifi":     {42: 0.20, 89: 0.15, 55: 0.12, 35: 0.10, 20: 0.10,
                  41: 0.08, 102: 0.08, 50: 0.07, 61: 0.06, 54: 0.04},
    "western":   {5: 0.20, 24: 0.15, 35: 0.12, 47: 0.10, 4: 0.08,
                  85: 0.08, 89: 0.07, 54: 0.07, 58: 0.07, 50: 0.06},
    "nautical":  {5: 0.22, 17: 0.15, 42: 0.12, 35: 0.10, 85: 0.10,
                  89: 0.08, 54: 0.07, 47: 0.06, 50: 0.06, 43: 0.04},
    "other":     {35: 0.18, 5: 0.15, 42: 0.10, 89: 0.10, 47: 0.10,
                  20: 0.08, 85: 0.08, 54: 0.07, 50: 0.07, 61: 0.07},
}

# Roof decoration
ROOF_DECO = {
    "modern":    {35: 0.30, 89: 0.20, 42: 0.15, 20: 0.10, 41: 0.08,
                  50: 0.07, 102: 0.05, 85: 0.05},
    "medieval":  {4: 0.22, 98: 0.18, 35: 0.12, 89: 0.12, 17: 0.10,
                  50: 0.08, 85: 0.08, 4: 0.10},
    "asian":     {35: 0.22, 5: 0.15, 89: 0.15, 17: 0.12, 24: 0.10,
                  50: 0.10, 85: 0.08, 18: 0.08},
    "scifi":     {89: 0.25, 42: 0.20, 55: 0.12, 35: 0.12, 20: 0.10,
                  50: 0.08, 41: 0.08, 101: 0.05},
    "western":   {24: 0.20, 35: 0.15, 4: 0.12, 89: 0.12, 159: 0.10,
                  50: 0.10, 85: 0.10, 43: 0.11},
    "nautical":  {5: 0.20, 35: 0.15, 89: 0.15, 42: 0.12, 85: 0.10,
                  50: 0.10, 41: 0.08, 17: 0.10},
    "other":     {35: 0.22, 89: 0.18, 42: 0.12, 5: 0.10, 20: 0.10,
                  50: 0.10, 85: 0.09, 41: 0.09},
}


def pick_block(weights, rng):
    """Pick a block ID based on weight distribution."""
    blocks = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    # Weighted random selection
    r = rng.random()
    cumulative = 0.0
    for block, prob in zip(blocks, probs):
        cumulative += prob
        if r < cumulative:
            return int(block)
    return int(blocks[-1])


def find_positions(grid):
    """Find exterior walls, interior air, roof surface, floor surface."""
    W, H, D = grid.shape
    solid = grid > 0
    # Exterior = solid blocks with at least one air neighbor on the outside
    exterior = set()
    interior_air = []
    roof_surface = []
    floor_surface = []

    for x in range(W):
        for y in range(H):
            for z in range(D):
                if grid[x, y, z] == 0:
                    # Check if interior (surrounded by solid on most sides)
                    neighbors_solid = 0
                    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                            if grid[nx, ny, nz] > 0:
                                neighbors_solid += 1
                    if neighbors_solid >= 3:
                        interior_air.append((x, y, z))
                    # Check roof: air above solid
                    if y > 0 and grid[x, y-1, z] > 0 and y == H - 2:
                        roof_surface.append((x, y, z))
                    # Check floor: air below solid at ground
                    if y == 1 and grid[x, 0, z] > 0:
                        floor_surface.append((x, y, z))

    return exterior, interior_air, roof_surface, floor_surface


def apply_decoration(grid, style, rng, density=0.15):
    """Apply style-appropriate decoration to a skeleton shell.
    density: fraction of eligible positions to fill (0-1)."""
    W, H, D = grid.shape
    ext_palette = EXTERIOR_DECO.get(style, EXTERIOR_DECO["other"])
    int_palette = INTERIOR_DECO.get(style, INTERIOR_DECO["other"])
    roof_palette = ROOF_DECO.get(style, ROOF_DECO["other"])

    _, interior_air, roof_surface, floor_surface = find_positions(grid)

    placed = {"exterior": 0, "interior": 0, "roof": 0, "floor": 0}

    # Exterior decoration: place on solid blocks at surface (walls)
    # Find solid blocks with air on at least one horizontal side
    for x in range(W):
        for y in range(1, H - 1):
            for z in range(D):
                if grid[x, y, z] == 0:
                    continue
                # Check if this is a surface block (has air neighbor horizontally)
                is_surface = False
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,0,1),(0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                        if grid[nx, ny, nz] == 0:
                            is_surface = True
                            break
                if is_surface and rng.random() < density * 0.5:
                    # Sometimes replace with decoration block
                    grid[x, y, z] = pick_block(ext_palette, rng)
                    placed["exterior"] += 1

    # Interior decoration: fill some interior air with furniture blocks
    for x, y, z in interior_air:
        if rng.random() < density:
            grid[x, y, z] = pick_block(int_palette, rng)
            placed["interior"] += 1

    # Roof decoration: place on top surface
    for x, y, z in roof_surface:
        if rng.random() < density * 1.5:  # roof is visible, more decoration
            grid[x, y, z] = pick_block(roof_palette, rng)
            placed["roof"] += 1

    # Floor decoration: carpets etc on floor surface
    for x, y, z in floor_surface:
        if rng.random() < density * 0.8:
            grid[x, y, z] = pick_block(int_palette, rng)
            placed["floor"] += 1

    return placed


def main():
    if len(sys.argv) < 5:
        print("usage: python src/apply_decoration.py <grid.npz> <style> <seed> <out.npz>")
        sys.exit(1)

    grid_file = sys.argv[1]
    style = sys.argv[2]
    seed = int(sys.argv[3])
    out_file = sys.argv[4]

    rng = random.Random(seed)

    npz = np.load(grid_file)
    grid = (npz["world"] if "world" in npz else npz["grid"]).copy()

    print(f"[deco] grid {grid.shape}, style={style}, seed={seed}")
    placed = apply_decoration(grid, style, rng, density=0.15)
    print(f"[deco] placed: {placed}")

    np.savez_compressed(out_file, grid=grid)
    print(f"[deco] saved: {out_file}")

    # Render comparison
    from PIL import Image
    g_orig = (npz["world"] if "world" in npz else npz["grid"]).copy()
    img0 = render2(g_orig, max_px=280, cutaway=True)
    img1 = render2(grid, max_px=280, cutaway=True)
    w = img0.width + img1.width + 10
    h = max(img0.height, img1.height)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    sheet.paste(img0, (0, 0))
    sheet.paste(img1, (img0.width + 10, 0))
    compare_path = out_file.replace(".npz", "_compare.png")
    sheet.save(compare_path)
    print(f"[deco] comparison: {compare_path}")


if __name__ == "__main__":
    main()
