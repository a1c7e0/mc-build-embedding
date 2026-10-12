"""Extract window decoration patterns from MC building corpus.

Identifies windows (glass blocks in walls) and extracts the decoration
around them: frames, sills, shutters, flower boxes, curtains.

Output: window decoration motifs + frame patterns per style.

usage: python src/window_deco.py [n_buildings] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

GLASS_BLOCKS = {20, 102}  # glass, glass_pane
WINDOW_SIZE = 5  # scan radius around window


def find_windows(grid):
    """Find window positions: glass blocks in vertical wall surfaces."""
    W, H, D = grid.shape
    windows = []
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            for z in range(1, D - 1):
                if grid[x, y, z] not in GLASS_BLOCKS:
                    continue
                # Check if it's in a wall (has solid blocks on opposite sides)
                horizontal_solid = (
                    (grid[x-1, y, z] > 0 and grid[x+1, y, z] > 0) or
                    (grid[x, y, z-1] > 0 and grid[x, y, z+1] > 0)
                )
                if horizontal_solid:
                    windows.append((x, y, z))
    return windows


def extract_window_context(grid, wx, wy, wz, radius=2):
    """Extract the block context around a window position."""
    W, H, D = grid.shape
    x0, x1 = max(0, wx - radius), min(W, wx + radius + 1)
    y0, y1 = max(0, wy - radius), min(H, wy + radius + 1)
    z0, z1 = max(0, wz - radius), min(D, wz + radius + 1)

    # Get the surrounding blocks as a 3D patch
    patch = grid[x0:x1, y0:y1, z0:z1]
    return patch, (x0, y0, z0)


def extract_window_frames(grid, windows):
    """Extract window frame patterns (blocks immediately surrounding glass)."""
    W, H, D = grid.shape
    frames = Counter()
    window_contexts = []

    for wx, wy, wz in windows:
        # Frame: the 8 blocks surrounding the window in the wall plane
        # Determine wall orientation
        if grid[wx-1, wy, wz] > 0 and grid[wx+1, wy, wz] > 0:
            # Wall runs along z-axis, window faces x
            # Frame = blocks at (wx, wy±1, wz±1) etc
            frame_blocks = []
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dy == 0 and dz == 0:
                        continue
                    bx, by, bz = wx, wy + dy, wz + dz
                    if 0 <= bx < W and 0 <= by < H and 0 <= bz < D:
                        frame_blocks.append(grid[bx, by, bz])
        else:
            # Wall runs along x-axis, window faces z
            frame_blocks = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    bx, by, bz = wx + dx, wy + dy, wz
                    if 0 <= bx < W and 0 <= by < H and 0 <= bz < D:
                        frame_blocks.append(grid[bx, by, bz])

        # Record the frame pattern (sorted tuple of block IDs)
        frame_key = tuple(sorted(frame_blocks))
        frames[frame_key] += 1

        # Also record full context (3x3x3 around window)
        patch, offset = extract_window_context(grid, wx, wy, wz)
        window_contexts.append({
            "pos": (wx, wy, wz),
            "frame": frame_key,
            "patch_shape": patch.shape,
        })

    return frames, window_contexts


def main():
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_json = sys.argv[2] if len(sys.argv) > 2 else "out_lib/window_deco.json"

    lib_dir = "out_lib"
    meta = [json.loads(l) for l in open(f"{lib_dir}/meta.jsonl", encoding="utf-8")]
    vtypes = json.load(open(f"{lib_dir}/vision_types.json", encoding="utf-8"))

    global_frames = Counter()
    style_frames = defaultdict(Counter)
    total_windows = 0
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

        windows = find_windows(g)
        if not windows:
            continue

        frames, _ = extract_window_frames(g, windows)
        total_windows += len(windows)
        processed += 1

        for frame_key, count in frames.items():
            global_frames[frame_key] += count
            style_frames[style][frame_key] += count

    print(f"[window] {processed} buildings, {total_windows} windows found")

    print(f"\n  Top 15 window frame patterns:")
    for frame_key, count in global_frames.most_common(15):
        blocks = Counter(frame_key)
        bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
        print(f"    {count:4d}x  [{len(frame_key)} blocks] {bs}")

    print(f"\n  Window frames per style (top 5):")
    for style in ["modern", "medieval", "asian", "scifi", "western", "nautical"]:
        if style not in style_frames:
            continue
        top = style_frames[style].most_common(5)
        print(f"    {style}:")
        for frame_key, count in top:
            blocks = Counter(frame_key)
            bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(4))
            print(f"      {count:4d}x  {bs}")

    # Save
    top_frames = []
    for frame_key, count in global_frames.most_common(100):
        best_style = "other"
        best_c = 0
        for s, counter in style_frames.items():
            if frame_key in counter and counter[frame_key] > best_c:
                best_c = counter[frame_key]
                best_style = s
        top_frames.append({
            "frame_blocks": list(frame_key),
            "count": count,
            "style": best_style,
            "block_distribution": dict(Counter(frame_key)),
        })

    json.dump({"n_buildings": processed, "total_windows": total_windows,
               "frame_patterns": top_frames}, open(out_json, "w"))
    print(f"\n[window] saved: {out_json} ({len(top_frames)} patterns)")


if __name__ == "__main__":
    main()
