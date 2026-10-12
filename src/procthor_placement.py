"""Learn furniture placement rules from ProcTHOR: position relative to room,
distance to walls, orientation distribution. Output: placement rules for decoration.

usage: python src/procthor_placement.py [out_json]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "out_lib/procthor_placement.json"
    furn_file = "out_lib/plans/procthor_furniture.jsonl"

    print("[placement] analyzing ProcTHOR furniture positions...")

    # Statistics: position relative to room, wall distance, height
    wall_distances = []       # distance from nearest wall
    height_distribution = []  # y position
    item_types = Counter()    # assetId prefix
    room_positions = Counter()  # near_wall / center / corner
    orientations = Counter()  # rotation y value

    with open(furn_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            furn = json.loads(line)
            for item in furn.get("objects", []):
                asset = item.get("assetId", "")
                pos = item.get("position", {})
                rot = item.get("rotation", {})

                # Classify item type (first word before _)
                item_type = asset.split("_")[0] if "_" in asset else asset
                item_types[item_type] += 1

                # Height (y position)
                y = pos.get("y", 0)
                height_distribution.append(y)

                # Orientation (rotation around y)
                ry = rot.get("y", 0)
                # Quantize to 4 cardinal directions
                cardinal = int(round(ry / 90)) % 4
                orientations[cardinal] += 1

                # Wall distance approximation
                # ProcTHOR positions are in meters within the house
                # Items near walls have small x or z values
                x, z = pos.get("x", 0), pos.get("z", 0)
                dist_to_wall = min(x, z)  # rough approximation
                wall_distances.append(dist_to_wall)

                if dist_to_wall < 1.0:
                    room_positions["near_wall"] += 1
                elif dist_to_wall > 3.0:
                    room_positions["center"] += 1
                else:
                    room_positions["mid"] += 1

    n = len(wall_distances)
    print(f"[placement] {n} furniture items analyzed")

    # Summarize
    wd = np.array(wall_distances)
    hd = np.array(height_distribution)

    print(f"\n  Wall distance: mean={wd.mean():.2f}m, median={np.median(wd):.2f}m, "
          f"std={wd.std():.2f}")
    print(f"  Height (y): mean={hd.mean():.2f}m, median={np.median(hd):.2f}m")

    print(f"\n  Position distribution:")
    for k, v in room_positions.most_common():
        print(f"    {k:12s}: {v:6d} ({v/n*100:.1f}%)")

    print(f"\n  Orientation (cardinal):")
    dir_names = ["north(0°)", "east(90°)", "south(180°)", "west(270°)"]
    for i in range(4):
        c = orientations.get(i, 0)
        print(f"    {dir_names[i]:12s}: {c:6d} ({c/n*100:.1f}%)")

    print(f"\n  Top 20 furniture types:")
    for t, c in item_types.most_common(20):
        print(f"    {t:20s}: {c:6d}")

    # Height bands (floor vs table vs wall)
    floor_items = int((hd < 0.5).sum())
    table_items = int(((hd >= 0.5) & (hd < 1.2)).sum())
    wall_items = int((hd >= 1.2).sum())
    print(f"\n  Height bands:")
    print(f"    floor (y<0.5):    {floor_items:6d} ({floor_items/n*100:.1f}%)")
    print(f"    table (0.5-1.2):  {table_items:6d} ({table_items/n*100:.1f}%)")
    print(f"    wall (y>=1.2):    {wall_items:6d} ({wall_items/n*100:.1f}%)")

    # Save placement rules
    rules = {
        "wall_distance_mean": float(wd.mean()),
        "wall_distance_median": float(np.median(wd)),
        "height_mean": float(hd.mean()),
        "height_median": float(np.median(hd)),
        "position_distribution": dict(room_positions),
        "orientation_distribution": {str(k): v for k, v in orientations.items()},
        "height_bands": {"floor": floor_items, "table": table_items, "wall": wall_items},
        "top_item_types": dict(item_types.most_common(50)),
    }
    json.dump(rules, open(out_json, "w"), indent=1)
    print(f"\n[placement] saved: {out_json}")


if __name__ == "__main__":
    main()
