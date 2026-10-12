"""Learn room-type → furniture-type associations from ProcTHOR.

Builds a mapping: which furniture goes in which room type.
Output: room_furniture_rules.json

usage: python src/procthor_room_furniture.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np


def main():
    plans_file = "out_lib/plans/procthor_plans.jsonl"
    furn_file = "out_lib/plans/procthor_furniture.jsonl"
    attrs_file = "out_lib/plans/procthor_attrs.jsonl"
    out_json = "out_lib/room_furniture_rules.json"

    # Load plans (room types and positions)
    plans = []
    with open(plans_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            plans.append(json.loads(line))

    # Load furniture
    furs = []
    with open(furn_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            furs.append(json.loads(line))

    print(f"[room_furn] {len(plans)} plans, {len(furs)} furniture sets")

    # For each house, match furniture to nearest room
    room_furniture = defaultdict(Counter)  # room_type -> furniture_type -> count
    room_size_dist = defaultdict(list)     # room_type -> [area values]

    matched = 0
    for plan, furn in zip(plans, furs):
        rooms = plan.get("rooms", [])
        objects = furn.get("objects", [])
        if not rooms or not objects:
            continue

        for obj in objects:
            asset = obj.get("assetId", "")
            furn_type = asset.split("_")[0] if "_" in asset else asset
            pos = obj.get("position", {})
            ox, oz = pos.get("x", 0), pos.get("z", 0)

            # Find nearest room
            best_room = None
            best_dist = float("inf")
            for room in rooms:
                rect = room.get("rect")
                if not rect:
                    continue
                x0, z0, x1, z1 = rect
                # Check if point is inside room rect
                if x0 <= ox <= x1 and z0 <= oz <= z1:
                    best_room = room
                    break
                # Distance to room
                cx = (x0 + x1) / 2
                cz = (z0 + z1) / 2
                dist = ((ox - cx) ** 2 + (oz - cz) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_room = room

            if best_room:
                room_type = best_room.get("type", "unknown")
                room_furniture[room_type][furn_type] += 1
                matched += 1

        # Room size stats
        for room in rooms:
            rect = room.get("rect")
            if rect:
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                room_size_dist[room.get("type", "unknown")].append(area)

    print(f"[room_furn] matched {matched} furniture items to rooms")

    # Report
    print(f"\n  Room types found:")
    for rt in sorted(room_furniture.keys()):
        total = sum(room_furniture[rt].values())
        n_rooms = len(room_size_dist.get(rt, []))
        print(f"    {rt:15s}: {total:6d} items, {n_rooms} rooms")

    print(f"\n  Furniture per room type (top 10):")
    for rt in ["Bedroom", "Kitchen", "Bathroom", "LivingRoom", "unknown"]:
        if rt not in room_furniture:
            continue
        top = room_furniture[rt].most_common(10)
        items = ", ".join(f"{f}({c})" for f, c in top)
        print(f"    {rt:15s}: {items}")

    # Room size stats
    print(f"\n  Room size (area in blocks²):")
    for rt in sorted(room_size_dist.keys()):
        sizes = room_size_dist[rt]
        if sizes:
            arr = np.array(sizes)
            print(f"    {rt:15s}: mean={arr.mean():.1f}, median={np.median(arr):.1f}")

    # Save rules
    rules = {}
    for rt, counter in room_furniture.items():
        total = sum(counter.values())
        rules[rt] = {
            "furniture_distribution": {f: c / total for f, c in counter.most_common(30)},
            "top_furniture": dict(counter.most_common(15)),
            "n_items": total,
            "room_size_mean": float(np.mean(room_size_dist.get(rt, [0]))),
        }

    json.dump({"room_furniture_rules": rules, "n_houses_analyzed": len(plans)},
              open(out_json, "w"), indent=1)
    print(f"\n[room_furn] saved: {out_json}")


if __name__ == "__main__":
    main()
