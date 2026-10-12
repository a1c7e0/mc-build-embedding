"""ProcTHOR metadata harvest: attributes (#2), materials (#3), furniture (#4).

One pass over the house JSONL produces:
  out_lib/plans/procthor_attrs.jsonl    per-house condition attributes
  out_lib/plans/procthor_furniture.jsonl per-house furniture placements
  out_lib/plans/procthor_materials.json  material-name histogram
  out_lib/plans/procthor_procgen.jsonl   proceduralParameters (re-generation)
usage: python src/procthor_attrs.py [in.jsonl.gz] [out_dir] [limit]
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "out_lib/procthor/train.jsonl.gz"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out_lib/plans"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(out_dir, exist_ok=True)
    mats = Counter()
    n = 0
    with gzip.open(src, "rt") as fin, \
            open(os.path.join(out_dir, "procthor_attrs.jsonl"), "w",
                 encoding="utf-8") as fa, \
            open(os.path.join(out_dir, "procthor_furniture.jsonl"), "w",
                 encoding="utf-8") as ff, \
            open(os.path.join(out_dir, "procthor_procgen.jsonl"), "w",
                 encoding="utf-8") as fp:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            h = json.loads(line)
            rooms = h.get("rooms", [])
            walls = h.get("walls", [])
            doors = h.get("doors", [])
            windows = h.get("windows", [])
            objs = h.get("objects", [])
            pts = [p for r in rooms for p in r.get("floorPolygon", [])]
            if not pts:
                continue
            xs = [p["x"] for p in pts]
            zs = [p["z"] for p in pts]
            area = (max(xs) - min(xs)) * (max(zs) - min(zs))
            wall_len = 0.0
            for w in walls:
                poly = w.get("polygon", [])
                if len(poly) >= 2:
                    wall_len += abs(poly[1]["x"] - poly[0]["x"]) + \
                        abs(poly[1]["z"] - poly[0]["z"])
                m = (w.get("material") or {}).get("name")
                if m:
                    mats[m] += 1
            for r in rooms:
                m = (r.get("floorMaterial") or {}).get("name")
                if m:
                    mats[m] += 1
            ys = sorted({round(p["y"], 2) for r in rooms
                         for p in r.get("floorPolygon", [])})
            fa.write(json.dumps({
                "n_rooms": len(rooms),
                "room_types": [r.get("roomType") for r in rooms],
                "n_doors": len(doors),
                "n_windows": len(windows),
                "window_density": round(len(windows) / max(wall_len, 1), 3),
                "footprint_area": round(area, 1),
                "storeys": len(ys),
                "wall_height": round(max((p["y"] for w in walls
                                          for p in w.get("polygon", [])),
                                         default=0), 2),
            }, ensure_ascii=False) + "\n")
            ff.write(json.dumps({
                "objects": [{"assetId": o.get("assetId"),
                             "position": o.get("position"),
                             "rotation": o.get("rotation")} for o in objs],
            }, ensure_ascii=False) + "\n")
            if "proceduralParameters" in h:
                fp.write(json.dumps(h["proceduralParameters"],
                                    ensure_ascii=False) + "\n")
            n += 1
    json.dump(dict(mats.most_common()),
              open(os.path.join(out_dir, "procthor_materials.json"), "w"))
    print(f"[attrs] {n} houses: attrs+furniture+procgen -> {out_dir}", flush=True)
    print(f"[attrs] distinct material names: {len(mats)}", flush=True)
    print(f"[attrs] top materials: {mats.most_common(8)}", flush=True)


if __name__ == "__main__":
    main()
