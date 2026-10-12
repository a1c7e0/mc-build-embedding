"""Render author-portfolio sheets to pick per-district style anchors.

Authors with >= N builds get one contact sheet each (blind: numbered tiles).
Output: out_lib/authors/sheet_<author>.png + authors.json (sheet -> idx list)
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:24]


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    min_b = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    outdir = os.path.join(lib, "authors")
    os.makedirs(outdir, exist_ok=True)

    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    by_url = {e["url"]: e for e in
              json.load(open("data/schematicsWithFinalUrl.json",
                             encoding="utf-8")) if isinstance(e, dict)}
    users = json.load(open(os.path.join(lib, "types2.json")))  # just reuse load
    types = users

    groups: dict[str, list[int]] = {}
    for r in meta:
        u = by_url.get(r.get("url", ""), {}).get("user", "")
        if u:
            groups.setdefault(u, []).append(r["idx"])
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ranked = [(u, idxs) for u, idxs in ranked if len(idxs) >= min_b][:top_n]

    manifest = {}
    for u, idxs in ranked:
        idxs = [i for i in idxs if types[str(i)] != "sprite"][:16]
        tiles, caps = [], []
        for k, i in enumerate(idxs):
            g = np.load(os.path.join(lib, meta[i]["file"]))["grid"]
            tiles.append(render(g, max_px=280))
            caps.append(f"#{k + 1}")
        name = slug(u)
        sheet(tiles, caps).save(os.path.join(outdir, f"sheet_{name}.png"))
        manifest[name] = {"user": u, "idxs": idxs}
        print(f"[authors] {u}: {len(idxs)} builds -> sheet_{name}.png",
              flush=True)
    json.dump(manifest, open(os.path.join(outdir, "authors.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
