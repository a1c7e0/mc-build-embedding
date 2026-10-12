"""Audit the anchor authors' items one sheet at a time (type truth for pools).

Renders each anchor author's items (max 16/author) as zoomed sheets so the
vision pass can give per-item types; results are written by hand afterwards
into out_lib/authors/type_overrides.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402

ANCHORS = ["massEntropy", "Smon", "GREGON", "UTD Projects", "xV3NoMxHD",
           "Breakc0re", "Tronic", "Leefield"]


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    outdir = os.path.join(lib, "authors")
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    by_url = {e["url"]: e for e in
              json.load(open("data/schematicsWithFinalUrl.json",
                             encoding="utf-8")) if isinstance(e, dict)}
    manifest = json.load(open(os.path.join(outdir, "authors.json"),
                              encoding="utf-8"))
    for name, info in manifest.items():
        if only and only not in name:
            continue
        user = info["user"]
        if user not in ANCHORS:
            continue
        idxs = info["idxs"]
        tiles, caps = [], []
        for k, i in enumerate(idxs):
            g = np.load(os.path.join(lib, meta[i]["file"]))["grid"]
            tiles.append(render(g, max_px=420))
            caps.append(f"#{k + 1}")
        sheet(tiles, caps, cols=4, cell=460).save(
            os.path.join(outdir, f"audit_{name}.png"))
        print(f"{name}: {len(idxs)} items idxs={idxs}", flush=True)


if __name__ == "__main__":
    main()
