"""Render a random sample sheet from a building library (quality check)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402

lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                    encoding="utf-8")]
rng = np.random.default_rng(seed)
picks = rng.choice(len(meta), size=min(n, len(meta)), replace=False)
tiles, caps = [], []
for k, i in enumerate(picks):
    g = np.load(os.path.join(lib, meta[int(i)]["file"]))["grid"]
    tiles.append(render(g, max_px=280))
    caps.append(f"#{k + 1} {meta[int(i)]['dims']}")
sheet(tiles, caps).save(os.path.join(lib, "sample_sheet.png"))
for k, i in enumerate(picks):
    print(f"#{k + 1} dims={meta[int(i)]['dims']} "
          f"desc={meta[int(i)].get('desc', '')[:80]}")
print(f"-> {lib}/sample_sheet.png")
