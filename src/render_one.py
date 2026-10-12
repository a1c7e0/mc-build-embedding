"""Render a few individual builds at higher resolution (render QA)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402

man = json.load(open("out/sheets/manifest.json"))
meta = [json.loads(l) for l in open("out/meta.jsonl")]
for j, ent in enumerate(man["sample_0.png"][:3]):
    i = ent["idx"]
    g = np.load(os.path.join("out", meta[i]["file"]))["grid"]
    render(g, max_px=600).save(f"out/one_{j}.png")
    print(f"one_{j}.png idx={i} dims={meta[i]['dims']} "
          f"desc={meta[i]['desc'][:150]}")
