"""Render samples from a vq_data npz for visual QA.

usage: python src/render_vq_samples.py data.npz out_prefix [n]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

CLASS_TO_ID = {0: 0, 1: 1, 2: 5, 3: 20, 4: 18, 5: 35}


def main() -> None:
    data_f = sys.argv[1]
    prefix = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    X = np.load(data_f)["X"][:n]
    ims = []
    for i, x in enumerate(X):
        g = np.zeros(x.shape, dtype=np.uint16)
        for c, b in CLASS_TO_ID.items():
            g[x == c] = b
        ims.append(render2(g, max_px=260))
    w = sum(im.width for im in ims) + 8 * (len(ims) - 1)
    h = max(im.height for im in ims)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    xoff = 0
    for im in ims:
        sheet.paste(im, (xoff, 0))
        xoff += im.width + 8
    sheet.save(prefix)
    print(f"[render-samples] {len(ims)} -> {prefix}", flush=True)


if __name__ == "__main__":
    main()
