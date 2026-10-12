"""Render before/after decoration comparison."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from render import render2

base = "out_lib/plans/demo/procthor_0.npz"
dec = "out_lib/plans/demo/procthor_0_decorated.npz"

g0 = np.load(base)["grid"] if "grid" in np.load(base) else np.load(base)["world"]
g1 = np.load(dec)["grid"] if "grid" in np.load(dec) else np.load(dec)["world"]

img0 = render2(g0, max_px=280, cutaway=True)
img1 = render2(g1, max_px=280, cutaway=True)

from PIL import Image
w = img0.width + img1.width + 10
h = max(img0.height, img1.height)
sheet = Image.new("RGB", (w, h), (255, 255, 255))
sheet.paste(img0, (0, 0))
sheet.paste(img1, (img0.width + 10, 0))
sheet.save("out_lib/plans/demo/procthor_0_decor_compare.png")
print("saved procthor_0_decor_compare.png (left=before, right=after)")
