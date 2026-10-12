"""Render furniture motifs clearly: standalone motif showcase + closeup interior."""
from __future__ import annotations
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from render import render2

# Load motifs
data = json.load(open("out_lib/deco_motifs_noskel.json"))
motifs = [m for m in data["motifs"] if m.get("n_types", 0) >= 2 and m["count"] >= 50]
print(f"showcasing {len(motifs)} motifs")

# Render each motif as 3x3x3 isolated block (add padding for visibility)
renders = []
for i, m in enumerate(motifs[:12]):
    pattern = np.array(m["pattern"], dtype=np.uint16).reshape(3, 3, 3)
    # Pad to 5x5x5 for better visibility
    padded = np.zeros((5, 5, 5), dtype=np.uint16)
    padded[1:4, 1:4, 1:4] = pattern
    img = render2(padded, max_px=120, cutaway=False)
    renders.append((img, m))

# Grid: 4 cols x 3 rows
cols, rows = 4, 3
cw, ch = 130, 150
sheet = Image.new("RGB", (cols * cw, rows * ch), (245, 245, 245))
from PIL import ImageDraw
draw = ImageDraw.Draw(sheet)
for idx, (img, m) in enumerate(renders):
    r, c = divmod(idx, cols)
    x, y = c * cw + 5, r * ch + 20
    sheet.paste(img, (x, y))
    blocks = {k: v for k, v in m["blocks"].items() if k != "0"}
    label = f"{m['count']}x " + ",".join(f"{k}" for k in list(blocks.keys())[:3])
    draw.text((x, y + 110), label, fill=(0, 0, 0))

out = "out_lib/motif_showcase.png"
sheet.save(out)
print(f"saved {out}")

# Also: big closeup of one building interior
grid = np.load("out_lib/buildings/b00021.npz")["grid"].copy()
# Apply some motifs manually at known interior positions
sys.path.insert(0, "src")
from apply_motifs import load_motifs, find_interior_air, place_motif
import random
rng = random.Random(42)
by_style, all_furn = load_motifs()
interior = find_interior_air(grid)
print(f"{len(interior)} interior positions")
placed = 0
for x, y, z in interior[:30]:
    if placed >= 5:
        break
    motif = all_furn[rng.randint(0, len(all_furn) - 1)]
    mx, my, mz = max(0, x-1), max(0, y-1), max(0, z-1)
    if place_motif(grid, motif, mx, my, mz):
        placed += 1

# Big render of the interior
img = render2(grid, max_px=800, cutaway=True)
img.save("out_lib/interior_closeup.png")
print(f"saved out_lib/interior_closeup.png")
