"""2D top-down plan diagrams (PIL), same visual style as
the drawing code at the end of src/procthor_to_dsl.py.

Input rooms are (type, x0, z0, x1, z1) in normalized [0,1] coords (metric
space of data.decode / metrics) and get stretched into the drawing area.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

COLORS = {"Bedroom": (120, 160, 220), "Bathroom": (120, 200, 200),
          "LivingRoom": (220, 180, 120), "Kitchen": (180, 220, 120)}
FALLBACK = (200, 200, 200)


def render_plan(rooms, path: str, px: int = 360, title: str | None = None) -> str:
    margin = 14
    top = 26 if title else margin
    im = Image.new("RGB", (px, px + (top - margin)), (250, 250, 245))
    dr = ImageDraw.Draw(im)
    if title:
        dr.text((margin, 6), title, fill=(30, 30, 30))
    w = px - 2 * margin
    for typ, x0, z0, x1, z1 in rooms:
        a = margin + max(0.0, min(1.0, x0)) * w
        b = top + max(0.0, min(1.0, z0)) * w
        c = margin + max(0.0, min(1.0, x1)) * w
        d = top + max(0.0, min(1.0, z1)) * w
        if c <= a or d <= b:
            # degenerate rect: draw a thin marker so it stays visible
            c, d = max(c, a + 2), max(d, b + 2)
        dr.rectangle([a, b, c, d], fill=COLORS.get(typ, FALLBACK),
                     outline=(60, 60, 60))
        dr.text((a + 2, b + 2), typ, fill=(30, 30, 30))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def render_grid(plans, path: str, cols: int = 4, cell: int = 220,
                title: str | None = None) -> str:
    """plans: list of room-lists; one mini plan per cell."""
    rows = (len(plans) + cols - 1) // cols
    head = 24 if title else 0
    im = Image.new("RGB", (cols * cell, rows * cell + head), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    if title:
        dr.text((8, 6), title, fill=(20, 20, 20))
    for i, rooms in enumerate(plans):
        cx, cy = (i % cols) * cell, (i // cols) * cell + head
        tile = Image.new("RGB", (cell, cell), (250, 250, 245))
        td = ImageDraw.Draw(tile)
        m, w = 10, cell - 20
        for typ, x0, z0, x1, z1 in rooms:
            a = m + max(0.0, min(1.0, x0)) * w
            b = m + max(0.0, min(1.0, z0)) * w
            c = m + max(0.0, min(1.0, x1)) * w
            d = m + max(0.0, min(1.0, z1)) * w
            if c <= a or d <= b:  # degenerate rect: keep a visible marker
                c, d = max(c, a + 2), max(d, b + 2)
            td.rectangle([a, b, c, d],
                         fill=COLORS.get(typ, FALLBACK), outline=(60, 60, 60))
        im.paste(tile, (cx, cy))
        dr.rectangle([cx, cy, cx + cell - 1, cy + cell - 1],
                     outline=(210, 210, 210))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path
