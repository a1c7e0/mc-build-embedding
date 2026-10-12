"""Headless isometric voxel renderer (PIL painter's algorithm + face culling).

No Minecraft install and no GL needed: numpy + PIL only. Each voxel's three
camera-facing faces (+y top, +x right, +z left) are drawn back-to-front in
(x+y+z) order with simple Lambert-style shading, then the image is downscaled
to the requested pixel budget.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Approximate legacy block-id colours (id -> RGB). Anything not listed gets a
# deterministic neutral shade; exact texture fidelity is irrelevant here.
_COLORS: dict[int, tuple[int, int, int]] = {
    1: (125, 125, 125), 2: (124, 183, 86), 3: (134, 96, 67), 4: (128, 128, 128),
    5: (157, 128, 79), 6: (90, 140, 60), 7: (60, 60, 60), 8: (63, 100, 180),
    9: (63, 100, 180), 10: (207, 92, 20), 11: (207, 92, 20), 12: (219, 207, 163),
    13: (136, 126, 120), 14: (143, 140, 125), 15: (136, 130, 127),
    16: (117, 117, 117), 17: (102, 81, 50), 18: (60, 120, 35), 19: (188, 186, 86),
    20: (200, 222, 232), 21: (80, 100, 160), 22: (80, 100, 160),
    23: (110, 110, 110), 24: (216, 203, 155), 25: (120, 80, 60),
    26: (160, 60, 60), 27: (120, 90, 60), 28: (120, 90, 60),
    29: (110, 120, 90), 30: (220, 220, 220), 31: (110, 150, 70),
    32: (140, 120, 70), 33: (120, 100, 70), 34: (120, 100, 70),
    35: (215, 215, 215), 36: (150, 130, 100), 37: (220, 210, 90),
    38: (200, 100, 130), 39: (140, 110, 80), 40: (190, 60, 60),
    41: (250, 220, 80), 42: (220, 220, 220), 43: (140, 140, 140),
    44: (140, 140, 140), 45: (150, 80, 70), 46: (200, 60, 50),
    47: (160, 130, 80), 48: (120, 130, 120), 49: (30, 25, 45),
    50: (240, 220, 130), 51: (230, 120, 30), 52: (70, 70, 90),
    53: (128, 128, 128), 54: (140, 110, 60), 56: (110, 170, 180),
    57: (100, 220, 200), 58: (150, 120, 70), 59: (140, 160, 70),
    60: (110, 80, 60), 61: (110, 110, 110), 62: (140, 110, 80),
    63: (170, 140, 80), 64: (150, 110, 60), 65: (140, 110, 60),
    66: (120, 90, 60), 67: (128, 128, 128), 69: (110, 110, 110),
    70: (128, 128, 128), 71: (200, 200, 200), 72: (150, 120, 70),
    73: (130, 100, 90), 76: (220, 120, 80), 77: (128, 128, 128),
    78: (240, 245, 250), 79: (160, 190, 230), 80: (245, 250, 255),
    81: (70, 130, 60), 82: (160, 170, 180), 83: (110, 170, 80),
    84: (140, 110, 70), 85: (150, 120, 70), 86: (210, 130, 40),
    87: (120, 50, 50), 88: (120, 100, 80), 89: (200, 180, 110),
    90: (90, 40, 120), 91: (215, 140, 40), 92: (220, 180, 150),
    95: (200, 220, 230), 96: (150, 120, 70), 97: (128, 128, 128),
    98: (135, 135, 135), 99: (160, 120, 90), 100: (160, 120, 90),
    101: (180, 185, 190), 102: (200, 222, 232), 103: (120, 160, 60),
    106: (70, 120, 50), 107: (150, 120, 70), 108: (150, 80, 70),
    109: (135, 135, 135), 110: (130, 120, 120), 111: (90, 150, 80),
    112: (90, 45, 45), 113: (90, 45, 45), 114: (90, 45, 45),
    115: (130, 50, 60), 116: (150, 120, 90), 117: (140, 120, 90),
    118: (110, 110, 120), 120: (210, 200, 160), 121: (220, 220, 180),
    122: (40, 30, 50), 123: (180, 130, 80), 124: (220, 170, 90),
    125: (157, 128, 79), 126: (157, 128, 79), 128: (216, 203, 155),
    129: (90, 180, 130), 130: (70, 50, 60), 133: (90, 180, 130),
    134: (102, 81, 50), 135: (190, 170, 110), 136: (140, 110, 70),
    137: (120, 100, 80), 138: (180, 220, 220), 139: (128, 128, 128),
    140: (160, 120, 90), 141: (110, 160, 70), 142: (110, 160, 70),
    143: (150, 120, 70), 145: (90, 90, 95), 146: (140, 110, 60),
    152: (190, 60, 50), 153: (143, 140, 125), 154: (110, 100, 100),
    155: (235, 230, 220), 156: (235, 230, 220), 158: (110, 100, 100),
    159: (160, 100, 75), 160: (200, 222, 232), 161: (60, 120, 35),
    162: (102, 81, 50), 163: (180, 120, 60), 164: (70, 55, 40),
    165: (120, 190, 100), 167: (200, 200, 200), 168: (120, 160, 150),
    169: (190, 215, 210), 170: (210, 190, 90), 171: (200, 200, 200),
    172: (160, 100, 75), 173: (25, 25, 25), 174: (190, 215, 245),
    175: (200, 100, 130), 179: (216, 203, 155), 180: (216, 203, 155),
    182: (216, 203, 155), 183: (150, 120, 70), 184: (190, 170, 110),
    185: (140, 110, 70), 186: (70, 55, 40), 187: (150, 120, 70),
    188: (190, 170, 110), 189: (140, 110, 70), 190: (70, 55, 40),
    191: (150, 120, 70), 192: (180, 120, 60), 193: (150, 120, 70),
    194: (190, 170, 110), 195: (140, 110, 70), 196: (70, 55, 40),
    198: (110, 90, 80), 199: (160, 140, 120), 200: (200, 210, 220),
    201: (160, 130, 110), 202: (190, 180, 175), 203: (135, 135, 135),
    205: (200, 190, 190), 206: (220, 215, 210), 207: (170, 130, 160),
    208: (170, 150, 130), 209: (120, 80, 140), 210: (150, 130, 110),
    211: (110, 100, 100), 212: (100, 120, 170), 213: (160, 100, 60),
    214: (120, 50, 50), 215: (120, 50, 50), 216: (150, 120, 100),
    218: (70, 70, 80), 219: (130, 120, 150), 220: (220, 220, 220),
    221: (120, 110, 120), 222: (120, 110, 120), 223: (150, 120, 70),
    224: (120, 100, 60), 225: (170, 140, 100), 226: (170, 140, 100),
    227: (160, 150, 130), 228: (170, 150, 120), 229: (170, 150, 120),
    230: (150, 120, 70), 231: (170, 140, 100), 232: (170, 140, 100),
    233: (120, 160, 140), 234: (150, 120, 70), 235: (190, 170, 110),
    236: (140, 110, 70), 237: (120, 50, 50), 238: (120, 50, 50),
    239: (110, 160, 70), 240: (110, 160, 70), 241: (200, 200, 200),
    242: (140, 120, 90), 243: (140, 120, 90), 244: (220, 220, 220),
    245: (128, 128, 128), 246: (150, 140, 140), 247: (110, 130, 150),
    248: (140, 140, 150), 249: (140, 140, 150), 250: (200, 220, 200),
    251: (180, 180, 180), 252: (120, 120, 130), 253: (120, 120, 130),
    254: (200, 200, 210), 255: (200, 200, 210),
}

_TOP = np.array([1.00, 1.00, 1.00])
_RIGHT = np.array([0.74, 0.74, 0.74])
_LEFT = np.array([0.55, 0.55, 0.55])


def block_color(bid: int) -> tuple[int, int, int]:
    if bid in _COLORS:
        return _COLORS[bid]
    # deterministic neutral fallback
    v = 90 + (bid * 37) % 120
    return (v, v, v)


def _shade(c: tuple[int, int, int], f: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.clip(ch * k, 0, 255)) for ch, k in zip(c, f))


def render(grid: np.ndarray, max_px: int = 440) -> Image.Image:
    """Render a [x, y, z] uint8 voxel grid (0 = air) to an RGB image."""
    dx, dy, dz = grid.shape
    hw, hh, ch = 8, 4, 8

    def proj(px: float, py: float, pz: float) -> tuple[float, float]:
        return ((px - pz) * hw, (px + pz) * hh - py * ch)

    # image bounds from the 8 corners of the bbox
    xs, ys = [], []
    for x in (0, dx):
        for y in (0, dy):
            for z in (0, dz):
                sx, sy = proj(x, y, z)
                xs.append(sx)
                ys.append(sy)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    W, H = int(maxx - minx) + 8, int(maxy - miny) + 8

    img = Image.new("RGB", (W, H), (252, 252, 254))
    draw = ImageDraw.Draw(img)

    occ = grid != 0
    if not occ.any():
        return img

    # surface blocks only (any empty face neighbour)
    pad = np.zeros((dx + 2, dy + 2, dz + 2), dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = occ
    empty = ~pad
    surface = occ & (
        empty[:-2, 1:-1, 1:-1] | empty[2:, 1:-1, 1:-1] |
        empty[1:-1, :-2, 1:-1] | empty[1:-1, 2:, 1:-1] |
        empty[1:-1, 1:-1, :-2] | empty[1:-1, 1:-1, 2:])
    xs_i, ys_i, zs_i = np.nonzero(surface)
    order = np.argsort(xs_i + ys_i + zs_i, kind="stable")

    ox, oy = -minx + 4, -miny + 4
    for idx in order:
        x, y, z = int(xs_i[idx]), int(ys_i[idx]), int(zs_i[idx])
        bid = int(grid[x, y, z])
        c = block_color(bid)
        x1, y1, z1 = x + 1, y + 1, z + 1
        # left face (+z) — drawn only when empty
        if z + 1 >= dz or not occ[x, y, z + 1]:
            p = [proj(x, y, z1), proj(x1, y, z1), proj(x1, y1, z1), proj(x, y1, z1)]
            draw.polygon([(sx + ox, sy + oy) for sx, sy in p], fill=_shade(c, _LEFT),
                         outline=_shade(c, _LEFT))
        # right face (+x)
        if x + 1 >= dx or not occ[x + 1, y, z]:
            p = [proj(x1, y, z), proj(x1, y, z1), proj(x1, y1, z1), proj(x1, y1, z)]
            draw.polygon([(sx + ox, sy + oy) for sx, sy in p], fill=_shade(c, _RIGHT),
                         outline=_shade(c, _RIGHT))
        # top face (+y)
        if y + 1 >= dy or not occ[x, y + 1, z]:
            p = [proj(x, y1, z), proj(x1, y1, z), proj(x1, y1, z1), proj(x, y1, z1)]
            draw.polygon([(sx + ox, sy + oy) for sx, sy in p], fill=_shade(c, _TOP),
                         outline=_shade(c, _TOP))

    if max(W, H) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    return img


def contact_sheet(tiles: list[Image.Image], cols: int = 4, cell: int = 300,
                  title: str = "") -> Image.Image:
    """Tile numbered renders into one sheet (for batch vision classification)."""
    n = len(tiles)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, tile in enumerate(tiles):
        t = tile.copy()
        t.thumbnail((cell - 28, cell - 28), Image.LANCZOS)
        cx, cy = (i % cols) * cell, (i // cols) * cell
        sheet.paste(t, (cx + (cell - t.width) // 2,
                        cy + 24 + (cell - 28 - t.height) // 2))
        draw.text((cx + 8, cy + 6), f"#{i + 1}", fill=(200, 30, 30))
    if title:
        draw.text((8, rows * cell - 16), title, fill=(40, 40, 40))
    return sheet


def render2(grid, max_px: int = 380, cutaway: bool = False) -> Image.Image:  # noqa: ANN001
    """Two opposite isometric views (0 deg and 180 deg yaw) side by side.

    A single view can hide the far side of a model (the famous 'sheep butt'
    audit failure); two diagonal views make blind classification reliable.
    cutaway=True opens the near side (two nearest wall planes removed for
    rendering only) — a dollhouse section view showing interior layout.
    """
    import numpy as np

    if cutaway:
        g = grid.copy()
        g[g.shape[0] - 1:, :, :] = 0
        g[:, :, g.shape[2] - 1:] = 0
        a = render(g, max_px)
    else:
        a = render(grid, max_px)
    b = render(np.ascontiguousarray(np.rot90(grid, k=2, axes=(0, 2))), max_px)
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width + 8, h), (255, 255, 255))
    out.paste(a, (0, (h - a.height) // 2))
    out.paste(b, (a.width + 8, (h - b.height) // 2))
    return out
