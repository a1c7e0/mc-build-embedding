"""Contact-sheet composer for stylegen renders (PIL only)."""
from __future__ import annotations

from PIL import Image, ImageDraw


def make_sheet(items, out_path: str, title: str, cols: int = 5,
               cell=(700, 400)) -> str:
    """items = [(label, png_path), ...]; one sheet image, labels per tile."""
    rows = max(1, (len(items) + cols - 1) // cols)
    head = 34
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1] + head),
                      (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(20, 20, 20))
    for i, (label, png) in enumerate(items):
        tile = Image.open(png).convert("RGB")
        resample = getattr(Image, "Resampling", Image).LANCZOS
        tile.thumbnail((cell[0] - 16, cell[1] - 44), resample)
        cx = (i % cols) * cell[0]
        cy = (i // cols) * cell[1] + head
        sheet.paste(tile, (cx + (cell[0] - tile.width) // 2,
                           cy + 30 + (cell[1] - 44 - tile.height) // 2))
        draw.text((cx + 10, cy + 8), label, fill=(170, 20, 20))
    sheet.save(out_path)
    return out_path
