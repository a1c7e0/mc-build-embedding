"""Render study sheets for vision-based inspection.

Produces, in out/sheets/:
  cluster_c{N}.png   - exemplars of each cluster (numbered tiles)
  boundary.png       - builds sitting between two clusters
  interp_{A}_{B}.png - centroid interpolation chain (nearest real builds)
  sample_{K}.png     - random stratified samples for vision classification
  manifest.json      - tile -> building idx mapping for every sheet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402


def sheet(tiles: list[Image.Image], captions: list[str], cols: int = 4,
          cell: int = 320) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, tile in enumerate(tiles):
        t = tile.copy()
        t.thumbnail((cell - 20, cell - 44), Image.LANCZOS)
        cx, cy = (i % cols) * cell, (i // cols) * cell
        img.paste(t, (cx + (cell - t.width) // 2,
                      cy + 30 + (cell - 44 - t.height) // 2))
        draw.text((cx + 8, cy + 8), captions[i], fill=(200, 20, 20))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--cell", type=int, default=320)
    args = ap.parse_args()
    sdir = os.path.join(args.out, "sheets")
    os.makedirs(sdir, exist_ok=True)

    meta = [json.loads(l) for l in
            open(os.path.join(args.out, "meta.jsonl"), encoding="utf-8")]
    load = lambda i: np.load(os.path.join(args.out, meta[i]["file"]))["grid"]  # noqa: E731
    manifest: dict[str, list[dict]] = {}

    def put(name: str, idxs: list[int], caps: list[str]) -> None:
        tiles = [render(load(i), max_px=args.cell - 60) for i in idxs]
        sheet(tiles, caps, cell=args.cell).save(os.path.join(sdir, name))
        manifest[name] = [{"tile": j + 1, "idx": i, "desc": meta[i]["desc"][:200]}
                          for j, i in enumerate(idxs)]
        print(f"[sheets] {name} ({len(idxs)} tiles)", flush=True)

    exemplars = json.load(open(os.path.join(args.out, "exemplars.json")))
    for c, idxs in exemplars.items():
        put(f"cluster_c{c}.png", idxs[:12],
            [f"c{c}#{j + 1}" for j in range(min(12, len(idxs)))])

    bidx = json.load(open(os.path.join(args.out, "boundary.json")))
    put("boundary.png", bidx[:16], [f"b{j + 1}" for j in range(16)])

    interp = json.load(open(os.path.join(args.out, "interp_chain.json")))
    a, b = interp["clusters"]
    chain = interp["chain"]
    idxs, caps = [], []
    for row in chain:
        i = row["nearest"][0]
        idxs.append(i)
        caps.append(f"t={row['t']}")
    put(f"interp_{a}_{b}.png", idxs, caps)

    # stratified random sample across clusters for vision labelling
    emb = np.load(os.path.join(args.out, "embeddings.npz"))
    cid = emb["cid"]
    rng = np.random.default_rng(42)
    per = 2
    picked: list[int] = []
    for c in sorted(set(cid.tolist())):
        m = np.nonzero(cid == c)[0]
        take = min(per, len(m))
        picked.extend(int(i) for i in rng.choice(m, take, replace=False))
    rng.shuffle(picked)
    for k in range(0, min(len(picked), 96), 16):
        chunk = picked[k:k + 16]
        put(f"sample_{k // 16}.png", chunk,
            [f"s{j + 1}" for j in range(len(chunk))])

    with open(os.path.join(sdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"[sheets] done -> {sdir}", flush=True)


if __name__ == "__main__":
    main()
