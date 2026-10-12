"""Aerial render of a generated world: hillshade + biome colours + layout."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    args = ap.parse_args()

    t = np.load(os.path.join(args.world_dir, "terrain.npz"))
    H, biome = t["height"].astype(float), t["biome"]
    layout = json.load(open(os.path.join(args.world_dir, "layout.json"),
                            encoding="utf-8"))
    spec = layout["spec"]
    wl = spec["water_level"]

    # hillshade (from a slightly smoothed height to avoid high-freq noise)
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, 1.4)
    gy, gx = np.gradient(Hs)
    shade = np.clip(0.75 + (gx * 0.7 - gy * 0.7) / 6.0, 0.35, 1.25)

    rgb = np.zeros(H.shape + (3,))
    rgb[biome == 0] = (0.25, 0.45, 0.75)    # water
    rgb[biome == 1] = (0.86, 0.80, 0.58)    # beach
    rgb[biome == 2] = (0.52, 0.72, 0.42)    # plain
    rgb[biome == 3] = (0.28, 0.52, 0.30)    # forest
    rgb[biome == 4] = (0.62, 0.60, 0.58)    # rock
    rgb *= shade[..., None]
    # faint contour feel from height
    rgb *= (0.85 + 0.15 * (H / H.max()))[..., None]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(np.clip(rgb, 0, 1).transpose(1, 0, 2), origin="lower")

    for road in layout["roads"]:
        pts = np.asarray(road["points"])   # already world coords
        ax.plot(pts[:, 0], pts[:, 1], color=(0.35, 0.3, 0.25),
                lw=road["width"] * 1.6, alpha=0.85)

    for dist in spec["districts"]:
        c = np.asarray(dist["center"]) * np.array(spec["size"])
        r = dist["radius"] * spec["size"][0]
        ax.add_patch(plt.Circle(c, r, fill=False, ec=(0.9, 0.2, 0.2),
                                lw=1.4, ls="--", alpha=0.8))
        ax.text(c[0], c[1] + r + 6, dist["label"], color=(0.85, 0.1, 0.1),
                fontsize=9, ha="center")

    for s in layout["slots"]:
        ax.plot(s["x"], s["z"], ".", color=(0.1, 0.1, 0.1), ms=2.5, alpha=0.6)
    for lm in layout["landmarks"]:
        ax.plot(lm["x"], lm["z"], "*", color=(1.0, 0.85, 0.1), ms=14,
                mec=(0.2, 0.2, 0.2))
        ax.text(lm["x"] + 4, lm["z"] + 4, lm["type"], fontsize=8,
                color=(0.95, 0.75, 0.0))
    if layout["trees"]:
        tr = np.asarray(layout["trees"])
        ax.plot(tr[:, 0], tr[:, 1], ".", color=(0.05, 0.25, 0.05), ms=1.6,
                alpha=0.5)

    ax.set_title(f"{spec['name']} — aerial (WL={wl})")
    fig.savefig(os.path.join(args.world_dir, "aerial.png"), dpi=110,
                bbox_inches="tight")
    print(f"[aerial] -> {os.path.join(args.world_dir, 'aerial.png')}",
          flush=True)


if __name__ == "__main__":
    main()
