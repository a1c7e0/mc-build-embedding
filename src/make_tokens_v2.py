"""Assemble vq_tokens.npz (tokens + idx + styles) from atlas2 extraction.

Used when tokens were produced by atlas2.py (which saves tokens only).
usage: python src/make_tokens_v2.py
"""
from __future__ import annotations

import json

import numpy as np

STYLES = ["modern", "medieval", "asian", "scifi", "western", "nautical",
          "other"]


def main() -> None:
    d = np.load("out_lib/atlas2/tokens.npz")
    tokens = d["tokens"]
    idx = np.load("out_lib/vq_data2.npz")["idx"]
    vt = json.load(open("out_lib/vision_types.json", encoding="utf-8"))
    styles = np.array([
        STYLES.index(vt.get(str(i), {}).get("style", "zz"))
        if vt.get(str(i), {}).get("style") in STYLES else 7
        for i in idx], dtype=np.int32)
    np.savez_compressed("out_lib/vq_tokens.npz", tokens=tokens, idx=idx,
                        styles=styles)
    print(f"[make-tokens] {tokens.shape} styles "
          f"{np.bincount(styles).tolist()}", flush=True)


if __name__ == "__main__":
    main()
