"""Which interpretable features define each cluster?

For every cluster: z-scored mean of the 43 handcrafted features vs the global
population. Prints the top over/under-expressed features per cluster — the
first step towards "does any latent axis map to a human concept?"

Output: out/axes.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    X = np.load(os.path.join(args.out, "features.npz"))["X"]
    names = json.load(open(os.path.join(args.out, "feature_names.json")))
    emb = np.load(os.path.join(args.out, "embeddings.npz"))
    cid = emb["cid"]
    n_hand = 43
    Xh = X[:, :n_hand]
    mu = Xh.mean(axis=0)
    sd = Xh.std(axis=0) + 1e-9

    report = {}
    for c in sorted(set(cid.tolist())):
        m = cid == c
        if m.sum() < 5:
            continue
        z = (Xh[m].mean(axis=0) - mu) / sd
        order = np.argsort(z)
        under = [(names[i], round(float(z[i]), 2)) for i in order[:5]]
        over = [(names[i], round(float(z[i]), 2)) for i in order[::-1][:5]]
        report[str(c)] = {"size": int(m.sum()), "over": over, "under": under}
        print(f"c{c} (n={int(m.sum())})")
        print(f"  over:  {over}")
        print(f"  under: {under}")

    with open(os.path.join(args.out, "axes.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("[axes] done ->", os.path.join(args.out, "axes.json"))


if __name__ == "__main__":
    main()
