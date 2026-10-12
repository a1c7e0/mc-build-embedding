"""Geodesic chains: walk the kNN graph between clusters through REAL builds.

Straight centroid interpolation crosses empty space (we saw step-function
retrieval). Instead: build a kNN graph over the PCA embedding, run Dijkstra,
and walk the shortest path of real builds from cluster A to cluster B. If the
latent space is a smooth manifold, consecutive builds should change gradually.

Outputs: out/geodesic_{A}_{B}.json and out/sheets/geodesic_{A}_{B}.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pairs", default="0-5,2-3,0-3")
    args = ap.parse_args()

    emb = np.load(os.path.join(args.out, "embeddings.npz"))
    Z32, cid = emb["Z32"], emb["cid"]
    meta = [json.loads(l) for l in
            open(os.path.join(args.out, "meta.jsonl"), encoding="utf-8")]
    exemplars = json.load(open(os.path.join(args.out, "exemplars.json")))
    n = Z32.shape[0]

    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(Z32)
    dist, idx = nn.kneighbors(Z32)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            rows.append(i)
            cols.append(int(j))
            vals.append(float(d))
    G = coo_matrix((vals, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)  # symmetrise
    print(f"[geo] knn graph k={args.k} edges={G.nnz}", flush=True)

    for pair in args.pairs.split(","):
        a, b = (int(v) for v in pair.split("-"))
        src = exemplars[str(a)][0]
        dst = exemplars[str(b)][0]
        d, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)
        if not np.isfinite(d[dst]):
            print(f"[geo] c{a}->c{b}: no path", flush=True)
            continue
        path = []
        node = dst
        while node != src:
            path.append(int(node))
            node = pred[node]
        path.append(int(src))
        path.reverse()
        # drop same-cluster lead-in/out, keep the interesting middle
        picks = [path[i] for i in
                 np.unique(np.linspace(0, len(path) - 1, 10).astype(int))]
        caps = []
        for p in picks:
            caps.append(f"c{int(cid[p])}")
        name = f"geodesic_{a}_{b}"
        tiles = [render(np.load(os.path.join(args.out, meta[p]["file"]))["grid"],
                        max_px=260) for p in picks]
        sheet(tiles, caps).save(os.path.join(args.out, "sheets", f"{name}.png"))
        with open(os.path.join(args.out, f"{name}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "clusters": [a, b],
                "path_len": len(path),
                "total_dist": float(d[dst]),
                "picks": [{"idx": int(p), "cid": int(cid[p]),
                           "desc": meta[p]["desc"][:150]} for p in picks],
            }, fh, ensure_ascii=False, indent=2)
        print(f"[geo] c{a}->c{b}: path_len={len(path)} "
              f"dist={float(d[dst]):.2f} picks={picks}", flush=True)


if __name__ == "__main__":
    main()
