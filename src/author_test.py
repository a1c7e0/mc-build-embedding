"""Test the style-family hypothesis: do same-author builds cluster together?

For authors with >=5 builds in the library: kNN(10) same-author rate in the
PCA embedding vs the random-pairing baseline. High lift = author/style is a
real factor in the latent space (props share style with their project-mates).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

out = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
meta = [json.loads(l) for l in open(f"{out}/meta.jsonl", encoding="utf-8")]
emb = np.load(f"{out}/embeddings.npz")
Z = emb["Z32"]

m = json.load(open("data/schematicsWithFinalUrl.json", encoding="utf-8"))
user_by_url = {e["url"]: e.get("user", "") for e in m if isinstance(e, dict)}
users = np.array([user_by_url.get(r.get("url", ""), "") for r in meta])

vals, counts = np.unique(users, return_counts=True)
big = set(vals[counts >= 5].tolist())
mask = np.isin(users, list(big))
print(f"buildings={len(meta)} known_author={int((users != '').sum())} "
      f"authors_total={len(vals)} authors>=5={len(big)} "
      f"buildings_in_them={int(mask.sum())}", flush=True)

Zm, um = Z[mask], users[mask]
nn = NearestNeighbors(n_neighbors=11).fit(Zm)
_, idx = nn.kneighbors(Zm)
hit = float(np.mean([np.mean(um[row[1:]] == um[i])
                     for i, row in enumerate(idx)]))
rng = np.random.default_rng(0)
perm = rng.permutation(len(um))
base = float(np.mean(um == um[perm]))
print(f"kNN10 same-author rate: {hit:.3f}  baseline: {base:.3f}  "
      f"lift: {hit / max(base, 1e-9):.1f}x", flush=True)

# top authors by count, for inspection
top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:10]
print("top authors:", [(c, u[:18]) for c, u in top], flush=True)
