"""Analyse the learned AE latent space (out/ae_latent.npz) and compare it
against the handcrafted+PCA space.

- UMAP(2D) + KMeans(k=8) -> out/ae/embeddings.npz, clusters.json, exemplars
- axis probing: Pearson corr of each latent dim vs the 43 interpretable
  features -> out/ae/axis_corr.json  ("does z_i mean anything?")
- renders exemplar contact sheets -> out/ae/sheets/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from umap import UMAP  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from embed import mine_labels  # noqa: E402
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    adir = os.path.join(args.out, "ae")
    os.makedirs(os.path.join(adir, "sheets"), exist_ok=True)

    Z = np.load(os.path.join(args.out, "ae_latent.npz"))["Z256"]
    Xh = np.load(os.path.join(args.out, "features.npz"))["X"][:, :43]
    names = json.load(open(os.path.join(args.out, "feature_names.json")))[:43]
    meta = [json.loads(l) for l in
            open(os.path.join(args.out, "meta.jsonl"), encoding="utf-8")]

    emb2 = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Z)
    cid = KMeans(n_clusters=args.k, n_init=10, random_state=42).fit_predict(Z)
    fn = np.array([mine_labels(r["desc"])[0] for r in meta])
    st = np.array([mine_labels(r["desc"])[1] for r in meta])
    np.savez_compressed(os.path.join(adir, "embeddings.npz"),
                        Z=Z, emb2=emb2, cid=cid, fn=fn, st=st)

    # axis probing: latent dim -> interpretable feature correlations
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    Xs = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-9)
    C = (Zs.T @ Xs) / Z.shape[0]
    axis_corr = {}
    for j in range(Z.shape[1]):
        order = np.argsort(np.abs(C[j]))[::-1][:3]
        axis_corr[str(j)] = [(names[k], round(float(C[j, k]), 3))
                             for k in order]
    strong = sorted(axis_corr.items(),
                    key=lambda kv: -abs(kv[1][0][1]))[:12]
    print("[ae] strongest latent<->feature correlations:", flush=True)
    for dim, corr in strong:
        print(f"  z{dim}: {corr}", flush=True)
    json.dump(axis_corr, open(os.path.join(adir, "axis_corr.json"), "w"),
              indent=2)

    # cluster profiles + exemplars + sheets
    clusters, exemplars = {}, {}
    for c in sorted(set(cid.tolist())):
        m = cid == c
        fn_top = sorted({l: int((fn[m] == l).sum()) for l in
                         set(fn[m].tolist())}.items(), key=lambda kv: -kv[1])[:4]
        clusters[int(c)] = {"size": int(m.sum()),
                            "function_top": fn_top}
        idxs = np.nonzero(m)[0]
        d = np.linalg.norm(Z[idxs] - Z[idxs].mean(axis=0), axis=1)
        picks = [int(i) for i in idxs[np.argsort(d)][:12]]
        exemplars[int(c)] = picks
        tiles = [render(np.load(os.path.join(args.out, meta[i]["file"]))["grid"],
                        max_px=260) for i in picks]
        sheet(tiles, [f"ae-c{c}#{j + 1}" for j in range(len(picks))]).save(
            os.path.join(adir, "sheets", f"ae_cluster_c{c}.png"))
        print(f"[ae] c{c} n={int(m.sum())} fn={fn_top}", flush=True)
    json.dump(clusters, open(os.path.join(adir, "clusters.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(exemplars, open(os.path.join(adir, "exemplars.json"), "w"))

    fig, ax = plt.subplots(figsize=(9, 8))
    for c in sorted(set(cid.tolist())):
        m = cid == c
        ax.scatter(emb2[m, 0], emb2[m, 1], s=8, label=f"c{c} ({int(m.sum())})",
                   alpha=0.75, linewidths=0)
    ax.legend(fontsize=8)
    ax.set_title("AE latent (256D) - UMAP + KMeans")
    fig.savefig(os.path.join(adir, "umap_ae.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[ae] done -> {adir}", flush=True)


if __name__ == "__main__":
    main()
