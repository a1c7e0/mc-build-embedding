"""Embedding + clustering: does a meaningful latent space exist?

Pipeline: standardise -> PCA(32) -> UMAP(2D) for visualisation.
Clustering: HDBSCAN (fallback KMeans). Validation against weak text labels
mined from each build's description (function words + style words).
Also emits cluster exemplars, boundary builds, and a centroid-interpolation
chain for the "what happens between clusters?" experiment.

Outputs: out/embeddings.npz, out/labels.json, out/clusters.json,
         out/exemplars.json, out/boundary.json, out/interp_chain.json,
         out/umap_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from umap import UMAP  # noqa: E402

FUNCTION_WORDS = {
    "house": ["house", "home", "cottage", "villa", "mansion", "cabin", "hut"],
    "castle": ["castle", "fortress", "fort", "keep", "citadel", "palace"],
    "tower": ["tower", "spire", "lighthouse", "minaret"],
    "temple": ["temple", "church", "cathedral", "shrine", "mosque", "chapel",
               "monastery"],
    "bridge": ["bridge", "aqueduct", "viaduct"],
    "farm": ["farm", "barn", "mill", "windmill", "field", "stable", "silo"],
    "settlement": ["village", "town", "city", "settlement", "district"],
    "ship": ["ship", "boat", "airship", "pirate"],
    "statue": ["statue", "sculpture", "pixel art", "pixelart", "figure"],
    "landscape": ["mountain", "terrain", "landscape", "cave", "waterfall",
                  "forest", "island"],
    "interior": ["interior", "room", "dungeon", "library", "tavern", "inn"],
    "machine": ["redstone", "machine", "contraption", "farm machine", "robot"],
}
STYLE_WORDS = {
    "modern": ["modern", "contemporary", "minimalist", "glass tower",
               "downtown"],
    "medieval": ["medieval", "middle ages", "rustic", "old"],
    "fantasy": ["fantasy", "magic", "elven", "dwarven", "fairy"],
    "scifi": ["sci-fi", "scifi", "futuristic", "space", "cyber", "steampunk"],
    "asian": ["japanese", "chinese", "asian", "pagoda", "zen", "korean"],
    "western": ["western", "colonial", "victorian", "gothic", "tudor",
                "baroque"],
}


def mine_labels(desc: str) -> tuple[str, str]:
    d = desc.lower()
    fn, st = "other", "other"
    for label, words in FUNCTION_WORDS.items():
        if any(re.search(r"\b" + re.escape(w), d) for w in words):
            fn = label
            break
    for label, words in STYLE_WORDS.items():
        if any(re.search(r"\b" + re.escape(w), d) for w in words):
            st = label
            break
    return fn, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    meta = [json.loads(l) for l in
            open(os.path.join(args.out, "meta.jsonl"), encoding="utf-8")]
    X = np.load(os.path.join(args.out, "features.npz"))["X"]
    n = len(meta)
    print(f"[embed] buildings={n} features={X.shape[1]}", flush=True)

    fn_labels, st_labels = [], []
    for rec in meta:
        fn, st = mine_labels(rec["desc"])
        fn_labels.append(fn)
        st_labels.append(st)

    Xs = StandardScaler().fit_transform(X)
    pca_dim = min(32, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42).fit(Xs)
    Z32 = pca.transform(Xs)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"[embed] PCA{pca_dim} explained var={var:.3f}", flush=True)

    emb2 = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Z32)

    try:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=max(15, n // 60))
        cid = cl.fit_predict(Z32)
        n_cl = len(set(cid.tolist()) - {-1})
        noise = int((cid == -1).sum())
        print(f"[embed] HDBSCAN clusters={n_cl} noise={noise}", flush=True)
        if n_cl < 3 or noise > n * 0.6:
            raise RuntimeError(f"degenerate clustering (clusters={n_cl}, "
                               f"noise={noise})")
    except Exception as e:  # pragma: no cover
        print(f"[embed] HDBSCAN unusable ({e}); fallback KMeans k={args.k}",
              flush=True)
        cid = KMeans(n_clusters=args.k, n_init=10,
                     random_state=42).fit_predict(Z32)

    # silhouette against weak labels (only labelled builds, PCA space)
    fn_arr = np.array(fn_labels)
    st_arr = np.array(st_labels)
    scores = {}
    for name, arr in [("function", fn_arr), ("style", st_arr)]:
        m = arr != "other"
        if m.sum() > 20 and len(set(arr[m])) > 1:
            scores[name] = round(float(silhouette_score(Z32[m], arr[m],
                                                        metric="cosine")), 3)
    print(f"[embed] label silhouette (cosine, PCA space): {scores}", flush=True)

    # cluster profiles: top words + dominant labels per cluster
    clusters = {}
    for c in sorted(set(cid.tolist())):
        m = cid == c
        fn_top = sorted({l: int((fn_arr[m] == l).sum())
                         for l in set(fn_arr[m].tolist())}.items(),
                        key=lambda kv: -kv[1])[:4]
        st_top = sorted({l: int((st_arr[m] == l).sum())
                         for l in set(st_arr[m].tolist())}.items(),
                        key=lambda kv: -kv[1])[:4]
        clusters[int(c)] = {
            "size": int(m.sum()),
            "function_top": fn_top,
            "style_top": st_top,
            "mean_dims": [round(float(v), 1) for v in
                          np.mean([[d for d in rec["dims"]] for rec, mm in
                                   zip(meta, m.tolist()) if mm], axis=0)],
        }
    print(f"[embed] cluster profiles:", flush=True)
    for c, prof in clusters.items():
        print(f"  c{c}: n={prof['size']} fn={prof['function_top']} "
              f"st={prof['style_top']}", flush=True)

    # exemplars: closest to centroid in PCA space
    exemplars = {}
    for c in sorted(set(cid.tolist())):
        if c == -1:
            continue
        m = np.nonzero(cid == c)[0]
        ctr = Z32[m].mean(axis=0)
        d = np.linalg.norm(Z32[m] - ctr, axis=1)
        exemplars[int(c)] = [int(i) for i in m[np.argsort(d)][:12]]

    # boundary builds: close to two centroids at once
    real = [c for c in sorted(set(cid.tolist())) if c != -1]
    ctrs = np.vstack([Z32[cid == c].mean(axis=0) for c in real])
    D = np.linalg.norm(Z32[:, None, :] - ctrs[None, :, :], axis=2)
    D.sort(axis=1)
    ratio = D[:, 0] / np.maximum(D[:, 1], 1e-9)
    bidx = [int(i) for i in np.argsort(ratio)[:60]]

    # interpolation between the two most distant cluster centroids
    ia, ib = 0, int(np.argmax(np.linalg.norm(ctrs - ctrs[0], axis=1)))
    chain = []
    for t in np.linspace(0, 1, 9):
        z = (1 - t) * ctrs[ia] + t * ctrs[ib]
        d = np.linalg.norm(Z32 - z, axis=1)
        order = np.argsort(d)
        picks = [int(i) for i in order if cid[i] != -1][:5]
        chain.append({"t": round(float(t), 3), "nearest": picks,
                      "fn": [fn_labels[i] for i in picks],
                      "st": [st_labels[i] for i in picks]})

    np.savez_compressed(
        os.path.join(args.out, "embeddings.npz"),
        Z32=Z32, emb2=emb2, cid=cid,
        fn=fn_arr, st=st_arr)
    for fname, obj in [
        ("labels.json", {"function": fn_labels, "style": st_labels}),
        ("clusters.json", clusters),
        ("exemplars.json", exemplars),
        ("boundary.json", bidx),
        ("interp_chain.json", {"clusters": [real[ia], real[ib]],
                               "chain": chain}),
    ]:
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)

    # plots
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in sorted(set(cid.tolist())):
        m = cid == c
        ax.scatter(emb2[m, 0], emb2[m, 1], s=8,
                   label=("noise" if c == -1 else f"c{c} ({int(m.sum())})"),
                   alpha=0.75, linewidths=0)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Minecraft builds - UMAP(PCA32)  k-profile HDBSCAN/KMeans")
    fig.savefig(os.path.join(args.out, "umap_clusters.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    for field, arr in [("function", fn_arr), ("style", st_arr)]:
        fig, ax = plt.subplots(figsize=(9, 8))
        for lab in sorted(set(arr.tolist())):
            m = arr == lab
            ax.scatter(emb2[m, 0], emb2[m, 1], s=8, label=f"{lab} ({int(m.sum())})",
                       alpha=0.75, linewidths=0)
        ax.legend(fontsize=8, loc="best")
        ax.set_title(f"UMAP coloured by mined {field} label")
        fig.savefig(os.path.join(args.out, f"umap_{field}.png"), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"[embed] done elapsed={time.time() - t0:.0f}s "
          f"boundary={len(bidx)} interp_clusters={[real[ia], real[ib]]}",
          flush=True)


if __name__ == "__main__":
    main()
