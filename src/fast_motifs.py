"""Fast motif extraction: sample 3x3x3 windows with stride, first 200 buildings."""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

lib_dir = "out_lib"
meta = [json.loads(l) for l in open(f"{lib_dir}/meta.jsonl", encoding="utf-8")]
vtypes = json.load(open(f"{lib_dir}/vision_types.json", encoding="utf-8"))

MOTIF = 3
STRIDE = 2  # skip every other position

global_motifs = Counter()
style_motifs = defaultdict(Counter)
processed = 0

for i, r in enumerate(meta[:200]):
    idx = str(r["idx"])
    vt = vtypes.get(idx, {})
    if vt.get("type") != "building":
        continue
    style = vt.get("style", "other")
    if style not in ("modern", "medieval", "asian", "scifi", "western", "nautical"):
        style = "other"
    gpath = os.path.join(lib_dir, r["file"])
    if not os.path.exists(gpath):
        continue
    g = np.load(gpath)["grid"]
    W, H, D = g.shape
    if W < MOTIF or H < MOTIF or D < MOTIF:
        continue
    processed += 1

    for x in range(0, W - MOTIF + 1, STRIDE):
        for y in range(0, H - MOTIF + 1, STRIDE):
            for z in range(0, D - MOTIF + 1, STRIDE):
                patch = g[x:x+MOTIF, y:y+MOTIF, z:z+MOTIF]
                n_solid = int((patch > 0).sum())
                if n_solid < 3 or n_solid == MOTIF ** 3:
                    continue
                # Must have at least 2 distinct non-air blocks (furniture-like)
                unique = set(patch.flatten().tolist()) - {0}
                if len(unique) < 1:
                    continue
                mk = tuple(patch.flatten().tolist())
                global_motifs[mk] += 1
                style_motifs[style][mk] += 1

print(f"processed {processed} buildings, {len(global_motifs)} unique motifs")

print(f"\nTop 20 motifs (by frequency):")
for mk, c in global_motifs.most_common(20):
    blocks = Counter(mk)
    bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
    n_types = len(set(mk) - {0})
    print(f"  {c:4d}x [{n_types} types]  {bs}")

print(f"\nMixed-block motifs (furniture-like, >=2 non-air types, count>=10):")
furn = [(mk, c) for mk, c in global_motifs.most_common(200)
        if len(set(mk) - {0}) >= 2 and c >= 10]
print(f"  found {len(furn)}")
for mk, c in furn[:15]:
    blocks = Counter(mk)
    bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
    print(f"  {c:4d}x  {bs}")

# Per style top motifs
print(f"\nPer-style top 3 motifs:")
for style in ["modern", "medieval", "asian", "scifi", "western", "nautical"]:
    if style not in style_motifs:
        continue
    top = style_motifs[style].most_common(3)
    for mk, c in top:
        blocks = Counter(mk)
        bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(4))
        print(f"  {style:10s} {c:4d}x  {bs}")
