"""Quick motif extraction on first 500 buildings."""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, "src")
from deco_motifs import extract_motifs_from_grid, hash_motif

lib_dir = "out_lib"
meta = [json.loads(l) for l in open(f"{lib_dir}/meta.jsonl", encoding="utf-8")]
vtypes = json.load(open(f"{lib_dir}/vision_types.json", encoding="utf-8"))

global_motifs = Counter()
style_motifs = defaultdict(Counter)
processed = 0

for i, r in enumerate(meta[:500]):
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
    motifs = extract_motifs_from_grid(g, style)
    processed += 1
    for k, c in motifs.items():
        global_motifs[k] += c
        style_motifs[style][k] += c

print(f"processed {processed} buildings, {len(global_motifs)} unique motifs")
print(f"\nTop 15 motifs:")
for mk, c in global_motifs.most_common(15):
    blocks = Counter(mk)
    bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
    print(f"  {c:4d}x  {bs}")

print(f"\nFurniture-like (mixed >=2 block types, count>=20):")
furn = [(mk, c) for mk, c in global_motifs.most_common(100)
        if len(set(mk) - {0}) >= 2 and c >= 20]
for mk, c in furn[:10]:
    blocks = Counter(mk)
    bs = ", ".join(f"{b}x{n}" for b, n in blocks.most_common(5))
    print(f"  {c:4d}x  {bs}")
