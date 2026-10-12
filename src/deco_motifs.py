"""Extract multi-block decoration patterns (motifs) from MC building corpus.

Instead of single blocks, learn 3D block arrangements that form furniture,
trim, and detail patterns. These motifs capture how MC players actually
build decorations (tables = planks+fence legs, chairs = stairs+signs, etc).

Outputs: out_lib/deco_motifs.json (motif patterns + frequencies + style)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

AIR = 0
WATER = 9

# Motif extraction: slide a 3x3x3 window over all "decoration" positions
# (interior air pockets + surface blocks) and record the block pattern
MOTIF_SIZE = 3


def extract_motifs_from_grid(grid, style, min_fill=3):
    """Extract 3x3x3 motifs from decoration areas of one building.
    min_fill: minimum non-air blocks in a motif to be considered.
    Returns Counter of motif_hash -> count."""
    W, H, D = grid.shape
    if W < MOTIF_SIZE or H < MOTIF_SIZE or D < MOTIF_SIZE:
        return Counter()

    solid = grid > 0
    motifs = Counter()

    for x in range(W - MOTIF_SIZE + 1):
        for y in range(H - MOTIF_SIZE + 1):
            for z in range(D - MOTIF_SIZE + 1):
                patch = grid[x:x+MOTIF_SIZE, y:y+MOTIF_SIZE, z:z+MOTIF_SIZE]

                # Must have some solid blocks but not ALL solid
                n_solid = int((patch > 0).sum())
                if n_solid < min_fill or n_solid == MOTIF_SIZE ** 3:
                    continue

                # Must contain at least one "decoration" block type
                # (not just structural stone/dirt)
                unique_blocks = set(patch.flatten().tolist()) - {AIR}
                if not unique_blocks:
                    continue

                # Normalize: represent as tuple of (block_id) in row-major order
                # We keep the actual block IDs to capture the arrangement
                motif_key = tuple(patch.flatten().tolist())
                motifs[motif_key] += 1

    return motifs


def hash_motif(motif_key):
    """Create a compact string representation of a motif."""
    return ",".join(str(b) for b in motif_key)


def motif_to_grid(motif_key, size=MOTIF_SIZE):
    """Convert flat motif tuple back to 3D grid."""
    arr = np.array(motif_key, dtype=np.uint16)
    return arr.reshape(size, size, size)


def main():
    lib_dir = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "out_lib/deco_motifs.json"

    meta_file = os.path.join(lib_dir, "meta.jsonl")
    vtypes_file = os.path.join(lib_dir, "vision_types.json")

    meta = [json.loads(l) for l in open(meta_file, encoding="utf-8")]
    vtypes = json.load(open(vtypes_file, encoding="utf-8")) if os.path.exists(vtypes_file) else {}

    print(f"[motif] processing {len(meta)} buildings")

    # Global motif frequency
    global_motifs = Counter()
    style_motifs = defaultdict(Counter)  # style -> motif -> count
    motif_examples = {}  # motif_hash -> first building idx (for later retrieval)

    processed = 0
    for i, r in enumerate(meta):
        if i % 500 == 0:
            print(f"[motif] {i}/{len(meta)}", flush=True)

        idx = str(r["idx"])
        vt = vtypes.get(idx, {})
        btype = vt.get("type", "other")
        style = vt.get("style", "other")
        if style not in ("modern", "medieval", "asian", "scifi", "western", "nautical", "other"):
            style = "other"

        if btype != "building":
            continue

        gpath = os.path.join(lib_dir, r["file"])
        if not os.path.exists(gpath):
            continue
        g = np.load(gpath)["grid"]

        motifs = extract_motifs_from_grid(g, style)
        processed += 1

        for motif_key, count in motifs.items():
            h = hash_motif(motif_key)
            global_motifs[motif_key] += count
            style_motifs[style][motif_key] += count
            if h not in motif_examples:
                motif_examples[h] = r["idx"]

    print(f"[motif] processed {processed} buildings")
    print(f"[motif] unique motifs: {len(global_motifs)}")

    # Top motifs overall
    print("\n[motif] Top 20 most common 3x3x3 motifs:")
    for motif_key, count in global_motifs.most_common(20):
        # Summarize: what blocks are in this motif?
        blocks = Counter(motif_key)
        block_str = ", ".join(f"{b}x{c}" for b, c in blocks.most_common(5))
        print(f"  count={count:4d}  blocks: {block_str}")

    # Top motifs per style
    print("\n[motif] Top 5 motifs per style:")
    for style in ["modern", "medieval", "asian", "scifi", "western", "nautical"]:
        if style not in style_motifs:
            continue
        top = style_motifs[style].most_common(5)
        print(f"  {style}:")
        for motif_key, count in top:
            blocks = Counter(motif_key)
            block_str = ", ".join(f"{b}x{c}" for b, c in blocks.most_common(4))
            print(f"    {count:4d}x  {block_str}")

    # Save top motifs (for reuse as decoration templates)
    top_motifs = []
    for motif_key, count in global_motifs.most_common(200):
        h = hash_motif(motif_key)
        # Find which style uses this most
        best_style = "other"
        best_count = 0
        for style, counter in style_motifs.items():
            if motif_key in counter and counter[motif_key] > best_count:
                best_count = counter[motif_key]
                best_style = style
        top_motifs.append({
            "hash": h,
            "pattern": list(motif_key),
            "count": count,
            "style": best_style,
            "example_idx": motif_examples.get(h),
            "blocks": dict(Counter(motif_key)),
        })

    output = {
        "motif_size": MOTIF_SIZE,
        "total_unique": len(global_motifs),
        "processed_buildings": processed,
        "top_motifs": top_motifs,
    }
    json.dump(output, open(out_json, "w"))
    print(f"\n[motif] saved: {out_json} ({len(top_motifs)} motifs)")

    # Also save a "furniture-like" subset: motifs with mixed block types
    # (likely furniture) vs uniform blocks (likely structural)
    furniture_motifs = [m for m in top_motifs
                       if len(m["blocks"]) >= 2 and m["count"] >= 50]
    print(f"[motif] furniture-like motifs (mixed blocks, count>=50): {len(furniture_motifs)}")
    for m in furniture_motifs[:10]:
        print(f"  {m['count']:4d}x  {m['blocks']}  style={m['style']}")


if __name__ == "__main__":
    main()
