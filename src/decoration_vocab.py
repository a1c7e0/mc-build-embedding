"""Extract decoration vocabulary from MC building corpus.

Classifies blocks as structure (walls/floor/roof) vs decoration
(furniture/trim/detail) based on position in the building.
Outputs: decoration patterns, block distribution by style,
exterior vs interior split.

usage: python src/decoration_vocab.py [lib_dir] [out_json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Legacy block ID categories (structural vs decorative)
STRUCTURAL_BLOCKS = {
    1,    # stone
    2,    # grass
    3,    # dirt
    4,    # cobblestone
    5,    # planks
    17,   # log
    24,   # sandstone
    45,   # bricks
    98,   # stone brick
    112,  # nether brick
    155,  # quartz
}

# Known decorative/non-structural block types
DECORATIVE_HINTS = {
    28, 29,    # piston
    35,        # wool
    42,        # iron block
    41,        # gold block
    46,        # TNT
    47,        # bookshelf
    50,        # torch
    53,        # wood stairs
    54,        # chest
    55,        # redstone wire
    58,        # crafting table
    61, 62,    # furnace
    63,        # sign
    64,        # door
    65,        # ladder
    66,        # rail
    68,        # wall sign
    69,        # lever
    70,        # stone pressure plate
    72,        # wood pressure plate
    75, 76,    # redstone torch
    77,        # stone button
    81,        # cactus
    85,        # fence
    86,        # pumpkin
    89,        # glowstone
    90,        # portal
    91,        # jack o lantern
    92,        # cake
    96,        # trapdoor
    99, 100,   # mushroom block
    101,       # iron bars
    102,       # glass pane
    103,       # melon
    104, 105,  # vine
    106,       # fence gate
    107,       # brick stairs
    108,       # stone brick stairs
    109,       # nether brick stairs
    110,       # sandstone stairs
    111,       # spruce stairs
    112,       # nether brick
    113,       # nether brick fence
    114,       # nether brick stairs
    115,       # nether wart
    116,       # enchanting table
    117,       # brewing stand
    118,       # cauldron
    119,       # end portal
    120,       # end portal frame
    121,       # end stone
    122,       # dragon egg
    123,       # redstone lamp off
    124,       # redstone lamp on
    125,       # double wood slab
    126,       # wood slab
    127,       # cocoa
    128,       # sandstone stairs
    129,       # emerald ore
    130,       # ender chest
    131,       # tripwire hook
    132,       # tripwire
    133,       # emerald block
    134,       # spruce wood stairs
    135,       # birch wood stairs
    136,       # jungle wood stairs
    137,       # command block
    138,       # beacon
    139,       # cobblestone wall
    140,       # flower pot
    141,       # carrots
    142,       # potatoes
    143,       # wood button
    144,       # skull
    145,       # anvil
    146,       # trapped chest
    147,       # light weighted pressure plate
    148,       # heavy weighted pressure plate
    151,       # daylight sensor
    152,       # redstone block
    153,       # nether quartz ore
    154,       # hopper
    155,       # quartz block
    156,       # quartz stairs
    157,       # activator rail
    158,       # dropper
    159,       # stained clay
    160,       # stained glass pane
    161,       # leaves2
    162,       # log2
    163,       # acacia stairs
    164,       # dark oak stairs
    165,       # slime block
    166,       # barrier
    167,       # iron trapdoor
    168,       # prismarine
    169,       # sea lantern
    170,       # hay block
    171,       # carpet
    172,       # hardened clay
    173,       # coal block
    174,       # packed ice
    175,       # double plant
    198,       # end rod
    199,       # chorus plant
    200,       # chorus flower
    201,       # purpur block
    202,       # purpur pillar
    203,       # purpur stairs
    204,       # purpur double slab
    205,       # purpur slab
    206,       # end stone bricks
    207,       # beetroot
    208,       # grass path
    209,       # end gateway
    210,       # repeating command block
    211,       # chain command block
    212,       # frosted ice
    213,       # magma block
    214,       # nether wart block
    215,       # red nether brick
    216,       # bone block
    217,       # structure void
    218,       # observer
    219,       # white shulker box
    220,       # orange shulker box
    221,       # magenta shulker box
    222,       # light blue shulker box
    223,       # yellow shulker box
    224,       # lime shulker box
    225,       # pink shulker box
    226,       # gray shulker box
    227,       # cyan shulker box
    228,       # purple shulker box
    229,       # blue shulker box
    230,       # brown shulker box
    231,       # green shulker box
    232,       # red shulker box
    233,       # black shulker box
    234,       # white glazed terracotta
    235,       # orange glazed terracotta
    236,       # magenta glazed terracotta
    237,       # light blue glazed terracotta
    238,       # yellow glazed terracotta
    239,       # lime glazed terracotta
    240,       # pink glazed terracotta
    241,       # gray glazed terracotta
    242,       # cyan glazed terracotta
    243,       # purple glazed terracotta
    244,       # blue glazed terracotta
    245,       # brown glazed terracotta
    246,       # green glazed terracotta
    247,       # red glazed terracotta
    248,       # black glazed terracotta
    249,       # concrete
    250,       # concrete powder
    251,       # kelp
    252,       # kelp plant
    253,       # dried kelp block
    254,       # turtle egg
    255,       # seagrass
}

# Water and air
AIR = 0
WATER = 9


def classify_block_position(grid, x, y, z):
    """Classify a block's position: 'wall', 'floor', 'roof', 'interior', 'surface'."""
    W, H, D = grid.shape
    # Check if at building boundary (surface)
    at_x_edge = x == 0 or x == W - 1
    at_y_bottom = y == 0
    at_y_top = y == H - 1
    at_z_edge = z == 0 or z == D - 1

    if at_y_bottom:
        return "floor"
    if at_y_top:
        return "roof"
    if at_x_edge or at_z_edge:
        return "surface"

    # Check if part of a wall (has solid neighbors forming a vertical barrier)
    # For interior classification: check if it's a "free-standing" block
    # surrounded by air (furniture) vs part of a wall
    return "interior"


def extract_decoration(grid, style_label="unknown"):
    """Extract decoration blocks from one building grid."""
    W, H, D = grid.shape
    # Find building bounding box (non-air)
    solid = grid > 0
    if not solid.any():
        return None

    # Get occupied bounds
    xs, ys, zs = np.nonzero(solid)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    z0, z1 = zs.min(), zs.max()

    # Classify each solid block
    block_positions = defaultdict(list)  # block_id -> [(pos_class, x, y, z)]
    structural_blocks = []
    decorative_blocks = []

    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                bid = grid[x, y, z]
                if bid == 0 or bid == WATER:
                    continue
                pos = classify_block_position(grid, x, y, z)

                # Structural if it's a common structural block at structural position
                if bid in STRUCTURAL_BLOCKS and pos in ("wall", "floor", "roof", "surface"):
                    structural_blocks.append((bid, pos, x, y, z))
                else:
                    # Decorative: either non-structural block, or structural block
                    # in a decorative position (interior/furniture)
                    if bid in DECORATIVE_HINTS or pos == "interior":
                        decorative_blocks.append((bid, pos, x, y, z))
                        block_positions[bid].append((pos, x, y, z))

    return {
        "style": style_label,
        "building_size": (int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)),
        "n_structural": len(structural_blocks),
        "n_decorative": len(decorative_blocks),
        "decorative_blocks": block_positions,
    }


def main():
    lib_dir = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "out_lib/decoration_vocab.json"

    meta_file = os.path.join(lib_dir, "meta.jsonl")
    vtypes_file = os.path.join(lib_dir, "vision_types.json")

    meta = [json.loads(l) for l in open(meta_file, encoding="utf-8")]
    vtypes = json.load(open(vtypes_file, encoding="utf-8")) if os.path.exists(vtypes_file) else {}

    print(f"[deco] processing {len(meta)} buildings")

    # Aggregate stats
    style_block_counts = defaultdict(Counter)  # style -> block_id -> count
    pos_block_counts = defaultdict(Counter)    # position -> block_id -> count
    total_deco_by_style = Counter()
    total_struct_by_style = Counter()

    for i, r in enumerate(meta):
        if i % 500 == 0:
            print(f"[deco] {i}/{len(meta)}", flush=True)

        # Get style and type from vision_types.json
        idx = str(r["idx"])
        vt = vtypes.get(idx, {})
        style = vt.get("style", "unknown")
        btype = vt.get("type", "other")
        if style not in ("modern", "medieval", "asian", "scifi", "western", "nautical", "other"):
            style = "other"

        # Only process type=building
        if btype != "building":
            continue

        # Load grid
        gpath = os.path.join(lib_dir, r["file"])
        if not os.path.exists(gpath):
            continue
        g = np.load(gpath)["grid"]

        result = extract_decoration(g, style)
        if result is None:
            continue

        total_deco_by_style[style] += result["n_decorative"]
        total_struct_by_style[style] += result["n_structural"]

        # Aggregate block counts by style and position
        for bid, positions in result["decorative_blocks"].items():
            style_block_counts[style][bid] += len(positions)
            for pos, _, _, _ in positions:
                pos_block_counts[pos][bid] += 1

    # Output summary
    print("\n[deco] === Summary ===")
    print(f"  Buildings processed: {len(meta)}")
    print(f"\n  Decoration blocks by style:")
    for style in sorted(total_deco_by_style.keys()):
        total = total_deco_by_style[style] + total_struct_by_style[style]
        pct = total_deco_by_style[style] / max(total, 1) * 100
        print(f"    {style:12s}: {total_deco_by_style[style]:6d} deco / "
              f"{total_struct_by_style[style]:6d} struct ({pct:.1f}% deco)")

    print(f"\n  Top decorative blocks by style:")
    for style in ["modern", "medieval", "asian", "scifi", "western", "nautical", "other"]:
        if style not in style_block_counts:
            continue
        top = style_block_counts[style].most_common(10)
        print(f"    {style}: {top}")

    print(f"\n  Top decorative blocks by position:")
    for pos in ["interior", "surface", "floor", "roof"]:
        if pos not in pos_block_counts:
            continue
        top = pos_block_counts[pos].most_common(10)
        print(f"    {pos}: {top}")

    # Save to JSON (convert numpy types to Python)
    def to_native(d):
        if isinstance(d, dict):
            return {int(k) if isinstance(k, np.integer) else str(k): to_native(v)
                    for k, v in d.items()}
        if isinstance(d, Counter):
            return {int(k) if isinstance(k, np.integer) else str(k): int(v)
                    for k, v in d.items()}
        if isinstance(d, (np.integer,)):
            return int(d)
        return d

    output = {
        "style_block_counts": to_native(dict(style_block_counts)),
        "pos_block_counts": to_native(dict(pos_block_counts)),
        "total_deco_by_style": to_native(dict(total_deco_by_style)),
        "total_struct_by_style": to_native(dict(total_struct_by_style)),
    }
    json.dump(output, open(out_json, "w"), indent=1)
    print(f"\n[deco] saved: {out_json}")


if __name__ == "__main__":
    main()
