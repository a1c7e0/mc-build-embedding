"""Style-based decoration vocabulary: maps styles to block distributions
per position (exterior/interior/floor/roof), learned from MC corpus.

Loads decoration_vocab.json and produces a normalized style→block
probability table for each decoration position.

Outputs: out_lib/deco_style_table.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Block ID → readable name (legacy IDs we care about)
BLOCK_NAMES = {
    1: "stone", 2: "grass", 3: "dirt", 4: "cobblestone", 5: "planks",
    12: "sand", 13: "gravel", 17: "log", 18: "leaves", 20: "glass",
    24: "sandstone", 35: "wool", 41: "gold", 42: "iron", 43: "double_stone_slab",
    45: "bricks", 47: "bookshelf", 50: "torch", 53: "wood_stairs",
    55: "redstone_wire", 85: "fence", 86: "pumpkin", 89: "glowstone",
    98: "stone_brick", 101: "iron_bars", 102: "glass_pane",
    155: "quartz", 159: "stained_clay", 171: "carpet", 173: "coal_block",
}

# Classify each block as decoration_type
DECO_CATEGORY = {
    35: "lighting_soft", 89: "lighting_bright", 50: "lighting_torch",
    75: "lighting_torch", 86: "lighting_lantern", 41: "accent_metal",
    42: "accent_metal", 101: "accent_metal", 102: "accent_glass",
    20: "accent_glass", 85: "trim_wood", 53: "trim_wood", 47: "furniture_wood",
    171: "textile", 35: "textile", 159: "textile", 173: "accent_dark",
    55: "redstone_tech", 13: "natural_ground", 12: "natural_ground",
    18: "natural_plant", 251: "natural_plant",
}


def load_vocab(path="out_lib/decoration_vocab.json"):
    return json.load(open(path, encoding="utf-8"))


def build_style_table(vocab):
    """Build style→block→probability table per position."""
    style_blocks = vocab.get("style_block_counts", {})
    pos_blocks = vocab.get("pos_block_counts", {})

    # Normalize: for each style, compute block distribution
    style_table = {}
    for style, counts in style_blocks.items():
        total = sum(counts.values()) or 1
        # Normalize to probabilities
        probs = {int(k): v / total for k, v in counts.items()}
        # Sort by probability
        sorted_probs = dict(sorted(probs.items(), key=lambda x: -x[1]))
        style_table[style] = sorted_probs

    # Position-specific: surface (exterior), interior, floor, roof
    pos_table = {}
    for pos, counts in pos_blocks.items():
        total = sum(counts.values()) or 1
        probs = {int(k): v / total for k, v in counts.items()}
        pos_table[pos] = dict(sorted(probs.items(), key=lambda x: -x[1]))

    return style_table, pos_table


def get_deco_blocks_for_style(style, pos, style_table, pos_table, top_n=8):
    """Get top decoration blocks for a given style and position.
    Combines style preferences with position-specific blocks."""
    style_probs = style_table.get(style, style_table.get("other", {}))
    pos_probs = pos_table.get(pos, {})

    # Combine: weighted average of style distribution and position distribution
    all_blocks = set(style_probs.keys()) | set(pos_probs.keys())
    combined = {}
    for b in all_blocks:
        sp = style_probs.get(b, 0)
        pp = pos_probs.get(b, 0)
        # Style weight 0.4, position weight 0.6
        combined[b] = 0.4 * sp + 0.6 * pp

    # Sort and return top N
    sorted_blocks = sorted(combined.items(), key=lambda x: -x[1])[:top_n]
    return [(int(b), round(p, 4)) for b, p in sorted_blocks]


def main():
    vocab = load_vocab()
    style_table, pos_table = build_style_table(vocab)

    print("[deco] Style decoration profiles:")
    for style in ["modern", "medieval", "asian", "scifi", "western", "nautical", "other"]:
        if style not in style_table:
            continue
        top = list(style_table[style].items())[:8]
        blocks = [(BLOCK_NAMES.get(int(b), f"#{b}"), round(p, 3))
                  for b, p in top]
        print(f"  {style:10s}: {blocks}")

    print("\n[deco] Position decoration profiles:")
    for pos in ["surface", "interior", "floor", "roof"]:
        if pos not in pos_table:
            continue
        top = list(pos_table[pos].items())[:8]
        blocks = [(BLOCK_NAMES.get(int(b), f"#{b}"), round(p, 3))
                  for b, p in top]
        print(f"  {pos:10s}: {blocks}")

    # Show combined style+position selections
    print("\n[deco] Combined style+position selections (top 5 each):")
    for style in ["modern", "medieval", "asian"]:
        for pos in ["surface", "interior", "roof"]:
            blocks = get_deco_blocks_for_style(style, pos, style_table, pos_table, top_n=5)
            names = [(BLOCK_NAMES.get(b, f"#{b}"), p) for b, p in blocks]
            print(f"  {style}/{pos}: {names}")

    # Save
    output = {
        "style_table": {s: {str(k): v for k, v in t.items()}
                        for s, t in style_table.items()},
        "pos_table": {p: {str(k): v for k, v in t.items()}
                      for p, t in pos_table.items()},
        "block_names": {str(k): v for k, v in BLOCK_NAMES.items()},
    }
    out_path = "out_lib/deco_style_table.json"
    json.dump(output, open(out_path, "w"), indent=1)
    print(f"\n[deco] saved: {out_path}")


if __name__ == "__main__":
    main()
