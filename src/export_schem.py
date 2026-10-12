"""Export the assembled world as a Sponge Schematic v2 (.schem, gzip NBT).

The file loads with WorldEdit/FAWE/Litematica in Minecraft Java.

Layout per Sponge Schematic v2:
  Version(2) DataVersion Width Height Length Offset[3] PaletteMax
  Palette{name: idx} BlockData{varint indices, x + z*W + y*W*L}
"""
from __future__ import annotations

import argparse
import gzip
import os
import struct

import numpy as np

# legacy id -> modern block name (approximate; unknown -> stone)
NAMES: dict[int, str] = {
    0: "minecraft:air", 1: "minecraft:stone", 2: "minecraft:grass_block",
    3: "minecraft:dirt", 4: "minecraft:cobblestone", 5: "minecraft:oak_planks",
    7: "minecraft:bedrock", 8: "minecraft:water", 9: "minecraft:water",
    10: "minecraft:lava", 11: "minecraft:lava", 12: "minecraft:sand",
    13: "minecraft:gravel", 14: "minecraft:gold_ore", 15: "minecraft:iron_ore",
    16: "minecraft:coal_ore", 17: "minecraft:oak_log", 18: "minecraft:oak_leaves",
    19: "minecraft:sponge", 20: "minecraft:glass", 21: "minecraft:lapis_ore",
    22: "minecraft:lapis_block", 23: "minecraft:dispenser",
    24: "minecraft:sandstone", 25: "minecraft:note_block",
    26: "minecraft:red_bed", 27: "minecraft:powered_rail",
    28: "minecraft:detector_rail", 29: "minecraft:sticky_piston",
    30: "minecraft:cobweb", 31: "minecraft:short_grass",
    32: "minecraft:dead_bush", 33: "minecraft:piston",
    35: "minecraft:white_wool", 37: "minecraft:dandelion",
    38: "minecraft:poppy", 39: "minecraft:brown_mushroom",
    40: "minecraft:red_mushroom", 41: "minecraft:gold_block",
    42: "minecraft:iron_block", 43: "minecraft:stone_slab",
    44: "minecraft:stone_slab", 45: "minecraft:bricks",
    46: "minecraft:tnt", 47: "minecraft:bookshelf",
    48: "minecraft:mossy_cobblestone", 49: "minecraft:obsidian",
    50: "minecraft:torch", 51: "minecraft:fire", 52: "minecraft:spawner",
    53: "minecraft:oak_stairs", 54: "minecraft:chest",
    56: "minecraft:diamond_ore", 57: "minecraft:diamond_block",
    58: "minecraft:crafting_table", 59: "minecraft:wheat",
    60: "minecraft:farmland", 61: "minecraft:furnace",
    63: "minecraft:oak_sign", 64: "minecraft:oak_door",
    65: "minecraft:ladder", 66: "minecraft:rail",
    67: "minecraft:cobblestone_stairs", 68: "minecraft:oak_wall_sign",
    69: "minecraft:lever", 70: "minecraft:stone_pressure_plate",
    71: "minecraft:iron_door", 72: "minecraft:oak_pressure_plate",
    73: "minecraft:redstone_ore", 76: "minecraft:redstone_torch",
    77: "minecraft:stone_button", 78: "minecraft:snow",
    79: "minecraft:ice", 80: "minecraft:snow_block", 81: "minecraft:cactus",
    82: "minecraft:clay", 83: "minecraft:sugar_cane",
    84: "minecraft:jukebox", 85: "minecraft:oak_fence",
    86: "minecraft:carved_pumpkin", 87: "minecraft:netherrack",
    88: "minecraft:soul_sand", 89: "minecraft:glowstone",
    91: "minecraft:jack_o_lantern", 96: "minecraft:oak_trapdoor",
    98: "minecraft:stone_bricks", 101: "minecraft:iron_bars",
    102: "minecraft:glass_pane", 103: "minecraft:melon",
    106: "minecraft:vines", 107: "minecraft:oak_fence_gate",
    108: "minecraft:brick_stairs", 109: "minecraft:stone_brick_stairs",
    110: "minecraft:mycelium", 111: "minecraft:lily_pad",
    112: "minecraft:nether_bricks", 113: "minecraft:nether_brick_fence",
    114: "minecraft:nether_brick_stairs", 116: "minecraft:enchanting_table",
    117: "minecraft:brewing_stand", 118: "minecraft:cauldron",
    120: "minecraft:end_portal_frame", 121: "minecraft:end_stone",
    122: "minecraft:dragon_egg", 123: "minecraft:redstone_lamp",
    125: "minecraft:oak_slab", 126: "minecraft:oak_slab",
    128: "minecraft:sandstone_stairs", 129: "minecraft:emerald_ore",
    130: "minecraft:ender_chest", 133: "minecraft:emerald_block",
    134: "minecraft:spruce_stairs", 135: "minecraft:birch_stairs",
    136: "minecraft:jungle_stairs", 137: "minecraft:command_block",
    138: "minecraft:beacon", 139: "minecraft:cobblestone_wall",
    140: "minecraft:flower_pot", 141: "minecraft:carrots",
    142: "minecraft:potatoes", 143: "minecraft:oak_button",
    145: "minecraft:anvil", 146: "minecraft:trapped_chest",
    152: "minecraft:redstone_block", 153: "minecraft:nether_quartz_ore",
    154: "minecraft:hopper", 155: "minecraft:quartz_block",
    156: "minecraft:quartz_stairs", 158: "minecraft:dropper",
    159: "minecraft:terracotta", 160: "minecraft:glass_pane",
    161: "minecraft:acacia_leaves", 162: "minecraft:acacia_log",
    163: "minecraft:acacia_stairs", 164: "minecraft:dark_oak_stairs",
    165: "minecraft:slime_block", 167: "minecraft:iron_trapdoor",
    168: "minecraft:prismarine", 169: "minecraft:sea_lantern",
    170: "minecraft:hay_block", 171: "minecraft:white_carpet",
    172: "minecraft:terracotta", 173: "minecraft:coal_block",
    174: "minecraft:packed_ice", 175: "minecraft:tall_flowers",
    179: "minecraft:red_sandstone", 180: "minecraft:red_sandstone_stairs",
}


def write_nbt(buf: bytearray, t: int, name: str, payload) -> None:  # noqa: ANN001
    buf.append(t)
    nb = name.encode("utf-8")
    buf += struct.pack(">h", len(nb)) + nb
    if t == 3:
        buf += struct.pack(">i", int(payload))
    elif t == 2:
        buf += struct.pack(">h", int(payload))
    elif t == 7:
        buf += struct.pack(">i", len(payload)) + bytes(payload)
    elif t == 11:
        buf += struct.pack(">i", len(payload))
        for v in payload:
            buf += struct.pack(">i", int(v))
    elif t == 8:
        sb = payload.encode("utf-8")
        buf += struct.pack(">h", len(sb)) + sb
    elif t == 10:
        for (st, sn, sp) in payload:
            write_nbt(buf, st, sn, sp)
        buf.append(0)
    else:
        raise ValueError(f"unhandled nbt type {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_dir")
    ap.add_argument("--name", default="valley_port")
    args = ap.parse_args()

    world = np.load(os.path.join(args.world_dir, "world_full.npz"))["world"]
    W, Hy, D = world.shape
    print(f"[schem] world {W}x{Hy}x{D}", flush=True)

    ids, counts = np.unique(world, return_counts=True)
    order = np.argsort(-counts)                 # common blocks -> low index
    ids = ids[order]
    pal = {int(b): NAMES.get(int(b), "minecraft:stone") for b in ids}
    idx_of = {int(b): i for i, b in enumerate(ids)}
    print(f"[schem] palette size {len(pal)}", flush=True)

    lut = np.zeros(65536, dtype=np.int32)
    for b, i in idx_of.items():
        lut[b] = i
    block_idx = lut[world]                      # (W, Hy, D)

    # Sponge order: x + z*W + y*W*D
    block_idx = block_idx.transpose(0, 2, 1).reshape(-1)

    # varint encode (vectorised): 1 byte for idx<128, 2 bytes otherwise
    one = block_idx < 128
    lengths = np.where(one, 1, 2).astype(np.int64)
    starts = np.cumsum(lengths) - lengths
    out = np.empty(int(lengths.sum()), dtype=np.uint8)
    m1 = ~one
    out[starts[one]] = block_idx[one].astype(np.uint8)
    out[starts[m1]] = ((block_idx[m1] & 0x7F) | 0x80).astype(np.uint8)
    out[starts[m1] + 1] = (block_idx[m1] >> 7).astype(np.uint8)

    buf = bytearray()
    buf += b"\x0a\x00\x00"                      # unnamed root compound
    write_nbt(buf, 3, "Version", 2)
    write_nbt(buf, 3, "DataVersion", 3465)      # 1.20.1
    write_nbt(buf, 2, "Width", W)
    write_nbt(buf, 2, "Height", Hy)
    write_nbt(buf, 2, "Length", D)
    write_nbt(buf, 11, "Offset", [0, 0, 0])
    write_nbt(buf, 3, "PaletteMax", len(pal))
    pal_tags = [(3, nm, idx_of[bid]) for bid, nm in pal.items()]
    write_nbt(buf, 10, "Palette", pal_tags)
    write_nbt(buf, 7, "BlockData", out.tobytes())
    buf.append(0)                               # close root compound

    dst = os.path.join(args.world_dir, f"{args.name}.schem")
    with gzip.open(dst, "wb") as fh:
        fh.write(buf)
    print(f"[schem] -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB gz)",
          flush=True)


if __name__ == "__main__":
    main()
