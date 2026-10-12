"""Zero-dependency readers for TFRecord + protobuf Example + schematic NBT.

Formats:
- TFRecord:  u64 len | u32 crc | data[len] | u32 crc
- tf.train.Example: {features{feature: map<string, Feature>}},
  Feature{bytes_list{value: repeated bytes}} — this dataset stores 'url' and
  'schematicData' (raw schematic file bytes).
- .schematic (MCEdit): gzip NBT, root 'Schematic', Blocks byte array indexed
  (y*Length + z)*Width + x, Data byte array (block metadata).
- .schem (Sponge v2): gzip NBT, Palette compound + BlockData varints,
  indexed x + z*Width + y*Width*Length.
"""
from __future__ import annotations

import gzip
import struct


# ------------------------------------------------------------- TFRecord ---
def iter_tfrecord(path):
    with open(path, "rb") as fh:
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            (length,) = struct.unpack("<Q", head)
            fh.read(4)                       # crc of length (unchecked)
            data = fh.read(length)
            fh.read(4)                       # crc of data
            if len(data) < length:
                return
            yield data


# ------------------------------------------------------------ protobuf ----
def _varint(buf: bytes, pos: int):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _varint(buf, pos)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:
            val, pos = _varint(buf, pos)
        elif wtype == 2:
            ln, pos = _varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wtype == 5:
            val = buf[pos:pos + 4]
            pos += 4
        elif wtype == 1:
            val = buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"wire type {wtype}")
        yield fnum, wtype, val


def example_features(record: bytes) -> dict[str, list[bytes]]:
    feats: dict[str, list[bytes]] = {}
    for fnum, wtype, val in _fields(record):
        if fnum != 1 or wtype != 2:                # Features
            continue
        for en, et, ev in _fields(val):            # map<string, Feature> entry
            if en != 1 or et != 2:
                continue
            key, blobs = None, []
            for fn, ft, fv in _fields(ev):         # entry: key=1, value=2
                if fn == 1 and ft == 2:
                    key = fv.decode("utf-8", "replace")
                elif fn == 2 and ft == 2:          # Feature
                    for gn, gt, gv in _fields(fv):
                        if gn == 1 and gt == 2:    # BytesList
                            for bn, bt, bv in _fields(gv):
                                if bn == 1 and bt == 2:
                                    blobs.append(bv)
            if key is not None:
                feats[key] = blobs
    return feats


# ----------------------------------------------------------------- NBT ----
def _nbt_payload(buf: bytes, pos: int, ttype: int):  # noqa: ANN001
    if ttype == 1:
        return buf[pos], pos + 1
    if ttype == 2:
        return struct.unpack_from(">h", buf, pos)[0], pos + 2
    if ttype == 3:
        return struct.unpack_from(">i", buf, pos)[0], pos + 4
    if ttype == 4:
        return struct.unpack_from(">q", buf, pos)[0], pos + 8
    if ttype == 5:
        return struct.unpack_from(">f", buf, pos)[0], pos + 4
    if ttype == 6:
        return struct.unpack_from(">d", buf, pos)[0], pos + 8
    if ttype == 7:
        ln = struct.unpack_from(">i", buf, pos)[0]
        pos += 4
        return buf[pos:pos + ln], pos + ln
    if ttype == 8:
        ln = struct.unpack_from(">H", buf, pos)[0]
        pos += 2
        return buf[pos:pos + ln].decode("utf-8", "replace"), pos + ln
    if ttype == 9:
        et = buf[pos]
        cnt = struct.unpack_from(">i", buf, pos + 1)[0]
        pos += 5
        out = []
        for _ in range(cnt):
            v, pos = _nbt_payload(buf, pos, et)
            out.append(v)
        return out, pos
    if ttype == 10:
        out = {}
        while True:
            tt = buf[pos]
            pos += 1
            if tt == 0:
                return out, pos
            ln = struct.unpack_from(">H", buf, pos)[0]
            name = buf[pos + 2:pos + 2 + ln].decode("utf-8", "replace")
            pos += 2 + ln
            v, pos = _nbt_payload(buf, pos, tt)
            out[name] = v
    if ttype == 11:
        cnt = struct.unpack_from(">i", buf, pos)[0]
        pos += 4
        out = list(struct.unpack_from(f">{cnt}i", buf, pos))
        return out, pos + 4 * cnt
    raise ValueError(f"nbt type {ttype}")


def nbt_parse(data: bytes) -> dict:
    """Parse gzipped-or-raw NBT; returns {name: payload} of the root compound."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    assert data[0] == 10, f"root not compound (tag {data[0]})"
    ln = struct.unpack_from(">H", data, 1)[0]
    pos = 3 + ln
    root, _ = _nbt_payload(data, pos, 10)
    return root


# -------------------------------------------------- schematic -> voxels ----
def schematic_to_grid(root: dict):  # noqa: ANN201
    """Returns (grid uint8 [x, y, z], palette_names dict or None)."""
    import numpy as np

    if "Palette" in root and "BlockData" in root:        # Sponge v2
        W = int(root["Width"])
        Hy = int(root["Height"])
        D = int(root["Length"])
        pal = {int(v): k for k, v in root["Palette"].items()}
        data = root["BlockData"]
        vals = np.empty(W * Hy * D, dtype=np.int32)
        pos = i = 0
        while i < vals.size:
            v = 0
            shift = 0
            while True:
                b = data[pos]
                pos += 1
                v |= (b & 0x7F) << shift
                if not b & 0x80:
                    break
                shift += 7
            vals[i] = v
            i += 1
        arr = vals.reshape(Hy, D, W).transpose(2, 0, 1)  # y,z,x -> x,y,z
        lut = np.zeros(max(pal) + 1 if pal else 1, dtype=np.uint8)
        for idx, name in pal.items():
            lut[idx] = _NAME_TO_LEGACY.get(name, 1)
        return lut[arr], pal

    W = int(root["Width"])
    Hy = int(root["Height"])
    D = int(root["Length"])
    blocks = np.frombuffer(root["Blocks"], dtype=np.uint8)
    arr = blocks.reshape(Hy, D, W).transpose(2, 0, 1)    # y,z,x -> x,y,z
    return np.ascontiguousarray(arr), None


# minimal modern-name -> legacy-id map for Sponge files
_NAME_TO_LEGACY = {
    "minecraft:air": 0, "minecraft:stone": 1, "minecraft:grass_block": 2,
    "minecraft:dirt": 3, "minecraft:cobblestone": 4, "minecraft:oak_planks": 5,
    "minecraft:bedrock": 7, "minecraft:water": 9, "minecraft:lava": 11,
    "minecraft:sand": 12, "minecraft:gravel": 13, "minecraft:oak_log": 17,
    "minecraft:oak_leaves": 18, "minecraft:glass": 20, "minecraft:bricks": 45,
    "minecraft:stone_bricks": 98, "minecraft:quartz_block": 155,
    "minecraft:sandstone": 24, "minecraft:terracotta": 172,
}
