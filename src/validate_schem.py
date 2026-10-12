"""Validate a Sponge .schem: structure, palette and BlockData length."""
import gzip
import struct
import sys

path = sys.argv[1]
raw = gzip.open(path, "rb").read()
print(f"decompressed {len(raw)} bytes")
assert raw[0] == 0x0a and raw[1:3] == b"\x00\x00", "root compound"
pos = 3
found = {}
while True:
    t = raw[pos]
    pos += 1
    if t == 0:
        break
    nlen = struct.unpack(">h", raw[pos:pos + 2])[0]
    pos += 2
    name = raw[pos:pos + nlen].decode()
    pos += nlen
    if t == 3:
        val = struct.unpack(">i", raw[pos:pos + 4])[0]
        pos += 4
    elif t == 2:
        val = struct.unpack(">h", raw[pos:pos + 2])[0]
        pos += 2
    elif t == 11:
        cnt = struct.unpack(">i", raw[pos:pos + 4])[0]
        pos += 4 + 4 * cnt
        val = cnt
    elif t == 7:
        cnt = struct.unpack(">i", raw[pos:pos + 4])[0]
        pos += 4 + cnt
        val = cnt
    elif t == 10:
        depth = 1
        start = pos
        while depth:
            tt = raw[pos]
            pos += 1
            if tt == 0:
                depth -= 1
                continue
            nn = struct.unpack(">h", raw[pos:pos + 2])[0]
            pos += 2 + nn
            if tt == 3:
                pos += 4
            elif tt == 8:
                sl = struct.unpack(">h", raw[pos:pos + 2])[0]
                pos += 2 + sl
            elif tt == 10:
                depth += 1
        val = "compound"
    elif t == 8:
        sl = struct.unpack(">h", raw[pos:pos + 2])[0]
        pos += 2 + sl
        val = sl
    else:
        raise SystemExit(f"unhandled tag type {t} at {pos}")
    found[name] = val

print(f"tags: {found}")
w, h, l = found["Width"], found["Height"], found["Length"]
n = w * h * l
bd = found["BlockData"]
print(f"blocks={n} blockdata_bytes={bd} palette_max={found['PaletteMax']} "
      f"version={found['Version']} dataversion={found['DataVersion']}")
assert bd >= n, "blockdata too short"
print("schem OK")
