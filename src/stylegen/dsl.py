"""Skeleton-DSL v0 construction helpers for the stylegen suite.

The per-style modules turn a style grammar plus an RNG seed into a
Skeleton-DSL v0 JSON (docs/SKELETON-DSL.md). The JSON is compiled
UNCHANGED by the existing deterministic compiler
src/skeleton_compile.py (json -> voxel shell -> npz/png).

Conventions used by the generators (compiler semantics, verified against
src/skeleton_compile.py):

* wall segments are INCLUSIVE of both endpoints; ``openings.at`` is the
  cell offset from ``from`` and a hole covers ``[at, at+width)`` cells.
* room/floor/fill rects use x1/z1 EXCLUSIVE (slice semantics).
* rooms are carved BEFORE walls are rasterised, so wall material always
  wins over room carving.

DSL v0 has no solid-volume primitive (the compiler only writes wall
shells, floor slabs and the roof), so raised platforms / decks / steps /
eave tiers are emitted as FILL ROWS: 1-cell-wide wall segments whose
openings carve the complement of a y-band list. A fill entry must carry
ALL y-bands of its cells at once -- a later entry overlapping the same
cells rewrites them full height and would clobber earlier bands -- which
is why fill_row takes a band list, not a single band.
"""
from __future__ import annotations

MATS = ("mineral", "wood", "glass", "organic")


def opening(kind: str, at: int, width: int, height: int, y: int) -> dict:
    return {"kind": kind, "at": int(at), "width": int(width),
            "height": int(height), "y": int(y)}


def wall(x0: int, z0: int, x1: int, z1: int, storeys, openings=None) -> dict:
    return {"from": [int(x0), int(z0)], "to": [int(x1), int(z1)],
            "storeys": [int(s) for s in storeys],
            "openings": [dict(o) for o in (openings or [])]}


def column(x: int, z: int, storeys) -> dict:
    """1-cell pier: a zero-length wall segment rasterises to one cell."""
    return wall(x, z, x, z, storeys, [])


def _merge(bands):
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in bands):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def fill_row(x: int, z0: int, z1: int, y_bands, sh: int,
             storey_lo: int = 0) -> dict | None:
    """Row of cells (x, z0..z1) solid exactly inside y_bands (absolute y).

    y_bands = [(y_lo, y_hi), ...] half-open, must sit inside
    [storey_lo*sh, (storey_hi+1)*sh) where storey_hi is derived from the
    top band. Everything outside the bands is carved to air.
    """
    bands = _merge(y_bands)
    if not bands:
        return None
    lo = storey_lo * sh
    top_hi = max(b for _, b in bands)
    storey_hi = max(storey_lo, (top_hi - 1) // sh)
    span_top = (storey_hi + 1) * sh
    if bands[0][0] < lo:
        raise ValueError(f"fill band {bands[0]} below storey base {lo}")
    ops = []
    cur = lo
    for a, b in bands:
        if a > cur:
            ops.append(opening("window", 0, z1 - z0 + 1, a - cur, cur - lo))
        cur = b
    if cur < span_top:
        ops.append(opening("window", 0, z1 - z0 + 1, span_top - cur, cur - lo))
    return wall(x, z0, x, z1, range(storey_lo, storey_hi + 1), ops)


def fill_rect(rect, y_bands, sh: int, storey_lo: int = 0) -> list:
    """Fill rows covering rect (x0, z0, x1, z1), x1/z1 EXCLUSIVE."""
    x0, z0, x1, z1 = rect
    out = []
    for x in range(int(x0), int(x1)):
        row = fill_row(x, int(z0), int(z1) - 1, y_bands, sh, storey_lo)
        if row is not None:
            out.append(row)
    return out


def fill_cell(x: int, z: int, y_bands, sh: int, storey_lo: int = 0) -> list:
    row = fill_row(x, z, z, y_bands, sh, storey_lo)
    return [row] if row is not None else []


def room(rid: str, rtype: str, rect, storeys) -> dict:
    x0, z0, x1, z1 = rect
    return {"id": rid, "type": rtype,
            "rect": [int(x0), int(z0), int(x1), int(z1)],
            "storeys": [int(s) for s in storeys]}


def slab(storey: int, rect) -> dict:
    x0, z0, x1, z1 = rect
    return {"storey": int(storey), "rect": [int(x0), int(z0), int(x1), int(z1)]}


def roof(kind: str, pitch: int, ridge: str, overhang: int = 0) -> dict:
    return {"kind": kind, "overhang": int(overhang),
            "pitch": int(pitch), "ridge": ridge}


def build(bid: str, sh: int, w: int, d: int, rooms, walls, floors, rf,
          material) -> dict:
    return {
        "id": bid,
        "scale": {"unit": "block", "storey_height": int(sh)},
        "footprint": {"w": int(w), "d": int(d)},
        "rooms": list(rooms),
        "walls": list(walls),
        "floors": list(floors),
        "roof": dict(rf),
        "material": {
            "wall": material["wall"], "floor": material["floor"],
            "roof": material["roof"], "frame": material.get("frame", "wood"),
        },
    }


def validate(sk: dict) -> list:
    """Static sanity checks so the compiler run cannot fail or clip."""
    issues = []
    W, D = int(sk["footprint"]["w"]), int(sk["footprint"]["d"])
    sh = int(sk.get("scale", {}).get("storey_height", 4))
    rooms = sk.get("rooms", [])
    n_st = 1 + max((int(s) for r in rooms for s in r.get("storeys", [0])),
                   default=0)
    if W < 3 or D < 3:
        issues.append(f"footprint too small {W}x{D}")
    for r in rooms:
        x0, z0, x1, z1 = [int(v) for v in r["rect"]]
        if not (0 <= x0 < x1 <= W and 0 <= z0 < z1 <= D):
            issues.append(f"room {r['id']} rect {r['rect']} outside {W}x{D}")
        if not r.get("storeys"):
            issues.append(f"room {r['id']} has no storeys")
    for i, w in enumerate(sk.get("walls", [])):
        (ax, az), (bx, bz) = w["from"], w["to"]
        for px, pz in ((ax, az), (bx, bz)):
            if not (0 <= int(px) <= W and 0 <= int(pz) <= D):
                issues.append(f"wall {i} endpoint ({px},{pz}) outside {W}x{D}")
        steps = max(abs(int(bx) - int(ax)), abs(int(bz) - int(az)))
        sts = [int(s) for s in w.get("storeys", [0])]
        span = (max(sts) - min(sts) + 1) * sh
        for op in w.get("openings", []):
            wid, hei = int(op["width"]), int(op["height"])
            at, oy = int(op["at"]), int(op["y"])
            if at < 0 or at + wid > steps + 1:
                issues.append(f"wall {i} opening [{at},{at + wid}) exceeds "
                              f"segment cells 0..{steps}")
            if wid < 1 or hei < 1 or oy < 0:
                issues.append(f"wall {i} opening degenerate {op}")
            if oy + hei > span:
                issues.append(f"wall {i} opening y {oy}+{hei} exceeds "
                              f"storey span {span}")
    for key, val in sk.get("material", {}).items():
        if key != "frame" and val not in MATS:
            issues.append(f"material {key}={val} not in {MATS}")
    rf = sk.get("roof", {})
    if rf.get("kind") not in ("flat", "gable", "hip", "shed"):
        issues.append(f"roof kind {rf.get('kind')!r}")
    if int(rf.get("pitch", 1)) < 1:
        issues.append("roof pitch < 1")
    _ = n_st
    return issues
