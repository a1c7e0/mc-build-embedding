"""Honest layout metrics for generated plans.

All rates are computed from the actual coordinates, never assumed:

  degenerate_rate  fraction of rooms with x1<=x0 or z1<=z0
  overlap_rate     fraction of intra-house room PAIRS with intersection
                   area > 0 (each unordered pair counted once)
  oob_rate         fraction of rooms not fully inside the [0,1]^2 plan
                   canvas (coordinates live on the margin grid
                   [-0.1, 1.1], so out-of-canvas values are representable
                   and this is a live quantity)
  clean_house_rate fraction of houses with zero degenerate, zero overlap
                   and zero out-of-bounds rooms

Houses with <2 rooms contribute no pairs; overlap_rate is reported over
the pairs that exist (n_pairs is always disclosed alongside).

self_test() injects known-bad rooms and asserts each metric fires on
exactly the injected cases -- guard against a metric that silently
returns a constant.
"""
from __future__ import annotations

EPS = 1e-9


def layout_counts(rooms) -> dict:
    """rooms: iterable of (type, x0, z0, x1, z1) in normalized coords."""
    rs = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rooms]
    n = len(rs)
    deg = sum(1 for x0, z0, x1, z1 in rs
              if x1 <= x0 + EPS or z1 <= z0 + EPS)
    oob = sum(1 for x0, z0, x1, z1 in rs
              if x0 < -EPS or z0 < -EPS or x1 > 1 + EPS or z1 > 1 + EPS)
    pairs = 0
    ov = 0
    for i in range(n):
        ax0, az0, ax1, az1 = rs[i]
        for j in range(i + 1, n):
            bx0, bz0, bx1, bz1 = rs[j]
            pairs += 1
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(az1, bz1) - max(az0, bz0)
            if iw > EPS and ih > EPS:
                ov += 1
    return {"rooms": n, "pairs": pairs, "degenerate": deg,
            "overlap_pairs": ov, "oob": oob}


def aggregate(counts_list) -> dict:
    H = len(counts_list)
    R = sum(c["rooms"] for c in counts_list)
    P = sum(c["pairs"] for c in counts_list)
    D = sum(c["degenerate"] for c in counts_list)
    O = sum(c["overlap_pairs"] for c in counts_list)
    B = sum(c["oob"] for c in counts_list)
    clean = sum(1 for c in counts_list
                if c["degenerate"] == 0 and c["overlap_pairs"] == 0
                and c["oob"] == 0)
    return {
        "houses": H,
        "rooms": R,
        "pairs": P,
        "degenerate_rate": D / R if R else 0.0,
        "overlap_rate": O / P if P else 0.0,
        "oob_rate": B / R if R else 0.0,
        "clean_house_rate": clean / H if H else 0.0,
    }


def self_test() -> dict:
    """Inject known-bad layouts; every metric must fire on exactly those."""
    good = [("Bedroom", 0.05, 0.05, 0.45, 0.45),
            ("Kitchen", 0.55, 0.55, 0.95, 0.95)]
    bad = [("Bedroom", 0.50, 0.50, 0.50, 0.90),   # degenerate: x1 == x0
           ("Kitchen", 0.10, 0.10, 0.60, 0.60),   # pair 1-2 overlaps
           ("Kitchen", 0.30, 0.30, 0.80, 0.80),
           ("Bathroom", 0.90, 0.90, 1.20, 1.20)]  # out of bounds
    cg = layout_counts(good)
    cb = layout_counts(bad)
    checks = {
        "good_all_zero": (cg["degenerate"] == 0 and cg["overlap_pairs"] == 0
                          and cg["oob"] == 0),
        "good_pairs": cg["pairs"] == 1,
        "bad_degenerate_exact": cb["degenerate"] == 1,
        "bad_overlap_exact": cb["overlap_pairs"] == 1,
        "bad_oob_exact": cb["oob"] == 1,
        "bad_pairs_exact": cb["pairs"] == 6,
    }
    out = {"pass": all(checks.values()), "checks": checks,
           "good": cg, "bad": cb}
    assert out["pass"], f"metrics self-test failed: {out}"
    return out
