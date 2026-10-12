"""Terrain v2.1: feature-driven, erosion-coupled world generation.

Pipeline (heightmap is FINALISED before any blocks are written — this is what
prevents floating pillars / hanging blocks):
  1. base relief      ridged multifractal (mountains) + fBm lowlands
  2. spec features    mountains / basins / plateaus (prompt-controlled)
  3. river            meandering polyline, variable width, parabolic banks
  4. thermal erosion  talus-angle relaxation (natural slopes & foothills)
  5. hydraulic erosion  vectorised droplets (gullies, coupled valleys)
  6. human layer      road levelling + farm terraces (Ht only)
  7. biomes           water / beach / plain / forest / rock with dithering
  8. block fill       single pass from final Ht + top-material overrides

Outputs (out/world/<name>/): world.npz, terrain.npz, layout.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.ndimage import (zoom, gaussian_filter, median_filter, label,
                           maximum_filter, minimum_filter)

AIR, STONE, GRASS, DIRT, SAND, WATER, LOG, LEAVES = 0, 1, 2, 3, 12, 9, 17, 18
GRAVEL, FARMLAND, WHEAT, STONE_BRICK = 13, 60, 59, 98
SANDSTONE, FENCE, GLOWSTONE = 24, 85, 89
BIOME = {"water": 0, "beach": 1, "plain": 2, "forest": 3, "rock": 4}


# ---------------------------------------------------------------- noise ---
def fbm(shape, rng, octaves=6, base=3, gain=0.55):
    out = np.zeros(shape, dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = base * 2 ** o
        g = rng.random((res + 2, res + 2)).astype(np.float32)
        z = zoom(g, (shape[0] / g.shape[0], shape[1] / g.shape[1]), order=3)
        z = z[:shape[0], :shape[1]]
        if z.shape != shape:
            z = np.pad(z, ((0, shape[0] - z.shape[0]),
                           (0, shape[1] - z.shape[1])), mode="edge")
        out += amp * z
        norm += amp
        amp *= gain
    out /= norm
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def ridged(shape, rng, octaves=6, base=3):
    out = np.zeros(shape, dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = base * 2 ** o
        g = rng.random((res + 2, res + 2)).astype(np.float32)
        z = zoom(g, (shape[0] / g.shape[0], shape[1] / g.shape[1]), order=3)
        z = z[:shape[0], :shape[1]]
        if z.shape != shape:
            z = np.pad(z, ((0, shape[0] - z.shape[0]),
                           (0, shape[1] - z.shape[1])), mode="edge")
        z = 1.0 - np.abs(2.0 * z - 1.0)      # ridges
        out += amp * z * z
        norm += amp
        amp *= 0.5
    out /= norm
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- rivers ---
def meander(pts: np.ndarray, rng, subdivisions: int = 6,
            wobble: float = 0.035, span: float = 1.0) -> np.ndarray:
    """Subdivide a polyline and push vertices with smooth noise -> meanders."""
    out = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for s in range(subdivisions):
            t = s / subdivisions
            out.append(a * (1 - t) + b * t)
    out.append(pts[-1])
    P = np.asarray(out, dtype=np.float32)
    n = len(P)
    phase = rng.uniform(0, 2 * np.pi)
    idx = np.arange(n)
    for axis in range(2):
        w = (np.sin(idx * 0.11 + phase + axis * 1.7) * 0.6 +
             np.sin(idx * 0.031 + phase * 1.3 + axis * 2.9) * 0.4)
        P[:, axis] += w * wobble * span      # amplitude ~ wobble*span blocks
    return P


def dist_to_polyline(X, Y, pts):
    best = np.full(X.shape, 1e9, dtype=np.float32)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-9
        t = np.clip(((X - ax) * vx + (Y - ay) * vy) / L2, 0, 1)
        d2 = (X - ax - t * vx) ** 2 + (Y - ay - t * vy) ** 2
        np.minimum(best, np.sqrt(d2), out=best)
    return best


# --------------------------------------------------------------- erosion ---
def thermal_erosion(H: np.ndarray, iters: int = 14, talus: float = 1.6,
                    rate: float = 0.45):
    for _ in range(iters):
        out = np.zeros_like(H)
        dep = np.zeros_like(H)
        for ax, sh in ((0, (1, 0)), (0, (-1, 0)), (1, (0, 1)), (1, (0, -1))):
            rolled = np.roll(H, sh, axis=ax)
            d = H - rolled
            m = np.clip(d - talus, 0, None) * rate * 0.25
            out += m
            dep += np.roll(m, (-sh[0], -sh[1]), axis=ax)
        H += dep - out
    return H


def hydraulic_erosion(H: np.ndarray, rng, droplets: int = 24000,
                      steps: int = 42):
    """Vectorised droplet erosion (Lague-style). Carves gullies and couples
    valleys / river banks to the surrounding terrain."""
    W, D = H.shape
    gx = np.gradient(H, axis=0)
    gy = np.gradient(H, axis=1)

    px = rng.uniform(2, W - 3, droplets).astype(np.float32)
    py = rng.uniform(2, D - 3, droplets).astype(np.float32)
    dx = np.zeros(droplets, np.float32)
    dy = np.zeros(droplets, np.float32)
    speed = np.ones(droplets, np.float32)
    water = np.ones(droplets, np.float32)
    sed = np.zeros(droplets, np.float32)
    inertia, cap_f, erode_f, dep_f, evap = 0.05, 3.2, 0.35, 0.22, 0.012

    for _ in range(steps):
        xi = np.clip(px.astype(int), 1, W - 2)
        yi = np.clip(py.astype(int), 1, D - 2)
        fx, fy = px - xi, py - yi
        h_old = (H[xi, yi] * (1 - fx) * (1 - fy) + H[xi + 1, yi] * fx * (1 - fy) +
                 H[xi, yi + 1] * (1 - fx) * fy + H[xi + 1, yi + 1] * fx * fy)
        grad_x = (gx[xi, yi] * (1 - fy) + gx[xi + 1, yi] * fx)
        grad_y = (gy[xi, yi] * (1 - fx) + gy[xi, yi + 1] * fx)

        dx = dx * inertia - grad_x * (1 - inertia)
        dy = dy * inertia - grad_y * (1 - inertia)
        norm = np.sqrt(dx * dx + dy * dy) + 1e-9
        dx /= norm
        dy /= norm
        px += dx
        py += dy
        bad = (px < 2) | (px > W - 3) | (py < 2) | (py > D - 3)
        px = np.clip(px, 2, W - 3)
        py = np.clip(py, 2, D - 3)

        xi, yi = px.astype(int), py.astype(int)
        fx, fy = px - xi, py - yi
        h_new = (H[xi, yi] * (1 - fx) * (1 - fy) + H[xi + 1, yi] * fx * (1 - fy) +
                 H[xi, yi + 1] * (1 - fx) * fy + H[xi + 1, yi + 1] * fx * fy)
        dh = h_new - h_old

        cap = np.maximum(-dh, 0.01) * speed * water * cap_f
        deposit = np.where((sed > cap) | (dh > 0),
                           np.where(dh > 0, np.minimum(dh, sed),
                                    (sed - cap) * dep_f), 0)
        sed -= deposit
        np.add.at(H, (xi, yi), deposit)

        eroded = np.where((sed <= cap) & (dh <= 0) & ~bad,
                          np.minimum((cap - sed) * erode_f, -dh * 0.9), 0)
        sed += eroded
        np.add.at(H, (xi, yi), -eroded)

        speed = np.sqrt(np.abs(speed ** 2 + dh * -2.2))
        water *= (1 - evap)
    return H


# ------------------------------------------------------- floating islands ---
def crack_map(s: int, rng, spacing: int = 18):  # noqa: ANN001
    """Voronoi crack map (ported from AjayaRamachandran/Procedural-Terrain-
    Generator generateCrackMap + the deep-layer erosion in generateIslandShape):
    cells split by Voronoi agents; blocks near cell boundaries get eroded
    with depth, giving the underside its fractured-slab look."""
    n = max(2, round((s / spacing) ** 2))
    agents = rng.random((n, 2)).astype(np.float32) * s
    xg = np.arange(s, dtype=np.float32)[:, None, None]
    zg = np.arange(s, dtype=np.float32)[None, :, None]
    d2 = (xg - agents[:, 0]) ** 2 + (zg - agents[:, 1]) ** 2
    part = np.partition(d2, 1, axis=-1)
    return (part[..., 1] - part[..., 0]) < (spacing * 0.35) ** 2


def build_floating_island(r: int, thickness: int, rng):  # noqa: ANN001
    """Noisy disc top + tapered rocky underside. A straight downward
    extrusion would read as a cylinder; the taper is what sells 'floating'.
    Only the largest connected blob survives (no stray floating fragments).
    Returns (grid [x,y,z], top_max, top_map, inside_map)."""
    s = 2 * r + 1
    top_span = max(thickness // 3, 3)
    g = None
    Xc = np.arange(s, dtype=np.float32)[:, None] - r
    Zc = np.arange(s, dtype=np.float32)[None, :] - r
    dd = np.sqrt(Xc ** 2 + Zc ** 2) / max(r, 1)
    wob = 0.85 + 0.30 * fbm((s, s), rng, octaves=4, base=4)
    inside = dd < wob
    lab, nlab = label(inside)
    if nlab > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        inside = lab == int(sizes.argmax())
    rel = np.clip(dd / np.maximum(wob, 1e-6), 0, 1)
    # crack map thins the plates at voronoi boundaries (Ajaya's depth cut per
    # column) -> stepped slab underside with vertical seams, no fragmentation
    edge = crack_map(s, rng)
    depth_scale = np.where(edge, 0.55 + 0.20 * rng.random((s, s)), 1.0)
    top = (top_span + fbm((s, s), rng, octaves=4, base=5) * top_span).astype(int)
    depth = ((1.0 - rel) ** 1.1 * thickness * 1.4 * depth_scale).astype(int)
    bottom = top - depth            # may go negative: full cone, no clipping
    shift = -int(min(bottom.min(), 0))
    bottom = bottom + shift
    top = top + shift
    gy = int(top.max()) + 2
    g = np.zeros((s, gy, s), dtype=np.uint16)
    ys = np.arange(gy, dtype=np.int32)[None, :, None]
    inside_c = inside[:, None, :]
    top_c = top[:, None, :]
    bottom_c = bottom[:, None, :]
    fill = (ys >= bottom_c) & (ys <= top_c) & inside_c
    g[fill] = STONE
    top_m = (ys == top_c) & inside_c
    g[top_m] = GRASS
    dirt_m = (ys >= top_c - 2) & (ys < top_c) & inside_c
    g[dirt_m] = DIRT
    return g, int(top.max()), top, inside


# ------------------------------------------------------------------ main ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    spec = json.load(open(args.spec, encoding="utf-8"))
    rng = np.random.default_rng(spec.get("seed", 0))

    W, D = spec["size"]
    H = spec["world_height"]
    wl = spec["water_level"]
    t = spec["terrain"]
    out_dir = os.path.join(args.out, "world", spec["name"])
    os.makedirs(out_dir, exist_ok=True)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(D, dtype=np.float32), indexing="ij")

    # 1-2. base relief + prompt features
    low = fbm((W, D), rng, octaves=3, base=2)
    detail = fbm((W, D), rng, octaves=6, base=5)
    gx = np.linspace(0, 1, W, dtype=np.float32)[:, None]
    gy = np.linspace(0, 1, D, dtype=np.float32)[None, :]
    gradient = 1.0 - (gx * 0.55 + gy * 0.55)
    height = 14 + (gradient * 0.42 + low * 0.36 + detail * 0.22) * t["relief"] * 1.7
    extra_water = []

    for feat in spec.get("features", []):
        cx, cy = np.asarray(feat["center"], dtype=np.float32) * np.array([W, D])
        r = feat["radius"] * W
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r
        mask = 1.0 - smoothstep(0.3, 1.05, d)
        if feat["type"] == "mountain":
            rf = ridged((W, D), rng, octaves=7, base=3)
            spur = fbm((W, D), rng, octaves=4, base=8)
            height += mask * feat["height"] * (0.35 + 0.5 * rf + 0.3 * spur)
        elif feat["type"] == "basin":
            height -= mask * feat["depth"] * (0.7 + 0.3 * low)
        elif feat["type"] == "plateau":
            height += mask * feat["height"] * (0.92 + 0.08 * detail)
        elif feat["type"] == "atoll":
            # ring island in an ocean: noise-wobbled radius, lagoon in the
            # middle, and the surroundings sink below water level
            ring_r = feat.get("ring_radius", 0.62)
            wob = 1.0 + 0.5 * (fbm((W, D), rng, octaves=4, base=7) - 0.5)
            band = np.exp(-((d / wob - ring_r) ** 2) / (2 * 0.20 ** 2))
            height += mask * feat["height"] * band
            lagoon = 1.0 - smoothstep(0.0, ring_r * 0.95, d / wob)
            # absolute lagoon floor below water level (depth-relative carving
            # can leave the lagoon dry on high base terrain)
            height = height * (1 - lagoon) + (wl - 2.5) * lagoon
            extra_water.append(lagoon > 0.5)
            # oceanise the surroundings so the ring reads as an atoll
            oceanize = smoothstep(1.0, 1.7, d)
            height = height * (1 - oceanize) + (wl - 3.5) * oceanize
            extra_water.append(oceanize > 0.5)

    # 3. river
    pts = np.asarray(t["river"], dtype=np.float32) * np.array([W, D])
    pts = meander(pts, rng, wobble=t.get("meander", 0.035), span=W)
    d_riv = dist_to_polyline(X, Y, pts)
    along = np.clip((X / W * 0.6 + Y / D * 0.6), 0, 1.2)     # widens seaward
    rw = t["river_width"] * (0.6 + 0.9 * along)
    cross = np.clip(d_riv / rw, 0, 2.5)
    carve = np.clip(1 - cross ** 2, 0, 1) ** 1.3
    height -= t["river_depth"] * 2.6 * carve
    height -= 5.5 * np.exp(-(d_riv / (rw * 2.8)) ** 2)       # broad valley

    height = np.clip(height, 6, H - 18)

    # 4-5. erosion
    height = thermal_erosion(height, iters=12, talus=1.7, rate=0.5)
    height = hydraulic_erosion(height, rng,
                               droplets=t.get("droplets", 22000))
    height = thermal_erosion(height, iters=6, talus=1.5, rate=0.35)

    # soft river banks: near the channel, blend towards locally smoothed terrain
    bank = np.exp(-(d_riv / (rw * 2.2)) ** 2)
    height = height * (1 - 0.45 * bank) + gaussian_filter(height, 3.0) * (0.45 * bank)
    height -= 3.4 * carve          # re-assert the channel after erosion
    height = gaussian_filter(height, 0.5)

    # sea corner
    sea = np.clip((gx * 0.7 + gy * 0.7) - 1.04, 0, 1) * 46
    height -= sea

    # spike/garbage kill: NaN/inf streaks (e.g. from polyline degeneracies)
    # clip to the ceiling as 114-tall pillars; small/narrow crests are artifacts
    height = np.nan_to_num(height, nan=float(wl), posinf=float(H - 30),
                           neginf=4.0)
    med9 = median_filter(height, size=9)
    diff = height - med9
    spike_lab, _ = label(np.abs(diff) > 5)
    sizes = np.bincount(spike_lab.ravel())
    small = np.isin(spike_lab, np.nonzero(sizes < 250)[0])
    height[small] = med9[small] + np.sign(diff[small]) * 1.5
    height = np.minimum(height, H - 26)

    # border repair: np.roll-based erosion wraps edges; mirror the rim inward
    height[:2, :] = height[2:4, :][::-1]
    height[-2:, :] = height[-3:-1, :][::-1]
    height[:, :2] = height[:, 2:4][:, ::-1]
    height[:, -2:] = height[:, -3:-1][:, ::-1]
    height = gaussian_filter(height, 0.8)

    height = np.round(np.clip(height, 4, H - 14))
    Ht = height.astype(np.int32)

    # smooth river cross-section: blend the channel bed over a tapered band
    # (a hard clamp leaves vertical wall/pillar artefacts along the banks)
    ch_mask = 1.0 - smoothstep(rw * 0.5, rw * 2.6, d_riv)
    bed = wl - 2 + 0.8 * (fbm((W, D), rng, octaves=3, base=12) - 0.5)  # rippled bed
    Ht = np.round(Ht * (1 - ch_mask) + np.minimum(Ht, bed) * ch_mask).astype(np.int32)

    # hydrology sanity: standing water only in the river corridor, the sea or
    # declared lagoons (atolls); erosion pits must not become random ponds
    water_zone = (d_riv < rw * 1.3) | ((gx * 0.7 + gy * 0.7) > 1.02)
    for m in extra_water:
        water_zone |= np.asarray(m) > 0.5
    Ht = np.where((Ht < wl + 4) & ~water_zone, wl + 4, Ht)

    # surface polish: kill 1-2 block deposit noise on flats while keeping
    # landforms (low-relief areas get truly flat, so no 1-block "pillar" steps)
    med5 = median_filter(Ht, size=5)
    local_relief = (maximum_filter(Ht, size=5) - minimum_filter(Ht, size=5))
    flat = local_relief <= 3
    Ht = np.where(flat, med5, np.round(0.65 * Ht + 0.35 * med5)).astype(np.int32)
    Ht = np.where((Ht < wl + 4) & ~water_zone, wl + 4, Ht)

    # 5b. shoals: noisy sandbars just below the surface near coasts
    shoal = ((fbm((W, D), rng, octaves=4, base=9) > 0.62)
             & (Ht >= wl - 4) & (Ht < wl - 1))
    Ht[shoal] = wl - 1

    # 6. human layer on Ht only (blocks come later, single pass)
    road_mask = np.zeros((W, D), dtype=bool)
    road_main = np.zeros((W, D), dtype=bool)
    roads_out = []
    road_specs = list(spec.get("roads", []))
    # auto network: MST over districts/landmarks/farms (connective tissue)
    if spec.get("roads_auto", True):
        anchors = [d["center"] for d in spec.get("districts", [])]
        anchors += [lm["at"] for lm in spec.get("landmarks", []) if "at" in lm]
        anchors += [f["center"]
                    for f in spec.get("decoration", {}).get("farm", [])]
        if len(anchors) >= 2:
            P = np.asarray(anchors, dtype=np.float32)
            n = len(P)
            in_tree = [0]
            while len(in_tree) < n:
                best = None
                for i in in_tree:
                    for j in range(n):
                        if j in in_tree:
                            continue
                        d2 = float(((P[i] - P[j]) ** 2).sum())
                        if best is None or d2 < best[0]:
                            best = (d2, i, j)
                road_specs.append({"name": f"auto_{len(road_specs)}",
                                   "width": 2,
                                   "points": [anchors[best[1]],
                                              anchors[best[2]]]})
                in_tree.append(best[2])
    # atoll ring roads
    for feat in spec.get("features", []):
        if feat["type"] == "atoll" and feat.get("ring_road", True):
            acx, acy = feat["center"]
            rr = feat["radius"] * feat.get("ring_radius", 0.68)
            circle = [[acx + rr * np.cos(t), acy + rr * np.sin(t)]
                      for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
            road_specs.append({"name": "ring_road", "width": 2,
                               "points": circle + [circle[0]]})
    for road in road_specs:
        if "follow_river" in road:
            # hug the river bank at an offset -> roads inherit the valley shape
            off = road["follow_river"]["offset"]
            seg = []
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                d = b - a
                n = np.array([-d[1], d[0]])
                n = n / (np.linalg.norm(n) + 1e-9)
                for frac in (0.0, 0.5):
                    seg.append(a * (1 - frac) + b * frac + n * off)
            seg.append(pts[-1] + np.array([-1, 1]) * off / 1.4)
            rpts = np.asarray(seg, dtype=np.float32)
        else:
            rpts = np.asarray(road["points"], dtype=np.float32) * np.array([W, D])
            rpts = meander(rpts, rng, subdivisions=4, wobble=0.012, span=W)
        d = dist_to_polyline(X, Y, rpts)
        strip = (d < road["width"]) & (Ht > wl + 1)
        idx = np.nonzero(strip)
        for k in range(len(idx[0])):
            x, z = int(idx[0][k]), int(idx[1][k])
            hh = Ht[max(x - 2, 0):x + 3, max(z - 2, 0):z + 3]
            Ht[x, z] = int(np.median(hh))
        road_mask |= strip
        if road.get("name") != "ring_road":
            road_main |= strip
        roads_out.append({"name": road["name"],
                          "points": [[float(a), float(b)] for a, b in rpts[::4]],
                          "width": road["width"]})

    farm_mask = np.zeros((W, D), dtype=bool)
    wheat_mask = np.zeros((W, D), dtype=bool)
    farms = []
    for f in spec.get("decoration", {}).get("farm", []):
        fx, fy = np.asarray(f["center"], dtype=np.float32) * np.array([W, D])
        fr = f["radius"] * W
        farms.append({"x": int(fx), "z": int(fy), "r": int(fr)})
        m = (((X - fx) ** 2 + (Y - fy) ** 2) < fr * fr) & (Ht > wl + 2)
        idx = np.nonzero(m)
        for k in range(len(idx[0])):
            x, z = int(idx[0][k]), int(idx[1][k])
            Ht[x, z] = int(round(Ht[x, z] / 3.0) * 3)      # terrace
            farm_mask[x, z] = True
            if (x // 4) % 2 == 0 and (z // 2) % 2 == 0:
                wheat_mask[x, z] = True

    # 7. biomes (dithered thresholds -> mottled transitions)
    dither = fbm((W, D), rng, octaves=5, base=10) - 0.5
    slope = np.maximum.reduce([
        np.abs(np.diff(Ht, axis=0, prepend=Ht[:1])),
        np.abs(np.diff(Ht, axis=1, prepend=Ht[:, :1]))])
    biome = np.full((W, D), BIOME["plain"], dtype=np.uint8)
    biome[Ht < wl] = BIOME["water"]
    beach_lvl = wl + 3 + dither * 2.4
    near_water = gaussian_filter((Ht < wl).astype(np.float32), 6.0) > 0.015
    beach = (Ht <= beach_lvl) & (biome != BIOME["water"]) & near_water
    biome[beach] = BIOME["beach"]
    forest_n = gaussian_filter(fbm((W, D), rng, octaves=5, base=6), 3.2)
    riparian = smoothstep(4, 26, d_riv)          # 0 near water -> 1 far
    forest_score = forest_n + 0.18 * (1 - riparian) + 0.10 * smoothstep(2, 6, slope)
    biome[(forest_score > 0.54 + dither * 0.18) &
          (biome == BIOME["plain"])] = BIOME["forest"]
    biome[(slope + dither * 2.0 > 5) | (Ht > wl + t["relief"] * 1.35)] = BIOME["rock"]

    # 8. single-pass block fill from final Ht
    world = np.zeros((W, H, D), dtype=np.uint16)
    ys = np.arange(H, dtype=np.int32)[None, :, None]
    ground = Ht[:, None, :]
    below = ys <= ground
    world[below] = STONE
    top = ys == ground
    world[top] = GRASS
    sand_b = np.broadcast_to(beach[:, None, :], world.shape)
    world[top & sand_b] = SAND
    world[(ys >= ground - 3) & (ys < ground) & sand_b] = SAND
    water_b = np.broadcast_to((Ht < wl)[:, None, :] &
                              (ys > ground) & (ys <= wl), world.shape)
    world[water_b] = WATER

    # top-material overrides
    # top-material overrides: warm sandstone for main roads, gravel for ring
    idx = np.nonzero(road_main & (Ht > wl))
    world[idx[0], Ht[idx], idx[1]] = SANDSTONE
    idx = np.nonzero(road_mask & ~road_main & (Ht > wl))
    world[idx[0], Ht[idx], idx[1]] = GRAVEL
    # roadside lanterns: fence post + glowstone cap along main roads
    ridx = np.nonzero(road_main & (Ht > wl))
    for k in range(0, len(ridx[0]), 60):
        x, z = int(ridx[0][k]), int(ridx[1][k])
        h = int(Ht[x, z])
        if h + 3 < H:
            world[x, h + 1:h + 3, z] = FENCE
            world[x, h + 3, z] = GLOWSTONE
    idx = np.nonzero(farm_mask & (Ht > wl + 1))
    world[idx[0], Ht[idx], idx[1]] = FARMLAND
    idx = np.nonzero(wheat_mask & (Ht > wl + 1))
    yw = np.minimum(Ht[idx] + 1, H - 1)
    world[idx[0], yw, idx[1]] = WHEAT

    # floating islands (3D sky features; heightmap cannot express them)
    islands_out = []
    island_slots = []
    for feat in spec.get("features", []):
        if feat["type"] != "floating_island":
            continue
        cx, cy = (np.asarray(feat["center"], dtype=np.float32) *
                  np.array([W, D])).astype(int)
        r = max(int(feat["radius"] * W), 6)
        th = int(feat.get("thickness", 16))
        alt = int(feat.get("altitude", 55))
        gi, top_y, top_map, inside_map = build_floating_island(r, th, rng)
        x0, z0, y0 = cx - r, cy - r, alt - top_y
        sx0, sy0, sz0 = gi.shape
        # never let the cone tip dig into the ground below: enforce clearance
        ground = int(Ht[max(cx, 0):min(cx + 1, W),
                        max(cy, 0):min(cy + 1, D)][0, 0])
        if y0 < ground + 14:
            y0 = ground + 14
        if y0 + sy0 > H - 2:
            y0 = H - sy0 - 2
        ok = (0 <= x0 and x0 + sx0 <= W and 0 <= z0 and z0 + sz0 <= D
              and y0 >= ground + 4)
        print(f"[worldgen] island ({cx},{cy}) ground={ground} top_y={top_y} "
              f"y0={y0} gy={sy0} stamped={ok}", flush=True)
        if not ok:
            continue
        for _ in range(3):
            tx = int(rng.integers(-r // 2, r // 2 + 1))
            tz = int(rng.integers(-r // 2, r // 2 + 1))
            col = gi[tx + r, :, tz + r]
            ys_ = np.nonzero(col == GRASS)[0]
            if not len(ys_):
                continue
            ty = int(ys_.max())
            for k in range(1, 5):
                if ty + k < gi.shape[1]:
                    gi[tx + r, ty + k, tz + r] = LOG
            for lx in (-1, 0, 1):
                for lz in (-1, 0, 1):
                    xx, zz = tx + r + lx, tz + r + lz
                    if 0 <= xx < gi.shape[0] and 0 <= zz < gi.shape[2]:
                        if (ty + 5 < gi.shape[1]
                                and gi[xx, max(ty - 6, 0):ty + 1, zz].any()):
                            gi[xx, ty + 5, zz] = LEAVES
        # settlement slots on the island top (buildings land here later)
        if feat.get("settle"):
            cand = np.nonzero(inside_map & (top_map >= top_map.max() - 2))
            if len(cand[0]):
                order = rng.permutation(len(cand[0]))
                picked: list[tuple[int, int]] = []
                for j in order:
                    tx, tz = int(cand[0][j]), int(cand[1][j])
                    if all((tx - a) ** 2 + (tz - b) ** 2 >= 24 ** 2
                           for a, b in picked):
                        picked.append((tx, tz))
                    if len(picked) >= int(feat.get("settle_slots", 3)):
                        break
                for tx, tz in picked:
                    island_slots.append({
                        "district": feat["settle"], "label": "island settlement",
                        "x": int(cx + tx - r), "z": int(cy + tz - r),
                        "y": y0 + int(top_map[tx, tz]),
                        "clusters": [3, 4], "on_island": True})
        world[x0:x0 + sx0, y0:y0 + sy0, z0:z0 + sz0] = gi
        # waterfall from the island underside (deepest = centre column)
        if feat.get("waterfall", True):
            nz = np.nonzero(gi[r, :, r])[0]
            if len(nz):
                wy = y0 + int(nz.min()) - 1
                fx, fz = x0 + r, z0 + r
                if (0 <= fx < W and 0 <= fz < D and Ht[fx, fz] < wl
                        and wy > wl + 1):
                    world[fx, wl + 1:min(wy, H - 1), fz] = WATER
        islands_out.append({"x": int(cx), "z": int(cy), "alt": int(y0 + top_y),
                            "r": r})
        print(f"[worldgen] floating island at ({cx},{cy}) r={r} alt={alt}",
              flush=True)

    # district slots
    slots = list(island_slots)
    for dist in spec["districts"]:
        cx, cy = np.asarray(dist["center"], dtype=np.float32) * np.array([W, D])
        r = dist["radius"] * W
        step = int(max(14, 46 * (1.0 - dist["density"])))
        for x in np.arange(cx - r, cx + r, step):
            for z in np.arange(cy - r, cy + r, step):
                px = int(np.clip(x + rng.integers(-step // 3, step // 3), 3, W - 4))
                pz = int(np.clip(z + rng.integers(-step // 3, step // 3), 3, D - 4))
                if (px - cx) ** 2 + (pz - cy) ** 2 > r * r:
                    continue
                if biome[px, pz] in (BIOME["water"], BIOME["rock"]):
                    continue
                hh = Ht[max(px - 4, 0):px + 5, max(pz - 4, 0):pz + 5]
                if hh.max() - hh.min() > 4:
                    continue
                slots.append({"district": dist["type"], "label": dist["label"],
                              "x": px, "z": pz, "y": int(Ht[px, pz]),
                              "clusters": dist["clusters"]})

    # trees
    tree_spots = []
    p_tree = spec.get("decoration", {}).get("tree_density", 0.012)
    for mask, mult in ((biome == BIOME["forest"], 1.0),
                       ((biome == BIOME["plain"]) & (riparian < 0.4), 0.3)):
        cand = np.nonzero(mask & (Ht > wl + 2) & (slope <= 5))
        n = int(len(cand[0]) * p_tree * 26 * mult)
        pick = rng.choice(len(cand[0]), size=min(max(n, 0), len(cand[0])),
                          replace=False)
        for i in pick:
            tree_spots.append((int(cand[0][i]), int(cand[1][i]),
                               int(Ht[cand[0][i], cand[1][i]])))

    landmarks = []
    for lm in spec.get("landmarks", []):
        lx, ly = np.asarray(lm["at"], dtype=np.float32) * np.array([W, D])
        px, pz = int(np.clip(lx, 4, W - 5)), int(np.clip(ly, 4, D - 5))
        landmarks.append({"type": lm["type"], "x": px, "z": pz,
                          "y": int(Ht[px, pz])})

    np.savez_compressed(os.path.join(out_dir, "world.npz"), world=world)
    np.savez_compressed(os.path.join(out_dir, "terrain.npz"),
                        height=Ht.astype(np.int16), biome=biome)
    json.dump({"spec": spec, "slots": slots, "roads": roads_out,
               "trees": tree_spots, "farms": farms, "landmarks": landmarks},
              open(os.path.join(out_dir, "layout.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"[worldgen] {spec['name']}: {W}x{H}x{D} slots={len(slots)} "
          f"trees={len(tree_spots)} farms={len(farms)} "
          f"landmarks={len(landmarks)}", flush=True)


if __name__ == "__main__":
    main()
