"""Head-to-head: metadata labels vs vision labels for library items.

Channels:
  A) metadata rules over title+subtitle+description+tags (no vision)
  B) vision on blind contact sheets (renders only, no titles)
Conflicts are later adjudicated individually with multi-view renders.

Emits: out_lib/compare/sample.json, sheet_A_*.png (renders for channel B),
       and prints metadata labels for the sample.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render  # noqa: E402
from render_samples import sheet  # noqa: E402

TYPE_KW = {
    "vehicle": ["ship", "boat", "airship", "submarine", "train", "carriage",
                "pirate", "vessel", "frigate", "spaceship", "tank"],
    "vegetation": ["tree", "bush", "cherry", "redwood", "forest", "leaves"],
    "statue": ["statue", "sculpture", "golem", "dragon", "giant", "monster",
               "herobrine", "bust", "creature"],
    "sprite": ["pixel art", "map art", "sprite", "pixelart", "poster"],
    "building": ["house", "home", "castle", "tower", "church", "temple",
                 "village", "mansion", "skyscraper", "apartment", "farm",
                 "barn", "hut", "cabin", "inn", "tavern", "fort", "palace",
                 "bridge", "city", "town", "hall", "shop", "market", "spawn",
                 "dwelling", "cottage", "manor", "keep", "citadel",
                 "lighthouse", "windmill", "mill", "stable", "workshop",
                 "factory", "warehouse", "station", "library", "arena",
                 "colosseum", "monument", "gate", "wall", "harbor", "port"],
}
STYLE_KW = {
    "modern": ["modern", "contemporary", "minimalist", "downtown", "urban",
               "brutalist", "futuristic"],
    "medieval": ["medieval", "fantasy", "middle age", "rustic", "old",
                 "gothic", "victorian", "steampunk"],
    "asian": ["japanese", "chinese", "asian", "pagoda", "zen", "korean",
              "oriental"],
    "scifi": ["sci-fi", "scifi", "space", "cyber", "alien", "spaceship"],
    "western": ["western", "colonial", "ranch", "desert", "wild west"],
    "nautical": ["pirate", "navy", "sailing", "nautical", "harbor", "dock"],
}


def meta_labels(rec, full):  # noqa: ANN001
    text = " ".join(str(full.get(k, "")) for k in
                    ("title", "subtitle", "description", "tags")).lower()
    t = "sprite" if min(rec["dims"]) <= 2 else "other"
    if t != "sprite":
        for cand, kws in TYPE_KW.items():
            if any(re.search(r"\b" + re.escape(k), text) for k in kws):
                t = cand
                break
    s = "other"
    for cand, kws in STYLE_KW.items():
        if any(re.search(r"\b" + re.escape(k), text) for k in kws):
            s = cand
            break
    return t, s


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    os.makedirs(os.path.join(lib, "compare"), exist_ok=True)
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    by_url = {e["url"]: e for e in
              json.load(open("data/schematicsWithFinalUrl.json",
                             encoding="utf-8")) if isinstance(e, dict)}

    types = json.load(open(os.path.join(lib, "types.json")))
    strong = [r["idx"] for r in meta
              if types[str(r["idx"])] != "structure"]
    weak = [r["idx"] for r in meta if types[str(r["idx"])] == "structure"]
    rng = np.random.default_rng(11)
    sample = (list(rng.choice(strong, 24, replace=False)) +
              list(rng.choice(weak, 24, replace=False)))
    rng.shuffle(sample)

    recs = []
    for k, i in enumerate(sample):
        r = meta[int(i)]
        full = by_url.get(r.get("url", ""), {})
        t, s = meta_labels(r, full)
        recs.append({"idx": int(i), "tile": k + 1, "dims": r["dims"],
                     "title": full.get("title", r.get("desc", ""))[:120],
                     "meta_type": t, "meta_style": s,
                     "url": r.get("url", "")})
    json.dump(recs, open(os.path.join(lib, "compare", "sample.json"), "w"),
              ensure_ascii=False, indent=2)

    # blind sheets for channel B: tiles numbered, no titles
    tiles = []
    for r in recs:
        g = np.load(os.path.join(lib, meta[r["idx"]]["file"]))["grid"]
        tiles.append(render(g, max_px=300))
    for k in range(0, 48, 16):
        sheet(tiles[k:k + 16],
              [f"s{r['tile']}" for r in recs[k:k + 16]]).save(
            os.path.join(lib, "compare", f"sheet_{k // 16}.png"))
    for r in recs:
        print(f"s{r['tile']}: meta_type={r['meta_type']} "
              f"meta_style={r['meta_style']} | {r['title'][:70]}", flush=True)
    print("-> sheets in", os.path.join(lib, "compare"), flush=True)


if __name__ == "__main__":
    main()
