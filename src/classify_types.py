"""Hybrid type classifier: geometry -> metadata (stem matching, multilingual)
-> vision (only for the residual bucket).

Types: building / vehicle / vegetation / statue / sprite / other
Output: out_lib/types2.json + out_lib/needs_vision.json (items for the
targeted vision pass).

Lessons from the head-to-head (src/compare_stats.py): metadata misses were
almost all keyword variants (Cathedral/Bungalow/Pagode/German titles), and
vision misses were flat pixel art (solved by geometry) plus small mechanisms
(needs zoomed single-item vision).
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# stem matching (no word-end anchor) handles spelling variants
TYPE_STEMS = {
    "vehicle": ["ship", "boat", "airship", "submarine", "train", "locomotive",
                "carriage", "pirate", "vessel", "frigate", "spaceship",
                "destroyer", "freighter", "tank", "car", "truck", "wagon",
                "schiff", "boot", "zug", "flugzeug", "sail"],
    "vegetation": ["tree", "bush", "cherry", "redwood", "forest", "leaves",
                   "garden", "baum", "blatt", "flor"],
    "statue": ["statue", "sculpt", "golem", "dragon", "giant", "monster",
               "herobrine", "bust", "creature", "figure", "character"],
    "building": ["house", "home", "castle", "tower", "church", "temple",
                 "cathedr", "chapel", "village", "mansion", "skyscraper",
                 "apartment", "farm", "barn", "hut", "cabin", "bungal",
                 "inn", "tavern", "fort", "palace", "bridge", "city",
                 "town", "hall", "shop", "market", "spawn", "dwelling",
                 "cottage", "manor", "keep", "citadel", "lighthouse",
                 "windmill", "mill", "stable", "workshop", "factory",
                 "warehouse", "station", "library", "arena", "colosseum",
                 "monument", "gate", "wall", "harbor", "port", "theatr",
                 "pagod", "pyramid", "dome", "abbey", "monastery",
                 "kirch", "schloss", "haus", "burg", "turm", "molen",
                 "kasteel"],
}
SPRITE_STEMS = ["pixel art", "pixelart", "map art", "sprite", "logo"]


def classify(rec, full):  # noqa: ANN001
    dims = rec["dims"]
    text = norm(" ".join(str(full.get(k, "")) for k in
                         ("title", "subtitle", "description", "tags")))
    if min(dims) <= 2 or any(s in text for s in SPRITE_STEMS):
        return "sprite", False      # settled by geometry, no vision needed
    words = set(re.findall(r"[a-z]+", text))
    votes = {}
    for cand, stems in TYPE_STEMS.items():
        n = 0
        for st in stems:
            for w in words:
                if len(st) >= 5:
                    n += w.startswith(st)
                else:
                    n += (w == st or w == st + "s")
        votes[cand] = n
    best = max(votes, key=lambda k: votes[k])
    if votes[best] == 0:
        return "other", True          # residual -> needs vision
    # tie-break precedence: building > vehicle > statue > vegetation
    for pref in ("building", "vehicle", "statue", "vegetation"):
        if votes[pref] == votes[best]:
            return pref, False
    return best, False


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    by_url = {e["url"]: e for e in
              json.load(open("data/schematicsWithFinalUrl.json",
                             encoding="utf-8")) if isinstance(e, dict)}

    out, need = {}, []
    counts = {}
    for r in meta:
        full = by_url.get(r.get("url", ""), {})
        t, needs_vision = classify(r, full)
        out[str(r["idx"])] = t
        counts[t] = counts.get(t, 0) + 1
        if needs_vision:
            need.append(r["idx"])
    json.dump(out, open(os.path.join(lib, "types2.json"), "w"))
    json.dump(need, open(os.path.join(lib, "needs_vision.json"), "w"))
    print(f"[types2] {counts}  needs_vision={len(need)}", flush=True)


if __name__ == "__main__":
    main()
