"""Score metadata-channel vs vision-channel labels against author-text truth.

Truth labels are assigned from the author's own title/description (the
'author text' channel); ambiguous items are excluded. Vision labels come from
the blind contact-sheet run (recorded below verbatim).
"""
import json

VISION = {
    1: ("building", "modern"), 2: ("building", "modern"), 3: ("other", "other"),
    4: ("vehicle", "nautical"), 5: ("building", "medieval"),
    6: ("building", "western"), 7: ("building", "western"),
    8: ("building", "modern"), 9: ("building", "medieval"),
    10: ("building", "western"), 11: ("building", "other"),
    12: ("building", "modern"), 13: ("vehicle", "nautical"),
    14: ("building", "modern"), 15: ("other", "other"),
    16: ("building", "modern"), 17: ("vehicle", "nautical"),
    18: ("building", "medieval"), 19: ("building", "scifi"),
    20: ("building", "modern"), 21: ("vehicle", "scifi"),
    22: ("building", "modern"), 23: ("building", "asian"),
    24: ("building", "modern"), 25: ("building", "modern"),
    26: ("building", "medieval"), 27: ("building", "modern"),
    28: ("building", "other"), 29: ("building", "other"),
    30: ("other", "other"), 31: ("building", "modern"),
    32: ("building", "western"), 33: ("vehicle", "modern"),
    34: ("vegetation", "modern"), 35: ("other", "modern"),
    36: ("building", "modern"), 37: ("other", "modern"),
    38: ("building", "asian"), 39: ("statue", "other"),
    40: ("other", "modern"), 41: ("statue", "other"),
    42: ("building", "modern"), 43: ("other", "modern"),
    44: ("building", "western"), 45: ("building", "modern"),
    46: ("other", "asian"), 47: ("vehicle", "scifi"),
    48: ("vehicle", "nautical"),
}

# type truth from author titles; "?" = ambiguous -> excluded
TYPE_TRUTH = {
    1: "other", 2: "building", 3: "other", 4: "?", 5: "other",
    6: "building", 7: "building", 8: "building", 9: "building",
    10: "building", 11: "building", 12: "building", 13: "vehicle",
    14: "other", 15: "?", 16: "building", 17: "vehicle", 18: "building",
    19: "?", 20: "building", 21: "sprite", 22: "building", 23: "building",
    24: "building", 25: "building", 26: "building", 27: "building",
    28: "building", 29: "building", 30: "sprite", 31: "building",
    32: "building", 33: "vehicle", 34: "other", 35: "other", 36: "building",
    37: "sprite", 38: "building", 39: "other", 40: "other", 41: "statue",
    42: "building", 43: "sprite", 44: "building", 45: "building",
    46: "building", 47: "vehicle", 48: "vehicle",
}

# style truth only where the author text clearly implies one
STYLE_TRUTH = {
    11: "medieval", 12: "modern", 17: "medieval", 21: "other", 24: "modern",
    26: "medieval", 27: "scifi", 29: "asian", 31: "asian", 33: "modern",
    36: "modern", 37: "other", 42: "modern", 43: "other", 45: "asian",
    47: "nautical", 48: "nautical",
}

sample = json.load(open("out_lib/compare/sample.json", encoding="utf-8"))
meta_by_tile = {r["tile"]: r for r in sample}


def score(field, truth):  # noqa: ANN001
    tot = ok_meta = ok_vis = 0
    meta_covered = vis_wrong = 0
    details = []
    for tile, t in truth.items():
        if t == "?":
            continue
        m = meta_by_tile[tile][field]
        v = VISION[tile][0 if field == "meta_type" else 1]
        tot += 1
        ok_meta += (m == t)
        ok_vis += (v == t)
        meta_covered += (m != "other")
        details.append((tile, t, m, v))
    return tot, ok_meta, ok_vis, meta_covered, details


tot, okm, okv, cov, _ = score("meta_type", TYPE_TRUTH)
print(f"TYPE  n={tot}  metadata={okm}/{tot} ({okm/tot:.0%})  "
      f"vision={okv}/{tot} ({okv/tot:.0%})")

tot, okm, okv, cov, _ = score("meta_style", STYLE_TRUTH)
print(f"STYLE n={tot}  metadata={okm}/{tot} ({okm/tot:.0%})  "
      f"vision={okv}/{tot} ({okv/tot:.0%})  "
      f"(metadata non-'other' coverage: {cov}/{tot})")

# per-class error patterns
print("\nmetadata type misses:")
for tile, t, m, v in score("meta_type", TYPE_TRUTH)[4]:
    if m != t:
        print(f"  s{tile}: truth={t} meta={m} title={meta_by_tile[tile]['title'][:50]}")
print("vision type misses:")
for tile, t, m, v in score("meta_type", TYPE_TRUTH)[4]:
    if v != t:
        print(f"  s{tile}: truth={t} vision={v} title={meta_by_tile[tile]['title'][:50]}")
print("style misses (meta | vision):")
for tile, t, m, v in score("meta_style", STYLE_TRUTH)[4]:
    if m != t or v != t:
        print(f"  s{tile}: truth={t} meta={m} vis={v} "
              f"title={meta_by_tile[tile]['title'][:45]}")
