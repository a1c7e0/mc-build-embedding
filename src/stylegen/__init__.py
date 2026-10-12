"""Procedural style-grammar -> Skeleton-DSL generators (stylegen).

Four style grammars (chinese_temple / japanese_house / gothic_church /
nautical_dock) emit Skeleton-DSL v0 JSON (docs/SKELETON-DSL.md) which is
compiled UNCHANGED by the existing deterministic compiler
src/skeleton_compile.py. Because every building is produced by explicit
construction parameters (seed + grammar knobs), the style label is exact
by construction -- see out_lib/stylegen/style_labels.json.
"""
