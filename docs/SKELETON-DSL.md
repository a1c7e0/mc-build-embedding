# Skeleton DSL v0 — hand-authored building skeletons as ground truth

Goal (per roadmap step 3): prove that "building logic can be REPRESENTED"
before any neural work. A skeleton is a small JSON; a deterministic
compiler turns it into a Minecraft shell (walls/floors/openings/roof).
No learning, no search — a dumb compiler. If this cannot express a real
building faithfully, no model will.

## Coordinate system

Right-handed block grid: x = width, y = up, z = depth. One unit = one
block. Origin at the building's footprint corner (x0, z0), y0 = ground.

## Schema (v0)

```jsonc
{
  "id": "house_01",
  "scale": { "unit": "block", "storey_height": 4 },   // knob: floors stack
  "footprint": { "w": 11, "d": 9 },                   // outer bounds, x/z
  "rooms": [
    { "id": "living", "type": "living",
      "rect": [1, 1, 6, 7],          // x0,z0,x1,z1 (inner coords, walls excluded)
      "storeys": [0] }               // which storeys this room occupies
  ],
  "walls": [
    { "from": [0, 0], "to": [11, 0],     // outer polygon edge, x/z
      "storeys": [0],
      "openings": [
        { "kind": "door",   "at": 5, "width": 2, "height": 3, "y": 0 },
        { "kind": "window", "at": 1, "width": 2, "height": 2, "y": 1 }
        // at = offset along the wall from `from`, y = sill height
      ]
    }
  ],
  "floors": [ { "storey": 0, "rect": [0, 0, 11, 9] } ],   // slabs
  "roof": { "kind": "flat|gable|hip|shed",
            "overhang": 1, "pitch": 1,       // gable/hip knobs
            "ridge": "x|z" },
  "material": { "wall": "mineral", "floor": "mineral",
                "roof": "mineral", "frame": "wood" }   // 6-class vocabulary
}
```

## Semantics

- `walls` are 1-block-thick shells along footprint edges; `openings`
  subtract door/window holes deterministically.
- `rooms` label the air volumes; the compiler carves interiors and the
  v0.1 metric checker verifies every room's closure (enclosure).
- `floors` are slabs separating storeys; `roof` covers the top storey.
- `storey_height` × storeys = total height; everything is parametric so
  "logical scaling" (same skeleton, more storeys) is trivial.

## Compiler

`src/skeleton_compile.py` — JSON → uint16 grid [x, y, z] (legacy ids).
Deterministic, no parameters. Output feeds the existing renderer/metrics,
so a skeleton can be scored with the same enclosure/wall-face F1 suite.

## Ground truth

10 hand-authored skeletons in `out_lib/skeletons/*.json`, each traced
from a real library building (rendered + probed). These are the test set
for "representation is expressive enough" — judge by: can the compiler's
shell pass enclosure checks AND look like the original?

## Roadmap positions

1. THIS: representation expressiveness (deterministic compile).
2. Then: skeleton → voxel as a LEARNED task (neural generator).
3. Then: condition injection (style/material/function).
4. City line plugs in at the interface (buildings arrive with skeletons).
