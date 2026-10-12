# stylegen — procedural style-grammar → Skeleton-DSL generators

Four style grammars (`chinese_temple`, `japanese_house`, `gothic_church`,
`nautical_dock`) generate Skeleton-DSL v0 JSON (`docs/SKELETON-DSL.md`)
from a random seed. Every building is compiled **unchanged** by the
existing deterministic compiler `src/skeleton_compile.py` into a voxel
shell (`.npz`) and a two-view cutaway render (`.png`), one contact sheet
per style plus a style label table.

Because a building is a deterministic function of (module, seed,
construction knobs), its style label is exact by construction — this is
the annotated dataset for style-conditioned training
(`out_lib/stylegen/style_labels.json`).

## Run

```
docker run --rm --cpus=1 --memory=1200m -v <repo>:/work -w /work \
    mcb-embed:latest python src/stylegen/generate_all.py --seeds 10
```

Outputs under `out_lib/stylegen/`:

```
<style>/<style>_vNN.json   Skeleton-DSL source (the label-generating params)
<style>/<style>_vNN.npz    compiled voxel shell (grid uint16 [x,y,z])
<style>/<style>_vNN.png    render2 cutaway + full view (compiler render)
<style>_sheet.png          contact sheet, 10 tiles, labels with seed/params
style_labels.json          style -> features/material + variant -> params
```

## Grammar → DSL construction map

| style | construction | DSL mechanism |
|---|---|---|
| chinese_temple | 台基 platform 1-3 blocks + front stair | fill rows (solid bands) |
| | 红柱 colonnade, symmetric bays | 1-cell wall segments (`column`) at platform edge |
| | 灰瓦坡顶 + 大出檐 | mineral gable/hip roof over the full grid, walls inset by the eave margin |
| | 叠檐 tier lines | eave-lip band `(s*sh-2, s*sh)` in the eave margin at every storey top |
| | 对称开间 + 宽大门 | mirrored window bays, 3-5 wide centred gate |
| japanese_house | 低矮坡顶 | gable, pitch 2-3 (low rise), walls inset under the roof |
| | 缘侧 + 柱列 | veranda margin + `column` pier row at the platform edge |
| | 障子窗 | 1x2 window arrays at 2-3 block spacing on every facade/storey |
| | 木构架 / 小入口 | all-wood palette; 1-2 wide, off-centre door |
| gothic_church | 高耸窄体量 | 25-31 x 13-15, sh 11-12 single volume (or 2 x sh 6 clerestory) |
| | 尖顶 | gable pitch 1 (pitch is inverse: small = steep) |
| | 高窗 | 1-2 wide x 5-8 high lancet arrays between bays |
| | 扶壁 | 1-cell full-height piers bridging a 2-deep slot outside the long walls |
| | 十字形平面 | 9-room cross tiling + 2 transept walls and 2 nave-flank arcade walls with tall arches |
| nautical_dock | 栈桥 + 木平台 | raised deck (fill rows at y=deck_h) on piling posts, doors at deck level |
| | 仓房 | warehouse box, 3-5 wide loading doors, small high windows |
| | shed 顶 | `shed_strip` plan: roof.kind="shed" over the covered transit strip, open loading deck beyond (plan `end_warehouse`: low gable over the whole pier) |
| | 大坡道 | stepped gangway strips descending deck → ground on the open side |
| | railings | deck-height railing rows closing the open deck |

Variants vary in footprint, storey count, roof kind/pitch, eave/deck
height, door/window sizes and positions — driven by `random.Random(seed)`
inside each module (one seed base per style: 1100/2200/3300/4400).

## DSL v0 expansions used (compiler-legal)

The compiler writes only wall shells, floor slabs and the roof, so solid
volumes (platforms, decks, steps, eave lips, railings, pilings) are
emitted as **fill rows**: 1-cell-wide wall segments whose openings carve
the *complement* of a y-band list (`dsl.fill_row`). Rules that must hold
or the compiler silently clobbers geometry:

1. Wall entries rasterise in list order, each entry rewriting its cells.
   Emit fills first, then main walls (their door/window carves must win),
   then freestanding piers.
2. One entry must carry **all** y-bands of its cells (a later entry over
   the same cells writes full height first, destroying earlier bands).
   When a post/railing overlaps a deck fill, its bands must be a superset
   (post = `(0, deck_h+1)`, so the deck block survives).
3. Rooms are carved *before* walls, so fills survive interior rooms — the
   platform under a room rect stays solid.
4. `openings.y` is relative to the entry's first storey; an entry spanning
   storeys `[0..n]` can carve through all of them (used for full-height
   portals/arches).
5. Wall segments are inclusive of both endpoints; `openings.at` is a cell
   offset and `[at, at+width)` must fit `steps+1` cells (checked by
   `dsl.validate`, run before every compile).

## Compiler quirks these generators work around

* `roof.kind="shed"` rasterises only the LOW HALF of the cross span
  (`span = cross//2+1` in `skeleton_compile.py`), so it is used exactly as
  the covered strip of a transit shed (`nautical_dock.shed_strip`) where
  half coverage is the design; japanese houses use low-pitch gable instead.
* The gable-end infill always sits at x=0/W-1 (or z=0/D-1), one cell at
  the grid edge: walls inset for eaves keep the infill as the eave gable
  board; `gothic_church`/`nautical_dock` place end walls exactly on the
  edge so the infill never floats.
* Materials: the compiler reads only `material.wall/floor/roof`
  (`frame` is ignored) and one value per slot per building — so "red
  columns" are wood-brown, and fill-row volumes are wall-material (a
  stone podium cannot be grey when the walls are timber).
* `n_st` (storey count) is derived from room storeys only — every module
  declares rooms on all used storeys.
* Floor slabs at storey s land at `s*sh-1` and are written before walls:
  a fill covering that cell must include the slab band or the slab is
  carved away (why gothic's plinth bands carry the inter-storey slab).

## Verification

`generate_all.py` validates every skeleton before compiling (bounds,
opening extents, y-extents, material vocabulary) and invokes the
compiler CLI per variant; a failure aborts the run. All 40 variants
compile clean. Connected-component probes (scipy 6- and 26-connectivity)
over the compiled grids show one component per building — no floating
geometry, despite vision reports of "floating blocks" at the 1-voxel
eave/stair steps (a known vision misread; probe over render).
