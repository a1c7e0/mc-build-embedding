# Map Generation: prompt -> world

Goal (2026-10): generate a grand, coherent Minecraft map from a text
prompt/worldview, with natural terrain and self-consistent detail.

## Architecture

```
prompt / worldview            prompts/<name>.json (spec: terrain features,
        |                     districts, roads, landmarks, decoration)
        v
worldgen.py                   fBm + ridged relief, prompt features
        |                     (mountain/basin/plateau), meandering river,
        |                     thermal + hydraulic erosion, spike clamp,
        |                     flats quantisation, hydrology sanity,
        |                     biome dithering, road levelling, farm terraces
        v
assemble.py                   library buildings stamped per district
        |                     (retrieved by PCA cluster id from the latent
        |                     study), stone skirts, MST streets, dock pier,
        |                     procedural trees/landmarks
        v
export_schem.py               Sponge .schem v2 (WorldEdit/FAWE/Litematica)
render_world.py / _closeup / _full    aerial + iso QA renders
```

Everything is prompt-driven: terrain features, district types and their
building clusters, road routing (including `follow_river`), landmark types,
tree density, farm placement.

## QA loop (the important part)

Every iteration is: run -> aerial + iso close-ups -> vision critique -> fix
the mechanism, not the symptom. What the loop caught and how it was fixed:

| symptom (vision) | actual cause (probe) | fix |
|---|---|---|
| floating pillars / spikes | erosion deposit noise + np.roll edge wrap | small-component spike clamp, border repair |
| white aerial with stray lines | road points scaled twice (world coords) | single coordinate convention |
| disconnected ponds everywhere | erosion pits below water level | standing water only in river corridor / sea |
| inland sand blobs | lifted pits fell into the beach band | beach gated by distance-to-water |
| straight, self-crossing river | meander amplitude formula off by n/6 | amplitude = wobble * span |
| 1-block "pillar" fields on flats | ±1 deposit noise on flat ground | flats quantised to local median |
| floating/sunk buildings | no foundation on uneven ground | stone skirt ring under footprints |
| trees on landmark roofs | trees placed before landmarks | landmarks first, occupancy marks |
| straight roads cutting hills | hand-placed polylines | follow_river offset routing + MST streets |
| dead "docks" district | districts placed 100+ blocks from water | districts re-plotted along the river |

Data probes beat guessing: `probe*.py` check floating blocks, column drift,
isolated pillars, heightmap spikes directly against the voxel data, so vision
reports can be attributed to data vs renderer vs misreading.

## Current output (valley_port v3.1)

512x96x512, ~25M blocks: river valley from NW mountains to SE sea, five
style-anchored districts (old-quarter ruins, brick market, heavy industry &
 docks, outer housing, temple gardens), each drawing from ONE author
portfolio of the rom1504/Planet-Minecraft library (kNN same-author lift
8.1x = style families are real) + per-district material palette
normalisation. Vision-audited style unity: market 8-9/10, industry 8/10,
temple 8/10 (vs "fragmented, no unified style" before the rework).
`out/world/valley_port/valley_port.schem` loads in Minecraft (validated
structurally).

Library v2 (out_lib/): 3262 complete standalone buildings from
minecraft-schematics.com (was 1770 map-pack fragments), hybrid type
classifier (geometry -> metadata stem-votes -> vision residuals:
building 1780 / vehicle 262 / statue 126 / vegetation 46 / sprite 260 /
needs-vision 788). Label-channel head-to-head (src/compare_stats.py):
metadata 82% vs blind vision 78% on type; both weak on style; hybrid chain
wins.

## Known gaps / next

- fallback pool injects style-intruder "grey boxes" when an author pool runs
  dry -> grow pools (more shards) instead of falling back
- bare terraces/cliffs need detail layer (crates, vines, path edges)
- 788 residual items await the zoomed vision pass (13 sheets)
- three-way generation comparison queued: retrieval stamp / motif+assembly /
  hierarchical VQ-VAE
- renderer: flat lighting, opaque water (colour by depth), aliasing
- bigger palette fidelity: legacy ids lose block states (stairs orientation)
- second prompt theme to prove prompt-generality (e.g. desert canyon city)
