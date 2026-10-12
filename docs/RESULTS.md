# Minecraft Build Embedding — Study Notes

Experiment log for the question from the design discussion:

> "Does Minecraft buildings have a meaningful latent space? What happens at
> the boundaries between clusters? Can vision classify building types?"

## Setup

- Dataset: `assskelad/minecraftbuildings` (HuggingFace) — voxelised player
  builds as `(x, y, z, legacy_block_id)` tuples + a free-text description.
  2,955 rows parsed, **1,770 buildings kept** (min 300 blocks, bbox <= 128,
  volume cap 3M voxels). Buildings are yaw-canonicalised (long axis -> x).
- Representation: 2,203-dim per-building vector =
  43 handcrafted interpretable features (material classes, geometry, column
  height stats, mirror symmetry, colour histogram) +
  12^3 coarse occupancy (1,728) + 3 silhouettes (432).
- Pipeline: standardise -> PCA(32, 58.6% var) -> UMAP(2D) / HDBSCAN or
  KMeans(k=8). All runs in docker, 1 CPU / 1.2 GB.
- Renderer: pure numpy+PIL isometric painter's algorithm (no Minecraft, no
  GL). QA'd against raw voxel data — "backing board" artefacts reported by
  vision were real terrain slabs (grass islands), not rendering bugs.

## Finding 1 — the latent space is semantically structured (YES)

KMeans k=8 in PCA(32) space. Vision review of 12 exemplars per cluster
(contact sheets):

| cluster | size | vision verdict | theme |
|---|---|---|---|
| c0 | 225 | coherent | nature-reclaimed industrial machinery |
| c1 | 326 | mostly | rustic wall-attached platforms/scaffolds, vines |
| c2 | 238 | coherent | industrial factory / plant structures |
| c3 | 109 | mixed fn | urban commercial (glass, storefronts) |
| c4 | 231 | coherent | wood+grey utility workshops, green accents |
| c5 | 124 | coherent | vine-overgrown ruined wall-mounted tech |
| c6 | 76 | mixed | dense flat solid crates/cubes |
| c7 | 441 | coherent | large open-frame industrial buildings |

6/8 clusters visually coherent, and the themes are fine-grained (the vision
model separated "overgrown machinery" from "overgrown ruined wall sections").
Weak text labels mined from descriptions did NOT match clusters
(silhouette -0.25) — the descriptions simply never use words like
"overgrown machinery", while the visual clustering does. Vision, not text,
is the right judge here.

## Finding 2 — interpretable axes map to real visual concepts

Top z-scored features per cluster (out/axes.json) line up with the visual
themes:

- c6: fill +1.88, footprint +1.39 -> solid dense cubes ("crate buildings")
- c3: footprint +0.98, glass +0.66 -> flat, glassy urban blocks
- c5: fill +1.3, surface -1.83 -> low surface area = chunky wall masses
- c1: sym_z +0.81, top_var -0.99 -> flat-topped symmetric platforms
- c0: top_var +0.9, col_std +0.79 -> jagged uneven silhouettes (machinery)

So "z_i = roof style / material / complexity" is achievable at least through
explicit feature axes; a learned encoder should rediscover the same factors.

## Finding 3 — cluster boundaries contain hybrids (YES)

The 60 builds closest to two centroids at once ("boundary set", out/boundary
sheet): vision reports explicit material and style mixing — wood+stone
panels, grey mechanical + organic wood, white mechanical blocks. These are
not merely low-detail builds; they bridge two themes.

## Finding 4 — straight latent interpolation FAILS, geodesic walk WORKS

- Centroid line interpolation (t = 0..1, nearest real build per point):
  **step function**. The straight line crosses empty space; the same nearest
  neighbour repeats for t = 0.25..0.5. Matches the known caveat that linear
  interpolation in embedding space need not be meaningful.
- kNN-graph geodesic (Dijkstra through real builds, k = 4):
  - c0 -> c5: machinery gradually swallowed by overgrowth, hybrids in between
  - c2 -> c7 -> c3: raw industrial -> concrete blocky -> polished glass urban
  The manifold carries the transition; the straight line does not.

Practical consequence for a generator: interpolate along the manifold
(retrieve chains), not along the chord; or train a decoder/VAE so the latent
space is filled (then chords are valid).

## Vision as classifier / inspector (capability check)

Contact sheets (16 numbered renders per call) let one vision call classify
16 builds; labels were sensible across the board (house/interior/machine/
landscape/bridge/tower...). Vision also acted as cluster auditor (coherence
verdicts above) and artefact hunter. Dataset descriptions are noisy weak
labels (often describe a whole screenshot rather than the build) — do not
treat them as ground truth.

## Finding 5 — learned 256-D AE latent vs handcrafted+PCA

Small MLP autoencoder over 24^3 occupancy (13824 -> 1024 -> 512 -> 256),
early-stopped at ep25 (best val BCE 0.401; the tiny model overfits fast on
1.6k samples). Results (out/ae/):

- **Cluster coherence: AE wins.** Vision audit of all 8 KMeans clusters:
  AE 8/8 coherent vs PCA-space 6/8. AE themes are fine-grained asset kinds:
  techno-organic circuitry, abstract greebled facades, modular facade
  frames, structural shells/cross-sections, terrain-embedded machinery,
  wall-mounted machinery dioramas. Caveat: several AE clusters share the
  "industrial machinery" umbrella — the split is on asset *format*, and
  source-project artifacts (many builds from the same map pack) likely
  contribute to coherence in BOTH spaces.
- **Axis probing: interpretable but redundant.** Strongest latent<->feature
  correlations: z249 <-> sym_z +0.76, z211 <-> sym_z -0.70 (a learned
  symmetry axis with two opposing directions), and ~7 dims all loaded on
  fill/col_mean ~+-0.7 (one density factor spread over many dims). So
  "z_i = one concept" does NOT emerge on its own — a plain AE distributes
  single factors across directions; disentanglement needs explicit pressure
  (VAE with factor prior, contrastive pairs, etc.).
- Learned latent beats handcrafted features for visual coherence;
  handcrafted features beat it for interpretability per dimension. Use both:
  handcrafted axes for control, learned latent for retrieval/clustering.

## Finding 6 — generative interpolation (decode the middle)

The AE decoder can finally answer "what does the space GENERATE between two
clusters" (out/ae/gen_interp_*.png, 9 points per line, occupancy thr 0.5):

- c0 -> c6: all 9 decoded tiles are coherent; compact form -> maze-like
  patterned structures -> compact form. Middle tiles are plausible hybrids.
  **In dense regions the latent line is generative and smooth.**
- c2 -> c3: t=0 coherent, middle decays to scattered voxels, t=1 nearly
  empty. **Across sparse regions the same operation dissolves into mush.**
  Continuity is local, not global — identical to the geodesic finding, now
  on the generative side.
- Random Gaussian samples (out/ae/gen_random.png): sharp-edged but ALL the
  same "noisy wall slab" — near-zero diversity. Sampling N(mu, sigma) over
  an unregularised AE latent does not give a generative model; the mass sits
  on one mean structure. A VAE/adversarial/diffusion prior is required for
  real sampling.

Bottom line for the architecture discussion: retrieval along the manifold
works today; generation between concepts works only locally and needs a
proper generative prior to go beyond.

## Limits and next steps

- Handcrafted+PCA space, not a learned encoder. Next: small 3D AE/VAE
  (32^3, 256-D latent) as in "Open-Ended Evolution for Minecraft Building
  Generation"; then latent interpolation becomes generative.
- Data is only 1,770 small builds. rom1504's minecraft-schematics-dataset
  (tfrecords, mined from minecraft-schematics.com) is the obvious scale-up.
- HDBSCAN found 0 clusters (all noise) at min_cluster_size=29 — the space is
  diffuse at that granularity; KMeans k=8 was used instead. Cluster count is
  a free parameter, not a finding.
- Renders are colour-approximate (legacy ids, no block states/data values).
