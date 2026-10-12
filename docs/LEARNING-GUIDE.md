# Learning Guide — What Knowledge This Project Actually Uses

A map of every discipline behind "prompt → Minecraft world", written for
someone who wants to learn the fields, not just run the pipeline. Each
section: the idea in plain words → where it shows up in our repo → what to
read next.

---

## 1. Representations: turning blocks into math

**The idea.** A building is a 3D grid (voxel grid) of block ids. To do
machine learning on it we convert it into numbers: first a fixed-size grid
(we crop to 32³ and map hundreds of block ids to 6 material classes), then
vectors ("embeddings") that capture similarity.

**Key concepts.** voxel, bounding box, class quantisation, embedding,
dimensionality reduction (PCA → UMAP), cosine similarity.

**Where in the repo.** `src/prep_vq_data.py` (grid → 32³ classes),
`src/features.py` (hand-crafted features + occupancy), `src/embed.py`,
`docs/RESULTS.md` (the finding: clusters are semantically meaningful).

**Read.** Any linear-algebra-for-ML intro for PCA; McInnes et al., *UMAP*
(https://umap-learn.readthedocs.io) for the pretty cluster maps.

## 2. Generative models: how machines invent things

**The idea, in three acts.**

1. *Autoencoder (AE)* — squeeze a building into a small vector and rebuild
   it. The vector space is continuous: you can walk between two buildings.
   We proved walks give gradual hybrids (docs/RESULTS.md).
2. *VQ-VAE* — same squeeze, but the middle is a **codebook**: a finite
   dictionary of "parts" (1024 of them). A building becomes a sequence of
   code indices — like words. This is *tokenisation* for 3D. Ours lives in
   `src/train_vq.py`, trained to 98% voxel accuracy.
3. *Autoregressive prior (AR)* — the missing piece. A VQ-VAE alone can
   only rebuild what it saw; random codes decode to garbage (we measured:
   2/10). A small GPT learns the *grammar of codes* — which part follows
   which — and then sampling produces plausible new buildings. This is
   exactly how text models work, one word at a time. Ours: `src/train_prior.py`.

**Key concepts.** latent space, reconstruction loss, commitment loss,
codebook / vector quantisation, token sequence, causal transformer,
perplexity, sampling temperature.

**Read.** van den Oord et al., *Neural Discrete Representation Learning*
(VQ-VAE, 2017); Karpathy's *Let's build GPT* (video + code, the clearest
transformer intro); our `docs/RELATED-WORK.md` for the Minecraft-specific
papers (World-GAN, T2BM, the 2026 Minecraft VQ "regime map").

## 3. Procedural generation: old-school algorithms that never sleep

**The idea.** Not everything needs learning. Our terrain is classic
algorithmic work: fractal noise (fbm) for hills, ridged noise for crests,
droplet erosion for valleys, thermal erosion for talus, Voronoi "crack
maps" for the floating-island undersides (borrowed from Ajaya's generator).
The renderer is a painter's-algorithm isometric rasteriser — no GPU, no
game engine.

**Key concepts.** Perlin/simplex noise, fractal Brownian motion,
erosion simulation, Voronoi tessellation, heightmap, painter's algorithm.

**Where.** `src/worldgen.py`, `src/render.py`, `docs/MAPGEN.md` (backlog).

**Read.** Sebastian Lague's *Procedural Landmass Generation* (video
series); the Voxel Plugin docs for floating-island math; any fbm primer.

## 4. Architecture theory — the layer we are about to add

This is the interesting new frontier: buildings that *make sense*. An
architect does not draw blocks; they reason in a hierarchy:

- **Program** — what spaces are needed (hall, rooms, kitchen, courtyard).
- **Adjacency / circulation** — which spaces connect to which; how people
  move. Encoded as a *bubble diagram* or *adjacency graph*.
- **Plan** — geometry that satisfies the graph: rooms are closed where
  privacy is needed, open where gathering happens; walls are continuous;
  doors align with flows.
- **Skeleton + infill** — a structural frame (grid of columns/walls/roof
  spans) first, then façades and details. *Your "basic skeleton" intuition
  is exactly this.*
- **Style grammar** — rules that repeat: window spacing, roof pitch, the
  Chinese eave (斗拱) as a parameterised motif.

**The founding papers.**

- Stiny & Gips (1972), *Shape Grammars* — design as rule systems over
  shapes. Stiny & Mitchell (1978), *The Palladian Grammar* — reverse-
  engineered a whole architectural style into rules.
- Müller et al. (2006), *Procedural Modeling of Buildings* (SIGGRAPH
  course) — the industrial-strength version (CGA shape grammar): mass →
  floor → room → façade → detail, one split-rule hierarchy. **Read this
  one first.**
- Merrell, Schkufza, Koltun (2010), *Computer-Generated Residential
  Building Layouts* — program (Bayesian network from real plans) → bubble
  diagram → constraint optimisation (adjacency, circulation, sunlight,
  furniture clearance) → floor plan. **This is "training architectural
  logic" before deep learning existed.**
- Space syntax (Hillier) — analyse plans as topological graphs
  (adjacency + accessibility only); predicts how buildings feel to move
  through. Modern twist: *Space Syntax-guided Post-training for Floor Plan
  Generation* (arXiv 2602.22507, 2026) — exactly our idea of using
  architectural logic as a training objective.

**How this plugs into our pipeline (the plan).**

1. *Detect* skeleton facts in the existing 3262-build library: wall
   continuity, room closure (is a "room" watertight?), door placement,
   roof spans. Cheap geometric analysis on voxel grids.
2. *Score* every building (architectural-logic metrics) and use the score
   as (a) a filter for map assembly and (b) an auxiliary training signal.
3. *Condition* the AR prior on a plan graph ("3 rooms, one courtyard,
   open south") so generation follows program, not just statistics.
4. *Grammar layer* for repeated details (eaves, colonnades): parameterised
   motif library + assembly program, instead of hoping a neural net
   remembers a 斗拱.

**Read next (in order).** Müller 2006 course notes → Merrell 2010 PDF
(vladlen.info/papers/architecture.pdf) → the space-syntax post-training
paper → Stiny & Mitchell 1978 for the romance of it.

## 5. Retrieval, ranking, and style

**The idea.** Most of our map quality today comes not from generation but
from *curation*: real buildings by real authors, chosen well. Embeddings
give similarity; MMR (maximal marginal relevance) gives diversity; author
portfolios give style families (an author's builds share a "grammar").

**Key concepts.** nearest-neighbour retrieval, MMR, style coherence,
author attribution as weak supervision.

**Where.** `src/assemble2.py` (pools, anchors, MMR), `src/author_sheets.py`.

## 6. Vision as the judge

**The idea.** Numbers lie about aesthetics, so we use vision-language
models as QA: render the world (isometric + aerial + close-ups), ask for
structured verdicts, and iterate. Two methodological lessons we paid for:
render *two opposite angles* (single views mislead — the "sheep face or
sheep butt" lesson), and when vision reports an artifact, *verify with a
voxel probe* before fixing (data beats impressions — caught real bugs and
avoided fake ones).

**Where.** `src/flash_vision.py`, `src/probe*.py`, every QA round in chat.

## 7. Systems (the boring part that decides if things run)

Docker images with CPU/memory caps; detached containers with
`--restart unless-stopped`; checkpoint-and-resume for anything long;
batch APIs vs live concurrency (and when each fails); keeping a WSL VM
alive with a `tail -f /dev/null` session. All of this is in
`src/flash_batch.py`, `run_prior.sh`, and the memory notes.

---

## Glossary (the words we use in chat)

- **latent** — the compressed vector/code sequence of a building.
- **codebook** — the dictionary of reusable parts in a VQ-VAE.
- **prior** — a model of what plausible code sequences look like.
- **tokenise** — turning a building into code indices.
- **MMR** — pick items that are relevant but not clones of what's placed.
- **skeleton (骨架)** — walls/floors/roof structure before detail.
- **program (功能布局)** — which rooms/spaces a building needs.
- **adjacency graph** — which spaces connect to which.
- **shape grammar** — rewrite rules that generate designs recursively.
- **space syntax** — analysis of plans as pure connection topology.

## Suggested path for the next months

1. Karpathy's GPT series → understand our `train_prior.py` line by line.
2. Müller 2006 + Merrell 2010 → understand what "architectural logic" is.
3. Re-read our `docs/RESULTS.md` → see what we already proved and why.
4. Train something small yourself (even on MNIST-level toys): feeling a
   loss go down teaches more than any paper.
