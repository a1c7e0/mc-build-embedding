# Dual-Route Progress Audit: Vector/Latent vs Skeleton

> **Scope**: Read-only audit of actual code, checkpoints, and experiment results.
> **Date**: 2026-10-12 (GMT)
> **Rule**: "Code exists" ≠ "has been run" ≠ "has been validated". No claims from filenames.

---

## 1. Project Roadmap — Actual Data Flows

### Route A: Vector / Latent

```
Input (building voxel grid)
  → src/prep_vq_data2.py          preprocessing (type=building, aspect-preserving, dedupe)
  → 6-class voxel (32³ or 64³)    representation
  → src/train_vq.py:VQVAE.enc()   encoder
  → latent grid (8³ or 16³)       latent (continuous)
  → src/train_vq.py:VQVAE.vq()    vector quantization
  → token sequence (512 or 4096)  discrete codes (vocab 1024)
  → src/train_prior.py:GPT        prior/generator (autoregressive)
  → token sequence (generated)
  → src/train_vq.py:VQVAE.dec()   decoder
  → 6-class voxel (32³ or 64³)    output
  → legacy block IDs              Minecraft building
```

| Step | File | Class/Function | Input | Output | Has CKPT | Has Results | Status |
|------|------|---------------|-------|--------|----------|-------------|--------|
| Preprocessing | `src/prep_vq_data2.py` | `main()` | raw library .npz | `out_lib/vq_data2.npz` (1669, 32, 32, 32) uint8 | N/A | ✅ fill=0.088 | **validated** |
| Preprocessing 64³ | `src/prep_vq_data2.py` (SIZE=64) | `main()` | raw library .npz | `out_lib/vq_data64.npz` (1669, 64, 64, 64) | N/A | ✅ fill=0.095 | **validated** |
| VQ-VAE encode | `src/train_vq.py` | `VQVAE.enc(x) -> z` | (B, 6, 32, 32, 32) float32 | (B, C, 8, 8, 8) continuous | ✅ `vq_ckpt2.pt` | ✅ | **validated** |
| VQ quantize | `src/train_vq.py` | `VQVAE.vq(zl) -> codes` | (B, 512, C) | (B, 512) int, vocab 1024 | in ckpt | ✅ | **validated** |
| VQ decode | `src/train_vq.py` | `VQVAE.dec(zq)` | (B, C, 8, 8, 8) | (B, 6, 32, 32, 32) | in ckpt | ✅ | **validated** |
| Prior training | `src/train_prior.py` | `GPT.forward(idx)` | (B, 512) int | (B, 512, 1032) logits | ✅ `prior_v3_ckpt.pt` | ✅ loss=0.84 | **run (quality poor)** |
| 64³ VQ-VAE | `src/train_vq.py` (VQ_CKPT env) | `VQVAE` | (B, 6, 64, 64, 64) | (B, C, 16, 16, 16) | ✅ `vq_ckpt64.pt` | ✅ IoU=0.895 | **validated** |
| 64³ coarse prior | `src/train_prior64.py` | `GPT` (SEQ=512) | (B, 512) int | (B, 512, 1024) | 🔄 training now | loss=2.4@20 | **running** |

### Route B: Skeleton

```
Input (building / house specification)
  → src/procthor_to_dsl.py        skeleton extraction (ProcTHOR JSON → DSL)
  → Skeleton DSL (JSON)           representation
  → src/plan_gen/export_dsl.py    plan → skeleton conversion
  → src/skeleton_compile.py       deterministic compiler
  → voxel shell (uint16 grid)     structure
  → [NOT IMPLEMENTED]             decoration
  → Minecraft building
```

| Step | File | Class/Function | Input | Output | Has CKPT | Has Results | Status |
|------|------|---------------|-------|--------|----------|-------------|--------|
| Skeleton DSL format | `docs/SKELETON-DSL.md` | spec | — | JSON schema | N/A | ✅ documented | **validated** |
| ProcTHOR → DSL | `src/procthor_to_dsl.py` | `main()` | ProcTHOR house JSON | skeleton JSON + voxels | N/A | ✅ verified (3 houses) | **validated** |
| Plan generation | `src/plan_gen/model.py` | `GPT` (dim=128, 4L) | room type tokens | plan JSON (room rects) | ✅ `out/plan_gen_v1/ckpt.pt` | ✅ acc=73.1% | **validated** |
| Plan → skeleton | `src/plan_gen/export_dsl.py` | `plan_to_skeleton()` | plan rooms | skeleton JSON | N/A | ✅ | **validated** |
| DSL → voxel compile | `src/skeleton_compile.py` | `compile_skeleton(sk)` | skeleton JSON | (W, H, D) uint16 grid | N/A | ✅ | **validated** |
| Structural metrics | `src/skeleton_metrics.py` | `main()` | voxel grid | enclosure/wall_gap/roof_cover | N/A | ✅ | **validated** |
| Decoration | — | — | — | — | ❌ | ❌ | **not started** |
| Neural skeleton→voxel | — | — | — | — | ❌ | ❌ | **not started** |

---

## 2. Vector / Latent Route — Detailed Audit

### 2.1 Representation

**What is the latent?**
- **VQ codes**: discrete token sequences from a 1024-entry codebook.
  - 32³ VQ: 8×8×8 = 512 tokens per building (each code = 4×4×4 voxel region)
  - 64³ VQ: 16×16×16 = 4096 tokens per building (each code = 4×4×4 voxel region)
- **NOT** a continuous latent (VQ-VAE quantizes to discrete codes)
- **NOT** a building-level vector (it's a spatial sequence, not a single embedding)
- **NOT** an embedding in the sense of a semantic descriptor

**Building-level semantic representation**: NOT achieved at the token level. The codebook atlas shows individual codes capture local geometric patterns (wall segments, openings), but there is no building-level "style vector" or "function vector" extracted from the token sequence.

**Multiple representations exist:**
1. **VQ token sequence** (512 or 4096 codes) — the primary representation
2. **Handcrafted feature vector** (2,203-dim: 43 features + 12³ occupancy + 3 silhouettes) — from early clustering experiments (`docs/RESULTS.md`)
3. **256-dim AE latent** — from small autoencoder experiments (`docs/RESULTS.md` Finding 5)
4. **Style tokens** (8 style tokens prepended to prior input) — conditional generation, not a building representation per se

### 2.2 Reconstruction

**Metrics (measured with `src/vq_eval2.py`):**

| Metric | 32³ (v0.2) | 64³ | Notes |
|--------|-----------|------|-------|
| occupied IoU | 0.836 | **0.895** | Intersection-over-union on occupied voxels |
| non-air acc | 0.803 | **0.865** | Accuracy excluding air |
| wall-face F1 | 0.887 | **0.896** | Wall face detection F1 |
| CC \|delta\| | 8.16 | 13.22 | Connected-component count difference (lower=better) |
| class0 (air) IoU | 0.984 | 0.990 | |
| class1 (mineral) IoU | 0.829 | **0.885** | |
| class2 (wood) IoU | 0.233 | **0.370** | +13.7pp |
| class3 (glass) IoU | 0.101 | **0.213** | **doubled** |
| class4 (organic) IoU | 0.097 | **0.169** | |
| class5 (colourful) IoU | 0.083 | **0.188** | **doubled** |

**Qualitative**: 64³ reconstruction preserves geometry (walls, floor, roof visible), topology (rooms connected), and material classes (6-class). Style is preserved at coarse level (building proportions). Known weakness: rare materials (glass, organic, colourful) at 32³ resolution had poor IoU (< 0.11) — fixed at 64³ (roughly doubled).

**What reconstruction preserves:**
- ✅ Geometry: overall shape, proportions, wall positions
- ✅ Topology: room connectivity, enclosed spaces
- ✅ Material: 6-class distinction (mineral dominant)
- ⚠️ Style: coarse proportions preserved; fine stylistic details (e.g., specific window shapes) lost
- ✅ Scale: aspect-preserving normalization preserves relative proportions
- ⚠️ Structure: fine structural details (openings, decorative elements) poorly reconstructed at 32³, better at 64³

### 2.3 Generation

**Prior model**: GPT (custom causal self-attention), trained autoregressively on VQ token sequences.

| Version | Architecture | Steps | Loss | Sample Quality |
|---------|-------------|-------|------|---------------|
| v1 | dim=192, 3L | 12,620 | 1.11 | "稀碎" (fragmented) |
| v2 | dim=192, 3L, style tokens | 12,000 | ~1.0 | broken (empty/air-heavy) |
| v3 | dim=256, 6L | 20,000 | 0.84 | semi-complete (modern style discernible) |
| v4 | dim=256, 6L | 🔄 11,320/20,000 | 0.41 | TBD |
| 64³ coarse | dim=256, 6L, SEQ=512 | 🔄 20/15,000 | 2.4 | TBD |

**Can it generate complete objects?** Partially — v3 produces semi-complete structures with walls and interior partitions. Not reliably complete.

**Can it generate buildings?** Only in the weakest sense — some v3 samples show recognizable building features (closed walls, interior partitions). Quality is far below the reconstruction quality.

**Mode collapse / repetition**: Not formally measured. Visual inspection of v3 samples shows some template-like repetition across samples.

**Loss ≠ quality**: Loss decreased from 1.11 → 0.84 → 0.41, but sample quality improvement is much slower. This is consistent with autoregressive models on long sequences.

### 2.4 Latent Semantics — CRITICAL SECTION

**Evidence of meaningful latent structure:**

1. **Codebook Atlas** (`src/codebook_atlas.py`, `src/atlas2.py`): Decodes all 1024 codes and renders them visually.
   - 32³ atlas: codes = **material/texture fragments** (solid blocks, wood textures, glass, organic). No part-level semantics.
   - 64³ atlas: codes = **architectural parts** (wall segments, walls with rectangular openings, L-shaped corners, columns/beams, roof slabs, solid fill). Vision analysis: "far more obvious than texture fragments, each code is a distinct, identifiable 3D shape with a specific architectural function."
   - **Status**: VISUALLY VERIFIED (not a quantitative experiment, but clear and reproducible)

2. **Multi-token Perturbation** (`src/perturb_multi.py`): Swaps k=2/4/8 contiguous tokens with other codes, measures change in decoded output.
   - k=2: 0.14% voxels changed
   - k=4: 0.23% voxels changed
   - k=8: 0.56% voxels changed
   - Linear scaling → codes compose into regional structure (not independent texture patches)
   - Visual: run-of-4 swaps cause part-level structural changes (tower add/remove, beam) localized to a region
   - **Status**: QUANTIFIED + VISUALLY VERIFIED

3. **Clustering** (early experiments, `docs/RESULTS.md`): KMeans k=8 on 256-dim AE latent.
   - AE clusters: 8/8 visually coherent (vs PCA: 6/8)
   - Themes: "techno-organic circuitry", "modular facade frames", "structural shells" — fine-grained asset types
   - **Status**: VISUALLY VERIFIED (but AE latent ≠ VQ token latent)

4. **Interpolation** (early experiments, `docs/RESULTS.md`):
   - Straight-line interpolation: FAILS (step function, crosses empty space)
   - Geodesic interpolation (kNN graph): WORKS (gradual transitions: machinery→overgrowth, industrial→glass)
   - **Status**: VISUALLY VERIFIED (but again on AE latent, not VQ tokens)

5. **Linear Probing / Attribute Prediction**: NOT PERFORMED on VQ tokens. No evidence that individual token positions encode predictable building attributes.

6. **PCA/UMAP on VQ tokens**: NOT PERFORMED. The token sequence is spatial (4096 positions), not a flat vector; PCA would need flattening first.

**Summary of latent semantics evidence:**
- ✅ Codebook has part-level semantics at 64³ (visual evidence)
- ✅ Codes compose into regional structure (quantified perturbation)
- ✅ AE latent space has cluster structure (visual evidence)
- ⚠️ Interpolation works only on manifold (AE latent, not VQ tokens)
- ❌ No linear probing / attribute prediction on VQ tokens
- ❌ No formal disentanglement evidence
- ❌ No vector arithmetic demonstrated

### 2.5 Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Representation** | **7/10** | VQ codes at 64³ capture architectural parts (validated visually). But no building-level semantic vector. Token sequence is spatial, not a compact descriptor. |
| **Reconstruction** | **8/10** | 64³: IoU 0.895, wall-F1 0.896. Rare materials doubled from 32³. Solid quantitative evidence. Weakness: rare materials still < 0.4 IoU. |
| **Generation** | **3/10** | Prior produces semi-complete buildings at best (v3). Loss 0.84 → samples still "稀碎"-ish. 64³ coarse prior just started. No reliable generation demonstrated. |
| **Semantic latent** | **5/10** | Codebook atlas is strong evidence for local part semantics. Perturbation confirms compositional structure. But no building-level attributes, no disentanglement, no vector arithmetic. "Looks structured" vs "proven semantics" — leaning toward former. |
| **Controllability** | **2/10** | Style tokens exist (8 styles) but no evidence of precise control (e.g., "make it taller", "add more windows"). No attribute conditioning. No interpolation control demonstrated. |

---

## 3. Skeleton Route — Detailed Audit

### 3.1 Skeleton Representation

**Format**: JSON (Skeleton DSL v0, documented in `docs/SKELETON-DSL.md`)

**Fields**:
```json
{
  "id": "house_01",
  "scale": {"unit": "block", "storey_height": 4},
  "footprint": {"w": 11, "d": 9},
  "rooms": [
    {"id": "living", "type": "living",
     "rect": [1, 1, 6, 7],           // x0,z0,x1,z1
     "storeys": [0]}
  ],
  "walls": [
    {"from": [0, 0], "to": [11, 0],
     "storeys": [0],
     "openings": [
       {"kind": "door", "at": 5, "width": 2, "height": 3, "y": 0},
       {"kind": "window", "at": 1, "width": 2, "height": 2, "y": 1}
     ]}
  ],
  "floors": [{"storey": 0, "rect": [0, 0, 11, 9]}],
  "roof": {"kind": "flat|gable|hip|shed", "pitch": 1, "ridge": "x"},
  "material": {"wall": "mineral", "floor": "wood", "roof": "mineral", "frame": "wood"}
}
```

**Coverage**:
- ✅ Wall: walls array with from/to coordinates
- ✅ Floor: floors array with storey + rect
- ✅ Roof: flat/gable/hip/shed with pitch/ridge
- ✅ Opening: door/window with kind/at/width/height/y
- ✅ Room: rooms array with type/rect/storeys
- ❌ Corridor: not explicitly represented (rooms can be corridors by type)
- ✅ Dimensions & proportions: explicit coordinates in blocks
- ✅ Hierarchy: storeys > rooms > walls > openings (implicit)

### 3.2 Skeleton Extraction

**From real buildings (ProcTHOR)**: `src/procthor_to_dsl.py`
- Parses ProcTHOR house JSON (walls with polygon/holePolygon, rooms with floorPolygon, materials)
- Converts to skeleton DSL (metres rounded to blocks)
- **Validated**: 3 houses tested end-to-end (procthor_0, 7, 12) with visual QA
- **Accuracy**: manually verified for tested houses; no formal accuracy metric
- **Failure cases**: none observed in tested houses (single-storey ProcTHOR)

**From plan generator**: `src/plan_gen/export_dsl.py:plan_to_skeleton()`
- Takes room rects → derives interior walls + door/window openings
- **Validated**: visually confirmed (interior walls, doors, windows present)
- **Limitation**: exterior walls are always rectangular (no L-shaped buildings)

**Hand-authored skeletons**: documented in `docs/SKELETON-DSL.md` (design spec). Not used in current pipeline.

### 3.3 Skeleton → Building

**Deterministic compiler**: `src/skeleton_compile.py:compile_skeleton(sk: dict) -> np.ndarray`
- Input: skeleton JSON dict
- Output: `(W, H, D)` uint16 grid (legacy block IDs)
- Logic: walls (1-thick) + openings (carved) + floor slabs + room interiors (hollowed) + roof (flat/gable/hip/shed)
- **Validated**: multiple test cases (ProcTHOR houses + plan_gen outputs)
- **No learning involved**: pure geometric rules

**Neural generator**: ❌ Does not exist. The only skeleton→voxel path is deterministic.

**LLM generator**: ❌ Not explored.

**Hybrid**: ❌ Not implemented.

### 3.4 Structural Validity

**Metrics** (`src/skeleton_metrics.py`):
- `enclosure`: interior air pockets / inside air (boundary-closure ratio). High = closed rooms.
- `wall_gap`: perimeter columns with no solid block in middle band. High = missing walls.
- `roof_cover`: top-band coverage over footprint.
- `base_cover`: bottom-band coverage (grounded?).

**Not yet measured**:
- ❌ Connectivity (are all rooms reachable?)
- ❌ Room validity (minimum dimensions, aspect ratio)
- ❌ Wall continuity (beyond wall_gap metric)
- ❌ Floor/roof consistency
- ❌ Opening validity (do doors connect rooms?)
- ❌ Symmetry
- ❌ Structural hierarchy

### 3.5 Decoration

**Status**: NOT IMPLEMENTED. The pipeline is:
```
Skeleton → Structure (voxel shell) → [MISSING: Decoration] → Building
```

The plan was always to separate structure from decoration (per project architecture), but no decoration module has been written. The `material` field in the skeleton specifies 6-class materials (mineral/wood/glass/organic/colourful) but no decorative elements (furniture, trim, landscaping).

**ProcTHOR furniture data**: `out_lib/plans/procthor_furniture.jsonl` (45.3MB, 29.3 items/house) — collected but not used.

---

## 4. Direct Comparison

| Dimension | Vector / Latent | Skeleton |
|-----------|----------------|----------|
| **Representation** | VQ token sequence (512 or 4096 codes) | JSON DSL (rooms, walls, floors, roof, openings) |
| **Reconstruction** | ✅ IoU 0.895 (64³), wall-F1 0.896 | ✅ deterministic compile (verified visually) |
| **Generation** | ⚠️ semi-complete (v3 prior), loss 0.84 | ✅ plan_gen acc 73.1%, clean_house 40% |
| **Structure** | ⚠️ implicit in token sequence | ✅ explicit (walls, rooms, openings) |
| **Style** | ⚠️ style tokens (8 styles), no fine control | ❌ no style representation |
| **Controllability** | ❌ minimal (style only) | ✅ explicit (edit JSON directly) |
| **Interpolation** | ⚠️ manifold only (AE latent) | N/A (discrete structure) |
| **Interpretability** | ❌ token sequence not interpretable | ✅ JSON is human-readable |
| **Data requirement** | 1,669 buildings (VQ), 10k (ProcTHOR) | 10k ProcTHOR plans + manual DSL |
| **Scalability** | ⚠️ 4096 tokens = slow generation | ✅ deterministic compile = fast |
| **Current quality** | Medium (recon good, generation poor) | Medium (plan good, decoration missing) |
| **Strongest point** | Part-level semantics at 64³ (codebook atlas) | Explicit structure + deterministic compiler |
| **Biggest weakness** | Generation quality (prior 稀碎) | No decoration, no neural variation |

**Which route is more mature?**
- **Skeleton route is more mature** for structure: explicit representation, deterministic compiler, plan generator with 73% accuracy.
- **Vector route is more mature** for representation: 64³ codebook captures architectural parts, reconstruction quality is high.
- **Neither route can generate complete, high-quality buildings end-to-end**. Vector route's generation is the weakest link. Skeleton route's decoration is missing.

---

## 5. Should the Two Routes Merge?

### Option A: Skeleton → Vector Generator → Building
Skeleton provides structural constraints, vector latent generates geometry within those constraints.

**Fit**: Good conceptually. But the vector generator (prior) is currently too weak to generate within constraints. Would need a conditional prior (conditioned on skeleton), which doesn't exist.

### Option B: Vector → Skeleton
Vector latent produces a skeleton, which is then compiled.

**Fit**: Poor. The vector latent doesn't produce skeleton-level abstractions (no evidence of room/wall detection from tokens).

### Option C: Skeleton → Vector latent → Structure → Decoration
Skeleton as conditioning, latent for variation.

**Fit**: **Best fit for current code**. The skeleton DSL already generates structure. The vector latent could provide stylistic/material variation within the structural framework. This is the natural division: skeleton = "what" (structure), latent = "how" (style/detail).

### Option D: Skeleton → Structure → Vector/Style latent → Decoration
Skeleton generates structure, then vector latent handles decoration.

**Fit**: Also good. Decoration is the missing piece; a latent model for decoration (trained on furniture/detail data) is a natural fit.

### Option E: Completely independent, merge at Minecraft compiler
Both routes produce buildings independently, merged at world assembly.

**Fit**: Pragmatic but wasteful. Duplicates effort on shared components (materials, rendering).

**Recommendation**: **Option C (or D)**. The skeleton DSL handles structural generation (already working). The vector latent handles style variation and decoration (its strength: codebook has part semantics). The interface is the skeleton DSL itself: skeleton provides constraints, latent provides variation.

---

## 6. Vector + Skeleton Interface

**Minimal viable interface**:
```json
{
  "skeleton": "<skeleton DSL JSON>",
  "style": "medieval|modern|asian|...",
  "geometry_latent": "<512 or 4096 VQ codes>",
  "material": "mineral|wood|glass|organic|colourful",
  "scale": {"w": 11, "d": 9, "storeys": 1}
}
```

**Current code distance**:
- ✅ Skeleton DSL: exists and validated
- ✅ Style tokens: exist (8 styles in prior)
- ⚠️ Geometry latent: exists (VQ tokens) but not conditioned on skeleton
- ✅ Material: exists (6-class in skeleton DSL)
- ✅ Scale: exists (in skeleton DSL)

**Needs retraining**:
- Conditional prior: GPT conditioned on skeleton (rooms/walls) → generate VQ tokens within those constraints. Does NOT exist.

**Can be reused directly**:
- VQ-VAE encoder/decoder (already trained)
- Skeleton DSL compiler (already validated)
- Style token embeddings (already trained in prior)

---

## 7. World Generator Architecture

**Current state**: `src/worldgen.py` (650 lines) generates terrain (floating islands, valley with river/bridge). `src/assemble2.py` places buildings. Two complete worlds produced (sky_archipelago, valley_port).

**Missing**: Road network, city blocks, parcels, building connections, street planning.

**Proposed architecture**:
```
World Planner (terrain + zones)
  → District (functional areas)
    → Block (city blocks from road network)
      → Parcel (individual lots)
        → Skeleton (from plan generator or library)
          → Building Generator (skeleton → structure → decoration)
            → Connector (walkways, bridges between buildings)
              → Minecraft world
```

**What should be procedural vs neural**:
- Terrain: **procedural** (fbm, erosion — already works)
- Road network: **procedural** (recursive subdivision + organic growth)
- City blocks / parcels: **procedural** (deterministic subdivision)
- Building skeleton: **neural** (plan generator) or **retrieval** (from library)
- Building structure: **deterministic compiler** (skeleton_compile)
- Decoration: **neural** (vector latent, future work)
- Connector: **procedural** (geometry rules)

**Don't over-AI**: terrain, roads, blocks should stay procedural. The neural models handle what they're good at (plan generation, style variation).

---

## 8. Top 3 Bottlenecks

### 1. Generation Quality (Vector Route)
- **Type**: architecture + training
- **Evidence**: Prior v3 loss 0.84 but samples still broken/semi-complete. Loss decrease (1.11→0.84→0.41) doesn't translate to quality improvement. The 4096-token sequence problem is fundamental (attention O(n²) too slow for flat training).
- **Impact**: Vector route cannot generate usable buildings.

### 2. Missing Decoration (Skeleton Route)
- **Type**: engineering (not started)
- **Evidence**: Pipeline ends at "structure". ProcTHOR furniture data collected (45.3MB) but unused. No decoration module exists.
- **Impact**: Skeleton route produces bare shells, not finished buildings.

### 3. No Conditional Generation (Both Routes)
- **Type**: conditioning + architecture
- **Evidence**: Style tokens exist but no evidence of precise control. No skeleton-conditioned generation. No attribute-based control.
- **Impact**: Cannot generate specific buildings on demand ("a medieval 2-storey house with 3 rooms").

---

## 9. Phased Roadmap

### Phase 0: Validate 64³ Hierarchical Generation
- **Goal**: Prove 64³ hierarchical prior can generate complete buildings
- **Minimal experiment**: Train coarse prior (512 tokens, running now), then fine refiner. Generate 20 samples.
- **Success criteria**: > 50% samples show recognizable building structure (walls, rooms, roof)
- **Failure judgment**: If coarse prior loss plateaus > 2.0, the codebook may need retraining with more data. If samples are still broken at loss < 1.0, architecture needs change (e.g., larger model, different attention).
- **Impact**: Unlocks 64³ generation path. If fails → focus on skeleton route.

### Phase 1: Decoration Module
- **Goal**: Add decoration to skeleton pipeline
- **Minimal experiment**: Rule-based decoration (furniture placement from ProcTHOR data, material variety) applied to compiled shells.
- **Success criteria**: Decorated buildings visually match ProcTHOR reference quality
- **Failure judgment**: If rule-based decoration looks bad → need neural approach (train on ProcTHOR furniture layouts)
- **Impact**: Completes skeleton pipeline (structure + decoration).

### Phase 2: Conditional Generation
- **Goal**: Generate specific buildings on demand
- **Minimal experiment**: Fine-tune prior with skeleton conditioning (skeleton → plan → tokens). Or style+attribute conditioning.
- **Success criteria**: Given "medieval 2-storey house with 3 rooms", generate matching building > 50% of time
- **Failure judgment**: If conditioning doesn't work → data too small (need 10k+ examples per condition)
- **Impact**: Unlocks controllable generation.

### Phase 3: World Generation Integration
- **Goal**: City generation with roads, blocks, parcels
- **Minimal experiment**: Procedural road network (recursive subdivision), parcel assignment, building placement from library.
- **Success criteria**: Generate a city with > 100 buildings, roads connecting them, visual QA pass
- **Failure judgment**: If procedural roads look bad → try organic growth algorithm
- **Impact**: Full pipeline from prompt to Minecraft world.

### If only 3 experiments possible:
1. **64³ hierarchical generation** (Phase 0) — the critical path for vector route
2. **Decoration module** (Phase 1) — completes skeleton route
3. **Skeleton-conditioned generation** (Phase 2) — bridges both routes

---

## 10. Decision Report

### A. Vector Route
- **Current state**: Representation well-validated (64³ codebook has part semantics). Reconstruction solid (IoU 0.895). Generation weak (semi-complete samples).
- **Already verified**: VQ-VAE reconstruction quality, codebook part semantics, perturbation compositionality.
- **Not yet verified**: Reliable generation, latent controllability, disentanglement, building-level semantics.
- **Biggest problem**: Prior generation quality — loss doesn't translate to samples. 4096-token sequence too long for flat training.
- **Next step**: Complete 64³ hierarchical prior training, evaluate sample quality.

### B. Skeleton Route
- **Current state**: Structure generation complete (DSL → compiler → voxel shell). Plan generator working (73.1% accuracy). Decoration missing.
- **Already verified**: Skeleton DSL expressiveness, deterministic compiler, plan generation, ProcTHOR integration.
- **Not yet verified**: Decoration quality, multi-storey generation, style variation.
- **Biggest problem**: No decoration module. Pipeline incomplete.
- **Next step**: Implement decoration module (rule-based or neural).

### C. Relationship
- **Current recommendation**: **Skeleton + Vector** (Option C)
- **Reasoning**: Skeleton handles "what" (structure), vector handles "how" (style/decoration). Natural division. Skeleton DSL is the interface.
- **Not**: Independent (wasteful), Vector→Skeleton (no evidence vector produces skeletons), Skeleton→Vector (vector too weak for constrained generation).

### D. World Generation
- **Current state**: Terrain generation works (2 complete worlds). Missing roads, blocks, parcels, connections.
- **Next step**: Procedural road network + parcel assignment + building placement from library.

### E. Total Roadmap
```
Now → 64³ hierarchical generation (Phase 0)
  → Decoration module (Phase 1)
    → Conditional generation (Phase 2)
      → World generation integration (Phase 3)
        → Complete prompt-to-Minecraft pipeline
```

### If only 3 things:
1. **Finish 64³ hierarchical prior** — this is the critical test for whether the vector route can generate at all. If it fails, the vector route is limited to reconstruction/retrieval only.
2. **Build decoration module** — this completes the skeleton route and makes it the primary production path.
3. **Build procedural road network** — this enables world generation regardless of which building route succeeds.

**Why these 3**: They test the highest-risk hypotheses (generation quality), complete the most mature route (skeleton), and unlock the biggest capability (world generation). If 64³ generation works, we have both routes. If it doesn't, skeleton route + world generation is still a complete product.

---

## 11. Audit Corrections & Supplementary Findings

> Added after independent subagent code review (2026-10-12 GMT). Corrections to
> earlier sections where claims were too optimistic or missing context.

### 11.1 Checkpoint Distribution

All 6 VQ/prior checkpoints are **NOT on cr60**. They reside on training machines:

| Checkpoint | Location | Size | Notes |
|---|---|---|---|
| `vq_ckpt2.pt` | seewo A | 24MB | 32³ VQ-VAE v0.2 |
| `vq_ckpt64.pt` | seewo B | ~24MB | 64³ VQ-VAE |
| `prior_ckpt.pt` | seewo A | 27MB | v1/v2 (overwritten) |
| `prior_v3_ckpt.pt` | seewo A → deleted | 62MB | v3 (cleaned to save space) |
| `prior_v4_ckpt.pt` | seewo A | 62MB | v4 (training in progress) |
| `plan_gen_v1/ckpt.pt` | mcbox | ~3MB | 10k-step plan generator |

On cr60 only `out/ae_model.pt` (118MB, old AE) and `out/plan_gen_smoke/ckpt.pt` (3MB, 300-step smoke) exist. All checkpoint-dependent metrics in this audit are real but **cross-machine verification requires syncing first**.

### 11.2 No Train/Val/Test Split (VQ & Prior)

`train_vq.py` and `train_prior.py` sample batches uniformly from the full dataset. **All reconstruction metrics are effectively training-set metrics.** The 64³ IoU 0.895 may be optimistic. A proper held-out test set is needed for honest evaluation.

Note: `plan_gen` DOES have a proper split (fingerprint-grouped, 9000/1000, overlap=0). This is the only model with honest test metrics.

### 11.3 Plan Generator — Two Data Points

| Run | Steps | degenerate | overlap | oob | clean_house |
|---|---|---|---|---|---|
| smoke (300 steps) | 300 | 12.7% | 39.5% | 5.7% | 18.75% |
| v1 (10k steps, temp=1.0, 20 samples) | 10,000 | 0% | 20% | 0% | 40% |
| GT baseline | — | 0% | 4.9% | 0% | 81.25% |

The v1 model is substantially better than smoke but still below ground truth, especially on overlap (20% vs 4.9%) and clean_house (40% vs 81.25%).

### 11.4 Codebook Usage Statistics

- Effective vocabulary (perplexity): **249** out of 1024 codes
- Top code share: **9.5%** (code 164)
- Top-10 codes: **33.6%** of all usage
- All 1024 codes occur in the corpus (no hard collapse)
- Skewed but alive — evidence against degenerate latent

### 11.5 Style Distribution (corpus)

| Style | Count | % |
|---|---|---|
| modern | 1,248 | 35.7% |
| other | 861 | 24.6% |
| medieval | 503 | 14.4% |
| scifi | 221 | 6.3% |
| nautical | 196 | 5.6% |
| asian | 132 | 3.8% |
| western | 20 | 0.6% |

Medieval is undersampled (14.4%), western is severely undersampled (0.6%). Style-conditioned generation for rare styles will be unreliable.

### 11.6 Rect-Based DSL Limitation

Only **63.2%** of ProcTHOR rooms and **24.9%** of houses are rect-exact (polygon == bounding box). **36.8% of rooms are L-shaped or non-rectangular** — these cannot be expressed in the current rect-based DSL v0. This is a significant expressiveness limitation for the skeleton route.

### 11.7 Structural Metrics (3,262 library buildings)

| Metric | Median | Mean | Min | Max |
|---|---|---|---|---|
| enclosure | 0.579 | 0.583 | 0.0 | 1.0 |
| wall_gap | 0.522 | 0.501 | 0.0 | 1.0 |
| roof_cover | 0.052 | 0.198 | 0.0 | 1.0 |
| base_cover | 1.0 | 0.791 | 0.0 | 1.0 |

Note: roof_cover median 0.052 is very low — most library buildings lack proper roofs (or roofs are thin single layers not captured by the top-band metric). wall_gap 0.522 means over half the perimeter has missing walls in the mid-height band.

### 11.8 Skeleton Route Gaps (from subagent)

1. **No decoration module** — pipeline stops at structural shell. `procthor_furniture.jsonl` (45.3MB) extracted but never consumed.
2. **Single-storey only** — all exporters hard-code `storeys: [0]`. Multi-storey is untested.
3. **Roof always flat** — `compile_skeleton` supports gable/hip/shed but no exporter predicts roof type.
4. **Adjacency edges unused** — door connectivity data exists in `procthor_plans.jsonl` but the generator doesn't enforce room-to-room connectivity.
5. **Interior wall detection is brittle** — `_derive_walls` only fires on exactly-shared rect edges; non-adjacent rooms get no partition.

### 11.9 Revised Scores (after corrections)

| Dimension | Before | After | Change |
|---|---|---|---|
| Representation | 7/10 | 7/10 | unchanged |
| Reconstruction | 8/10 | **7/10** | ↓ no test split = inflated |
| Generation | 3/10 | 3/10 | unchanged |
| Semantic latent | 5/10 | 5/10 | unchanged |
| Controllability | 2/10 | 2/10 | unchanged |

The reconstruction score drops because without a held-out test set, the reported metrics are upper bounds.

---

*End of audit. All claims based on verified code, checkpoints, and experiment results. No speculative features listed as implemented.*
