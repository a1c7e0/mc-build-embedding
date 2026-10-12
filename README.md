# mc-build-embedding

Minecraft building generation: skeleton DSL compiler, style-aware decoration, and procedural world generation.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  World Generation                │
│  terrain → roads → building placement → render   │
├─────────────────────────────────────────────────┤
│               Building Generation                │
│  plan_gen → skeleton DSL → compile → decoration  │
├─────────────────────────────────────────────────┤
│                Decoration System                 │
│  MC motifs + ProcTHOR placement + style rules    │
└─────────────────────────────────────────────────┘
```

### Three Lines

1. **Skeleton Route** (structure): plan generator → skeleton DSL → deterministic compiler → voxel shell
2. **Decoration** (style): MC corpus motif extraction + ProcTHOR furniture placement + style-material mapping
3. **World Generation** (application): terrain + roads + building placement + rendering

## Key Components

### Skeleton DSL (`src/skeleton_compile.py`)
Deterministic compiler: JSON skeleton → Minecraft voxel shell.
Supports walls, rooms, openings (doors/windows), floors, roof (flat/gable/hip/shed), materials.

### Plan Generator (`src/plan_gen/`)
GPT model (dim=128, 4L, ~805k params) trained on 10,000 ProcTHOR house plans.
Test accuracy: 73.1% (type 89.4%, coord 67.3%).

### Decoration System
- `src/decoration_vocab.py` — extract decoration patterns from 3,262 MC buildings
- `src/deco_style_table.py` — style→block probability table (7 styles × 4 positions)
- `src/motifs_noskeleton.py` — skeleton removal + multi-block motif extraction (245 furniture patterns)
- `src/apply_motifs.py` — motif-based furniture placement (3×3×3 multi-block patterns)
- `src/decorate_rooms.py` — room-aware decoration (bedroom→beds, kitchen→fridges)

### ProcTHOR Integration
- `src/procthor_ingest.py` — JSON → 6-class voxels with roof rasterization
- `src/procthor_plans.py` — plan ground truth (37,897 rooms, 63.2% rect-exact)
- `src/procthor_to_dsl.py` — house → skeleton DSL → compiled shell
- `src/procthor_placement.py` — furniture placement rules (58,474 items analyzed)
- `src/procthor_room_furniture.py` — room→furniture associations (145,590 matches)

### World Generation
- `src/worldgen.py` — terrain (fbm, erosion, rivers, floating islands)
- `src/assemble2.py` — building placement with style-anchored districts
- `src/roads.py` — road/path overlay with building collision avoidance
- `src/render_world.py` / `src/render_closeup.py` — visualization

## Data Sources

- **MC Building Library**: [rom1504/minecraft-schematics-dataset](https://github.com/rom1504/minecraft-schematics-dataset) + [assskelad/minecraftbuildings](https://huggingface.co/datasets/assskelad/minecraftbuildings) (3,262 buildings)
- **ProcTHOR-10K**: [allenai/procthor-10k](https://huggingface.co/datasets/allenai/procthor-10k) (10,000 house plans + furniture)

## Usage

```bash
# Compile a skeleton to voxel shell
python src/skeleton_compile.py skeleton.json output.npz

# Generate a plan (program → plan → skeleton → shell)
python src/plan_gen/sample.py --types Bedroom,Bathroom,Kitchen --dsl

# Apply decoration
python src/decorate_rooms.py grid.npz skeleton.json modern 42 output.npz

# Generate a world with roads
python src/roads.py out/world/valley_port
```

## Results

### VQ-VAE (Vector Route — paused)
- 32³: occupied IoU 0.836, wall-F1 0.887
- 64³: occupied IoU 0.895, wall-F1 0.896 (rare materials doubled)
- 64³ codebook: architectural part semantics (wall segments, openings, corners)

### Skeleton Route
- Plan generator: 73.1% test accuracy
- Compiler: 100% deterministic, supports walls/openings/floors/roof
- Decoration: 245 furniture motifs, 7-style material mapping, room-aware placement

### World Generation
- valley_port: 512×512 terrain, 72 buildings, roads + paths
- Reproducibility: verified (same seed → identical world)

## License

MIT
