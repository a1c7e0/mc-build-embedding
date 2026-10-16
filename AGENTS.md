# Repository Guidelines

## Project Structure & Module Organization

This Python research repository generates Minecraft buildings and worlds. Run commands from the repository root; many scripts use relative data paths.

- `src/`: skeleton compilation, decoration, terrain, assembly, rendering, schematic export, and experimental VQ-VAE/prior models.
- `src/plan_gen/`: GPT-based room-layout generation, training, metrics, and DSL export.
- `src/stylegen/`: seeded architectural grammars and DSL validation.
- `docs/`: DSL specification, architecture audits, experiment results, and roadmap notes. Historical reports may describe older implementations.
- `prompts/`: JSON world specifications; `scripts/`: data-fetching utilities.
- `out_lib/`: derived analysis, motif libraries, and style examples. Raw datasets and several large artifacts are ignored.
- `out/`: generated runs and previews; keep new disposable outputs here.

## Build, Test, and Development Commands

Use Python 3.11, matching the Docker image:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Neural training and sampling additionally require PyTorch; it is absent from `requirements.txt`. `Dockerfile.torch` defines a CPU installation.

```bash
docker build -t mcb-embed:latest .
python src/skeleton_compile.py skeleton.json out/shell.npz out/shell.png
python src/stylegen/generate_all.py --seeds 1 --out out/stylegen_check
PYTHONPATH=src python -c "from plan_gen.metrics import self_test; print(self_test())"
```

These build the base image, compile a supplied skeleton, validate/render four style examples, and check layout metrics, respectively.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Add type hints and CLI docstrings where consistent with nearby code. No formatter or linter is configured. Preserve voxel axis order `[x, y, z]`, `uint16` block IDs, and documented DSL semantics.

## Testing Guidelines

There is no centralized test framework or coverage threshold. Use focused assertions, existing metric self-tests, and generated previews. Compare voxel arrays for deterministic changes; inspect cutaway renders for geometry changes.

For neural pipeline changes, run `python src/plan_gen/smoke.py --steps 300 --batch 32 --samples 48` when PyTorch and `out_lib/plans/procthor_plans.jsonl` are available. Report missing prerequisites and distinguish measured results from historical metrics.

## Commit & Pull Request Guidelines

History currently contains one descriptive commit; no formal commit convention is established. Use concise, action-oriented subjects. PRs should explain behavior changes, affected pipeline stages, validation commands, dataset/checkpoint prerequisites, and relevant issues. Include before/after renders for visual changes. Keep raw datasets, checkpoints, credentials, and disposable outputs out of commits; preserve intentionally tracked derived library assets.
