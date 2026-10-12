"""stylegen driver: generate -> validate -> compile -> contact sheets.

Every style module turns (style grammar, seed) into a Skeleton-DSL v0
JSON. The JSON is compiled UNCHANGED with the EXISTING deterministic
compiler (src/skeleton_compile.py, invoked as a CLI) into a voxel shell
(npz) and a two-view cutaway render (png). One contact sheet per style
and a style label table (style = construction parameters, exact by
construction) complete the annotated dataset.

usage: python src/stylegen/generate_all.py [--seeds N] [--styles a,b]
                                           [--out OUT_DIR]

Run inside the project container:
  docker run --rm --cpus=1 --memory=1200m -v <repo>:/work -w /work \\
      mcb-embed:latest python src/stylegen/generate_all.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from stylegen import (  # noqa: E402
    chinese_temple, gothic_church, japanese_house, nautical_dock,
)
from stylegen.dsl import validate  # noqa: E402
from stylegen.sheet import make_sheet  # noqa: E402

COMPILER = os.path.join(SRC, "skeleton_compile.py")

STYLES = {
    chinese_temple.NAME: {
        "module": chinese_temple, "seed_base": 1100,
        "material": {"wall": "wood", "floor": "mineral", "roof": "mineral"},
        "features": [
            "podium raised 1-2 blocks with front stair (platform)",
            "wood colonnade at platform edge, symmetric bays",
            "mineral gable/hip roof, deep overhanging eave",
            "eave-lip tier band per storey (double-eave look)",
            "wide centred gate (3-5 blocks)",
        ],
    },
    japanese_house.NAME: {
        "module": japanese_house, "seed_base": 2200,
        "material": {"wall": "wood", "floor": "wood", "roof": "wood"},
        "features": [
            "low gable roof (pitch 2-3) with wide eaves",
            "engawa veranda band with wood pier colonnade",
            "dense 1x2 shoji window arrays, 2-3 block spacing",
            "all-timber material palette",
            "small off-centre entrance (1-2 wide)",
        ],
    },
    gothic_church.NAME: {
        "module": gothic_church, "seed_base": 3300,
        "material": {"wall": "mineral", "floor": "mineral", "roof": "mineral"},
        "features": [
            "tall narrow massing (21-29 x 13-17, sh 10-12)",
            "steep gable roof (pitch 1-2)",
            "narrow tall lancet window arrays",
            "full-height buttress piers in a 2-deep wall slot",
            "cruciform room tiling + transept walls with tall arches",
        ],
    },
    nautical_dock.NAME: {
        "module": nautical_dock, "seed_base": 4400,
        "material": {"wall": "wood", "floor": "wood", "roof": "wood"},
        "features": [
            "raised wooden pier deck (2-3) on pilings",
            "warehouse box with 3-5 wide loading doors at deck level",
            "shed roof over transit strip / low gable over wharf hall",
            "stepped gangway ramp down from the deck",
            "all-timber material palette",
        ],
    },
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def compile_one(json_path: str, npz_path: str, png_path: str) -> str:
    """Invoke the EXISTING compiler CLI; return its stdout."""
    res = subprocess.run(
        [sys.executable, COMPILER, json_path, npz_path, png_path],
        cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"compiler failed for {json_path}\n{res.stdout}\n{res.stderr}")
    return res.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="variants per style (>=8 required by the spec)")
    ap.add_argument("--styles", type=str, default="",
                    help="comma-separated subset of styles")
    ap.add_argument("--out", type=str,
                    default=os.path.join(ROOT, "out_lib", "stylegen"))
    args = ap.parse_args()

    wanted = [s.strip() for s in args.styles.split(",") if s.strip()] \
        or list(STYLES)
    for name in wanted:
        if name not in STYLES:
            raise SystemExit(f"unknown style {name!r}; known: {list(STYLES)}")
    os.makedirs(args.out, exist_ok=True)

    labels = {
        "generated_at": utc_now(),
        "generator": "src/stylegen/generate_all.py",
        "compiler": "src/skeleton_compile.py",
        "label_semantics":
            "style is a deterministic function of the construction "
            "parameters (module + seed + knobs), so labels are exact",
        "styles": {},
        "variants": {},
    }
    compiled = 0
    total = 0
    sheet_files = []

    for name in wanted:
        spec = STYLES[name]
        mod = spec["module"]
        labels["styles"][name] = {
            "module": f"src/stylegen/{name}.py",
            "seed_base": spec["seed_base"],
            "material": spec["material"],
            "discriminative_features": spec["features"],
        }
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        tiles = []
        for i in range(args.seeds):
            seed = spec["seed_base"] + i
            vid = f"{name}_v{i + 1:02d}"
            sk, params = mod.generate(seed)
            issues = validate(sk)
            if issues:
                raise SystemExit(f"{vid}: invalid skeleton: {issues}")
            json_path = os.path.join(out_dir, f"{vid}.json")
            npz_path = os.path.join(out_dir, f"{vid}.npz")
            png_path = os.path.join(out_dir, f"{vid}.png")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(sk, fh, indent=2)
                fh.write("\n")
            log = compile_one(json_path, npz_path, png_path)
            compiled += 1
            total += 1
            labels["variants"][vid] = {
                "style": name, "seed": seed, "params": params,
                "files": {
                    "skeleton": os.path.relpath(json_path, ROOT),
                    "voxels": os.path.relpath(npz_path, ROOT),
                    "render": os.path.relpath(png_path, ROOT),
                },
                "compile_log": log,
            }
            short = ", ".join(
                f"{k}={params[k]}" for k in
                ("storeys", "roof", "pitch", "footprint")
                if k in params)
            tiles.append((f"{vid}  seed={seed}  {short}", png_path))
            print(f"[stylegen] {vid}: ok {log}", flush=True)

        sheet_path = os.path.join(args.out, f"{name}_sheet.png")
        make_sheet(tiles, sheet_path,
                   f"{name} -- {args.seeds} seeded variants "
                   f"(cutaway + full view, src/skeleton_compile.py renders)")
        sheet_files.append(os.path.relpath(sheet_path, ROOT))
        print(f"[stylegen] sheet -> {sheet_path}", flush=True)

    labels["summary"] = {
        "variants": total,
        "compiled_ok": compiled,
        "per_style": {n: args.seeds for n in wanted},
        "sheets": sheet_files,
    }
    labels_path = os.path.join(args.out, "style_labels.json")
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)
        fh.write("\n")
    print(f"[stylegen] labels -> {labels_path}", flush=True)
    print(f"[stylegen] DONE {compiled}/{total} compiled", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
