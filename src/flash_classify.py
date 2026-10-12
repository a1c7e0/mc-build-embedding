"""Bulk type/style classification: double-view render + stateless flash vision.

For every item in out_lib/needs_vision.json: render two opposite isometric
views (0/180 deg) into one image, ask mimo-v2.6-flash (no conversation
context, one image + one prompt), parse a strict-format answer.

Output: out_lib/vision_types.json  {idx: {"type":..., "style":..., "note":...}}
Logs progress to stdout; safe to re-run (skips already-labelled items).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render import render2  # noqa: E402

PROMPT = (
    "You see ONE Minecraft build from two opposite angles (left and right "
    "views of the same model). Classify it. Answer EXACTLY in this format "
    "and nothing else:\n"
    "type: building|vehicle|vegetation|statue|sprite|other\n"
    "style: modern|medieval|asian|scifi|western|nautical|other\n"
    "note: <2-5 words>"
)


def main() -> None:
    lib = sys.argv[1] if len(sys.argv) > 1 else "out_lib"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                        encoding="utf-8")]
    need = json.load(open(os.path.join(lib, "needs_vision.json")))
    out_path = os.path.join(lib, "vision_types.json")
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}

    tmp = os.path.join(lib, "vision_tmp")
    os.makedirs(tmp, exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "flash_vision.py")

    n_run = 0
    norm_type = {"treehouse": "building", "house": "building", "cabin": "building",
                 "hut": "building", "ship": "vehicle", "boat": "vehicle",
                 "tree": "vegetation", "bush": "vegetation",
                 "sculpture": "statue", "pixel": "sprite", "logo": "sprite"}
    def work(idx: int):  # noqa: ANN002
        rec = meta[idx]
        g = np.load(os.path.join(lib, rec["file"]))["grid"]
        png = os.path.join(tmp, f"i{idx}.png")
        render2(g, max_px=380).save(png)
        text = ""
        for attempt in range(3):
            try:
                res = subprocess.run(
                    [sys.executable, script, png, PROMPT, "--max-tokens", "600"],
                    capture_output=True, text=True, timeout=240)
                text = res.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError,
                    OSError):
                text = ""
            if re.search(r"note:\s*\S", text):
                break
            time.sleep(2 + attempt * 2)
        if not re.search(r"note:\s*\S", text):
            return idx, None
        t = s = "other"
        note = ""
        m = re.search(r"type:\s*([\w\- ]+?)(?:\n|$|/)", text)
        if m:
            t = m.group(1).strip().lower()
            t = norm_type.get(t, t if t in ("building", "vehicle",
                                            "vegetation", "statue",
                                            "sprite", "other") else "other")
        m = re.search(r"style:\s*(\w+)", text)
        if m:
            s = m.group(1).lower()
            s = s if s in ("modern", "medieval", "asian", "scifi", "western",
                           "nautical", "other") else "other"
        m = re.search(r"note:\s*(.+)", text)
        if m:
            note = m.group(1).strip()[:60]
        return idx, {"type": t, "style": s, "note": note}

    todo = [i for i in need if str(i) not in done]
    if os.environ.get("FLASH_ALL") == "1":
        todo = [r["idx"] for r in meta if str(r["idx"]) not in done]
    if limit:
        todo = todo[:limit]
    n_run = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work, i) for i in todo]):
            idx, rec = fut.result()
            if rec is None:
                # do not poison vision_types with a fake "other": retry next run
                print(f"[flash] skip {idx}: no answer after 3 attempts",
                      flush=True)
                continue
            done[str(idx)] = rec
            n_run += 1
            if n_run % 20 == 0:
                json.dump(done, open(out_path, "w"))
                print(f"[flash] {n_run}/{len(todo)} done "
                      f"(last {idx}: {rec['type']}/{rec['style']})", flush=True)
    json.dump(done, open(out_path, "w"))
    counts = {}
    for v in done.values():
        counts[v["type"]] = counts.get(v["type"], 0) + 1
    print(f"[flash] finished {n_run} new, total {len(done)}: {counts}",
          flush=True)


if __name__ == "__main__":
    main()
