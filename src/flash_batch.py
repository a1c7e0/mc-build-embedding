"""Bulk vision classification via the MiMo Batch API (OpenAI protocol).

Ready-to-use companion of flash_classify.py: same rendering (double isometric
views), same prompt and answer format, same vision_types.json output — but one
asynchronous batch submission instead of live per-item calls. Use for large
jobs (thousands of items); for a few hundred items the parallel live runner is
faster than waiting out the batch window.

Modes:
  build [limit]     render unlabelled items -> out_lib/batch_input.jsonl
  submit            upload jsonl, create batch, save out_lib/batch_state.json
  status            print batch status / request counts
  collect           download output.jsonl, merge into out_lib/vision_types.json

usage: python src/flash_batch.py {build|submit|status|collect} [limit]
       MIMO_BATCH_BASE=https://batch-api-<region>.xiaomimimo.com/v1 ...
The docs endpoint is https://batch-api-${region}.xiaomimimo.com/v1 — set the
region via MIMO_BATCH_BASE or --base-url on first use. API key is read from
~/.hermes/.env / ~/.hermes/config.yaml and never printed.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flash_vision import get_key  # noqa: E402
from render import render2  # noqa: E402

PROMPT = (
    "You see ONE Minecraft build from two opposite angles (left and right "
    "views of the same model). Classify it. Answer EXACTLY in this format "
    "and nothing else:\n"
    "type: building|vehicle|vegetation|statue|sprite|other\n"
    "style: modern|medieval|asian|scifi|western|nautical|other\n"
    "note: <2-5 words>"
)
MODEL = "mimo-v2.6-flash"
DEFAULT_BASE = os.environ.get(
    "MIMO_BATCH_BASE", "https://batch-api.xiaomimimo.com/v1")


def api(method: str, path: str, body=None, raw=False):  # noqa: ANN001
    """Minimal REST call against the batch API (Bearer auth).

    The batch host has an intermittent line (SSL EOF / connect timeouts):
    retry transient failures with backoff, fail fast on client errors."""
    url = DEFAULT_BASE.rstrip("/") + path
    data = None
    headers = {"Authorization": f"Bearer {get_key()}"}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    last = ""
    for attempt in range(5):
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read() if raw else json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (408, 429) or e.code >= 500:
                last = f"HTTP {e.code}: {e.read()[:200]!r}"
            else:
                raise SystemExit(
                    f"HTTP {e.code} on {path}: {e.read()[:300]!r}\n"
                    f"If the host is wrong, set MIMO_BATCH_BASE="
                    f"https://batch-api-<region>.xiaomimimo.com/v1") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        import time as _time
        _time.sleep(3 * (attempt + 1))
    raise SystemExit(f"giving up on {path} after 5 attempts: {last}")


def upload(path: str) -> str:
    """multipart/form-data upload of a jsonl file -> file id (with retries)."""
    boundary = "----mcbatch" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f"name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(path)}\"\r\n"
        f"Content-Type: application/jsonl\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    url = DEFAULT_BASE.rstrip("/") + "/files"
    import time as _time
    last = ""
    for attempt in range(5):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Authorization": f"Bearer {get_key()}",
                     "Content-Type":
                         f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)["id"]
        except urllib.error.HTTPError as e:
            if e.code in (408, 429) or e.code >= 500:
                last = f"HTTP {e.code}: {e.read()[:200]!r}"
            else:
                raise SystemExit(
                    f"upload HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        _time.sleep(3 * (attempt + 1))
    raise SystemExit(f"upload giving up after 5 attempts: {last}")


def parse_answer(text: str) -> dict:
    t = s = "other"
    note = ""
    m = re.search(r"type:\s*([\w\- ]+?)(?:\n|$|/)", text)
    if m:
        t = m.group(1).strip().lower()
        t = t if t in ("building", "vehicle", "vegetation", "statue",
                       "sprite", "other") else "other"
    m = re.search(r"style:\s*(\w+)", text)
    if m:
        s = m.group(1).lower()
        s = s if s in ("modern", "medieval", "asian", "scifi", "western",
                       "nautical", "other") else "other"
    m = re.search(r"note:\s*(.+)", text)
    if m:
        note = m.group(1).strip()[:60]
    return {"type": t, "style": s, "note": note}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    lib = "out_lib"
    state_path = os.path.join(lib, "batch_state.json")
    out_path = os.path.join(lib, "vision_types.json")

    if mode in ("build", "build-all"):
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        meta = [json.loads(l) for l in open(os.path.join(lib, "meta.jsonl"),
                                            encoding="utf-8")]
        need = json.load(open(os.path.join(lib, "needs_vision.json")))
        done = json.load(open(out_path)) if os.path.exists(out_path) else {}
        tmp = os.path.join(lib, "vision_tmp")
        os.makedirs(tmp, exist_ok=True)
        if mode == "build-all":
            todo = [r["idx"] for r in meta if str(r["idx"]) not in done]
        else:
            todo = [i for i in need if str(i) not in done]
        if limit:
            todo = todo[:limit]
        jl_path = os.path.join(lib, "batch_input.jsonl")
        with open(jl_path, "w", encoding="utf-8") as out:
            for idx in todo:
                g = np.load(os.path.join(lib, meta[idx]["file"]))["grid"]
                png = os.path.join(tmp, f"i{idx}.png")
                render2(g, max_px=380).save(png)
                with open(png, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                out.write(json.dumps({
                    "custom_id": f"item-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "temperature": 0,
                        "max_tokens": 600,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": PROMPT}]}],
                    }}) + "\n")
        print(f"[batch] wrote {len(todo)} requests -> {jl_path}", flush=True)

    elif mode == "submit":
        jl_path = os.path.join(lib, "batch_input.jsonl")
        if not os.path.exists(jl_path):
            raise SystemExit("run `build` first")
        # the line resets large bodies: split into ~250-request chunks, one
        # batch per chunk; state stores every batch id
        lines = [l for l in open(jl_path, encoding="utf-8") if l.strip()]
        chunk_size = 250
        ids: list[str] = []
        for ci in range(0, len(lines), chunk_size):
            part = os.path.join(lib, f"batch_part_{ci // chunk_size}.jsonl")
            open(part, "w", encoding="utf-8").writelines(
                lines[ci:ci + chunk_size])
            file_id = upload(part)
            # the batches endpoint 500s while the file is still validating
            for _ in range(20):
                rec = api("GET", f"/files/{file_id}")
                if rec.get("status") == "active":
                    break
                time.sleep(3)
            batch = api("POST", "/batches", {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "name": f"{lib}-part{ci // chunk_size}"})
            ids.append(batch["id"])
            print(f"[batch] part {ci // chunk_size}: {batch['id']} "
                  f"({min(chunk_size, len(lines) - ci)} requests)", flush=True)
            json.dump({"ids": ids}, open(state_path, "w"))
        print(f"[batch] submitted {len(ids)} batches", flush=True)

    elif mode == "status":
        if not os.path.exists(state_path):
            raise SystemExit("no batch_state.json — nothing submitted")
        st = json.load(open(state_path))
        ids = st.get("ids") or [st.get("id")]
        all_done = True
        for bid in ids:
            b = api("GET", f"/batches/{bid}")
            rc = b.get("request_counts", {})
            print(f"[batch] {bid}: {b.get('status')} "
                  f"{rc.get('completed', 0)}/{rc.get('total', 0)}", flush=True)
            if b.get("status") != "completed":
                all_done = False
        if all_done:
            print("[batch] ALL COMPLETED", flush=True)

    elif mode == "collect":
        if not os.path.exists(state_path):
            raise SystemExit("no batch_state.json — nothing submitted")
        st = json.load(open(state_path))
        ids = st.get("ids") or [st.get("id")]
        done = json.load(open(out_path)) if os.path.exists(out_path) else {}
        n_new = 0
        for bid in ids:
            b = api("GET", f"/batches/{bid}")
            if b.get("status") != "completed":
                print(f"[batch] {bid}: {b.get('status')} — skipped", flush=True)
                continue
            out_file = b.get("output_file_id") or b.get("error_file_id")
            if not out_file:
                continue
            raw = api("GET", f"/files/{out_file}/content", raw=True)
            with open(os.path.join(lib, f"batch_output_{bid}.jsonl"),
                      "wb") as fh:
                fh.write(raw)
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = rec["custom_id"].split("-", 1)[1]
                if idx in done:
                    continue
                try:
                    text = rec["response"]["body"]["choices"][0]["message"][
                        "content"]
                except (KeyError, TypeError, IndexError):
                    continue
                done[idx] = parse_answer(text)
                n_new += 1
        json.dump(done, open(out_path, "w"))
        print(f"[batch] collected {n_new} new -> {out_path} "
              f"(total {len(done)})", flush=True)

    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
