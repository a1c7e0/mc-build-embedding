"""Resubmit wedged batch parts as fresh batches; append ids to batch_state.

usage: python src/resubmit_parts.py 1 4 7
(part numbers = indices of batch_part_<n>.jsonl)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import flash_batch as fb  # noqa: E402

lib = "out_lib"
state_path = os.path.join(lib, "batch_state.json")
st = json.load(open(state_path))
ids = st.get("ids") or [st.get("id")]
for n in sys.argv[1:]:
    part = os.path.join(lib, f"batch_part_{n}.jsonl")
    fid = fb.upload(part)
    for _ in range(20):
        rec = fb.api("GET", f"/files/{fid}")
        if rec.get("status") == "active":
            break
        time.sleep(3)
    batch = fb.api("POST", "/batches", {
        "input_file_id": fid,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "name": f"{lib}-part{n}-retry"})
    ids.append(batch["id"])
    json.dump({"ids": ids}, open(state_path, "w"))
    print(f"[resubmit] part {n} -> {batch['id']}", flush=True)
print(f"[resubmit] state now tracks {len(ids)} batches", flush=True)
