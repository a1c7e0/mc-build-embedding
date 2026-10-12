"""Stateless vision calls to mimo-v2.6-flash (OpenAI-compatible).

No conversation context: one image + one prompt per request. Used for bulk
classification where the pro-model conversation channel is overkill.

usage: python src/flash_vision.py <image.png> "<prompt>" [--model NAME]
API key is read from ~/.hermes/config.yaml and never printed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.request


def get_key() -> str:
    # 1) hermes secrets env (KEY=value, never printed)
    for path in (os.path.expanduser("~/.hermes/.env"),
                 os.path.expanduser("~/.env")):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for line in text.splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, val = line.split("=", 1)
            name = name.strip().upper()
            val = val.strip().strip("\"'")
            if val and ("KEY" in name) and any(
                    s in name for s in ("XIAOMI", "MIMO", "OPENROUTER", "API")):
                return val
    # 2) hermes config.yaml inline key
    try:
        text = open(os.path.expanduser("~/.hermes/config.yaml"),
                    encoding="utf-8").read()
    except OSError:
        text = ""
    m = re.search(r"""^\s*api_key:\s*["']?([A-Za-z0-9_\-\.]+)["']?""",
                  text, re.M)
    if m:
        return m.group(1)
    raise SystemExit("no api_key found in env files or config")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("prompt")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    args = ap.parse_args()

    with open(args.image, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": args.prompt},
            ],
        }],
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {get_key()}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip():
        # some reasoning models put the answer elsewhere when max_tokens is tight
        text = msg.get("reasoning_content") or json.dumps(msg)[:500]
    print(text)


if __name__ == "__main__":
    main()
