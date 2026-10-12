#!/bin/bash
# Download the minecraftbuildings dataset CSV (HuggingFace).
set -euo pipefail
cd "$(dirname "$0")/../data"
URL="https://huggingface.co/datasets/assskelad/minecraftbuildings/resolve/main/dataset.csv"
if [ -s dataset.csv ]; then
  echo "[fetch] dataset.csv already present ($(du -h dataset.csv | cut -f1))"
  exit 0
fi
curl -L --retry 5 --retry-delay 10 -C - -o dataset.csv "$URL"
echo "[fetch] done: $(du -h dataset.csv | cut -f1)"
