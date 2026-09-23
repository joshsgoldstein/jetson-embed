#!/usr/bin/env bash
# Pull the candle model files the docs refer to, so you can read them next to
# this repo's code. Pinned to the candle version in Cargo.toml.
set -euo pipefail
ver="${1:-0.11.0}"
dest="$(dirname "$0")/../references/candle-$ver"
mkdir -p "$dest"
base="https://raw.githubusercontent.com/huggingface/candle/$ver/candle-transformers/src/models"
for f in bert.rs xlm_roberta.rs nomic_bert.rs qwen3.rs modernbert.rs with_tracing.rs; do
  curl -sSfL "$base/$f" -o "$dest/$f" && echo "fetched $f"
done
curl -sSfL "https://raw.githubusercontent.com/huggingface/candle/$ver/candle-transformers/src/utils.rs" -o "$dest/utils.rs"
echo "-> $dest"
