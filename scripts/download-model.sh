#!/usr/bin/env bash
# Download a Hugging Face embedding model into ./models/<org>__<name>/ so the
# Rust server can load it from disk. Uses the `hf` CLI (pip install huggingface_hub).
# On the AGX, point MODELS_DIR at the SSD: MODELS_DIR=/ssd/hf-cache/embed ./scripts/download-model.sh BAAI/bge-base-en-v1.5
set -euo pipefail
repo="${1:?usage: download-model.sh <org/name>}"
dest="${MODELS_DIR:-./models}/${repo//\//__}"
mkdir -p "$dest"
hf download "$repo" --local-dir "$dest" \
  --include "config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json" \
            "model.safetensors" "1_Pooling/config.json" "modules.json" "sentence_bert_config.json" \
            "config_sentence_transformers.json" "vocab.txt" "merges.txt" "vocab.json" "added_tokens.json"
echo "-> $dest"; du -sh "$dest"
