# Start here

This repository is two things at once: a working embedding server for the
Jetson AGX Orin, and a guided tour of what is inside one. Every module in
`src/` has a chapter here that explains the idea it implements, in the order
you would want to learn them.

## What the server does

You send text. It returns a list of numbers (a *vector*) such that texts with
similar meaning get vectors that point in similar directions. That is the whole
product. Everything else -- tokenizers, attention, pooling, CUDA -- is
machinery for computing that vector fast and correctly.

```
POST /v1/embeddings  {"input": ["hello world"], "model": "bge-base-en-v1.5"}
-->  {"data": [{"embedding": [0.012, -0.044, ...768 numbers...]}]}
```

The API matches OpenAI's, so any client that talks to OpenAI embeddings can
point at this box instead.

## The pipeline, and where each step lives

```
  "hello world"
       |
       v
  [1] tokenizer.json     text -> integer token ids            src/embedder.rs   (docs 06)
       |
       v
  [2] backbone           ids  -> one vector PER TOKEN         src/backbones/*   (docs 03, 04)
       |
       v
  [3] pooling            many token vectors -> one vector     src/pooling.rs    (docs 05)
       |
       v
  [4] normalise          scale to length 1                    src/pooling.rs    (docs 05)
       |
       v
  [5] HTTP response                                            src/server.rs     (docs 07)
```

Step 2 is the model. Steps 1, 3 and 4 are *decisions the model's authors made
at training time* that you must reproduce exactly, or the numbers are wrong
in ways that look right (docs 09 is about catching that).

## Reading order

| # | Chapter | You will understand |
|---|---|---|
| 01 | [Embeddings 101](01-embeddings-101.md) | What a vector is for, cosine similarity, why dimensions matter |
| 02 | [The model registry](02-model-registry.md) | How one binary serves BGE, nomic, arctic and pplx from their Hub files |
| 03 | [Transformer anatomy](03-transformer-anatomy.md) | Every layer in `qwen3_bidir.rs`, and how BERT differs |
| 04 | [Bidirectional attention](04-bidirectional-attention.md) | The one change that turns a chat model into an embedding model |
| 05 | [Pooling and normalisation](05-pooling-and-normalization.md) | CLS vs mean vs last token, and Matryoshka truncation |
| 06 | [Tokenization and batching](06-tokenization-and-batching.md) | Special tokens, padding, why batches are sorted by length |
| 07 | [Serving](07-serving.md) | The OpenAI API shape, the one-permit semaphore, structured errors |
| 08 | [Jetson memory](08-jetson-memory.md) | Unified memory, quadratic attention, the numbers from the GLiNER work |
| 09 | [Verification](09-verification.md) | Proving the Rust port matches Python to 0.999 cosine |
| 10 | [Your first CUDA kernel](10-first-cuda-kernel.md) | Replacing mean pooling with a hand-written kernel |
| 11 | [Roadmap](11-roadmap.md) | GTE port, quantisation, flash attention, and what to do next |

## Running it

```bash
# 1. Reference vectors from the Python stack (once per model)
cd baseline && pip install -r requirements.txt
python embed_baseline.py --model sentence-transformers/all-MiniLM-L6-v2

# 2. Build (Mac: CPU or --features metal ; Jetson: --features cuda, see docs 08)
cargo build --release

# 3. Prove it is correct before serving anything
MODEL_DIR=~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash> \
  ./target/release/jetson-embed verify --reference baseline/reference/sentence-transformers__all-MiniLM-L6-v2.json

# 4. Serve
./target/release/jetson-embed serve --model-dir <dir> --port 8015
curl -s localhost:8015/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": "hello world"}' | head -c 200
```

The model directory is whatever `hf download` (or `scripts/download-model.sh`)
produced: `config.json`, `tokenizer.json`, `model.safetensors`, and the
`1_Pooling/` folder if the model has one.

## A note on why Rust

Not speed -- the GPU does the heavy lifting either way. Rust here buys you
three things: a single static binary with no Python environment to drift on
the Jetson, explicit control over every allocation on a box where GPU memory
is RAM (docs 08), and a path to writing CUDA kernels yourself (docs 10)
without leaving the language.
