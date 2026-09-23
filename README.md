# jetson-embed

An OpenAI-compatible text-embedding server in Rust, built on
[candle](https://github.com/huggingface/candle), for the Jetson AGX Orin --
and a guided tour of what is inside one. **Start with
[docs/00-start-here.md](docs/00-start-here.md).**

```
POST /v1/embeddings  {"input": ["hello world"]}
-->  {"data": [{"embedding": [0.012, -0.044, ...]}], "usage": {"prompt_tokens": 4}}
```

## Models

One binary serves any of these from its Hugging Face download directory.
The registry (`src/registry.rs`) reads `config.json` to pick the backbone
and `1_Pooling/config.json` to pick the pooling.

| Family | Backbone | Examples | Status |
|---|---|---|---|
| BERT | candle `bert` | BAAI/bge-*-en-v1.5, all-MiniLM-L6-v2, arctic-embed v1 | written, unverified |
| XLM-RoBERTa | candle `xlm_roberta` | BAAI/bge-m3, arctic-embed-l-v2.0, multilingual-e5 | written, unverified |
| NomicBERT | candle `nomic_bert` | nomic-embed-text-v1 / v1.5 | written, unverified |
| Qwen3 bidirectional | **this repo** (`qwen3_bidir.rs`) | perplexity-ai/pplx-embed-v1-0.6b | written, unverified |
| GTE | -- | arctic-embed-m-v2.0, gte-multilingual | not yet (docs/11) |

"Unverified" means: the code exists and the reference vectors are generated,
but `verify` has not yet been run against them. See *Status* below.

## Quick start

```bash
# reference vectors from the Python stack (once per model)
cd baseline && pip install -r requirements.txt
python embed_baseline.py --model sentence-transformers/all-MiniLM-L6-v2
cd ..

# build: Mac CPU (default), Mac GPU (--features metal), Jetson (--features cuda, see below)
cargo build --release

# prove correctness, then serve
export MODEL_DIR=~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>
./target/release/jetson-embed verify --reference baseline/reference/sentence-transformers__all-MiniLM-L6-v2.json
./target/release/jetson-embed serve --port 8015
```

`MODEL_DIR` is any directory with `config.json`, `tokenizer.json`,
`model.safetensors` (+ `1_Pooling/` if the model has it): a Hub cache
snapshot, or the output of `scripts/download-model.sh <org/name>`.

## Configuration

All flags are also environment variables (`--max-batch-tokens` = `MAX_BATCH_TOKENS`).

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | required | Model directory |
| `MODEL_NAME` | dir name | Name in `/v1/models` and accepted in `"model"` |
| `DTYPE` | `f16` | `f32` / `f16` / `bf16`. CPU forces f32 |
| `DEVICE` | auto | `cpu` / `cuda` / `metal`; auto = best compiled-in |
| `MAX_SEQ_LEN` | model's | Truncate inputs to this many tokens |
| `MAX_BATCH_TOKENS` | `16384` | Bound on inputs x padded length per forward pass -- **the memory knob** (docs/08) |
| `DEFAULT_PREFIX` | `""` | Prepended to every input (nomic: `search_document: `) |
| `MAX_INPUTS` | `256` | Inputs per request before a 413 |
| `PORT` / `HOST` | `8015` / `0.0.0.0` | |

## Building for the Jetson

candle's CUDA backend needs `nvcc`. JetPack 7.2 hosts do not ship it, so
build inside the JetPack 6 container (it runs unchanged on the JP7 host):

```bash
docker build -f Dockerfile.jetson -t jetson-embed:cuda .
docker run --rm --runtime nvidia -p 8015:8015 -v /ssd/hf-cache/embed/bge-base:/model \
  -e MODEL_DIR=/model jetson-embed:cuda
```

Never use the `sbsa` / `cu130` toolchains or wheels on an Orin: they target
sm_90/sm_100, not sm_87.

## Layout

```
src/
  main.rs            CLI: serve | embed | verify
  settings.rs        flags and env vars
  registry.rs        model dir -> backbone + pooling + limits
  backbones/         Encoder trait; bert, xlm_roberta, nomic (candle), qwen3_bidir (here)
  pooling.rs         cls / mean / last-token, L2 norm, Matryoshka truncation
  embedder.rs        tokenize -> sort -> batch by token budget -> encode -> pool
  server.rs          axum: /v1/embeddings, /v1/models, /health
  verify.rs          cosine check against baseline/reference/*.json
baseline/            Python sentence-transformers reference generator
docs/                the course: 00 to 11
scripts/             download-model.sh, fetch-candle-sources.sh
Dockerfile.jetson    CUDA build in the l4t-jetpack container
```

## Status (2026-09-23)

* All source written; **not yet compiled**: the dev Mac's Gatekeeper daemon
  was wedged, blocking first-launch of the Rust toolchain. Compile and
  `verify` are the next step; expect a round of compiler fixes.
* Reference vectors generated for all-MiniLM-L6-v2, arctic-embed-s and
  pplx-embed-v1-0.6B (`baseline/reference/`, git-ignored; regenerate with
  the script).
* Not deployed anywhere. Memory budget on the AGX: ~3-4 GiB (docs/08).
