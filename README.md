# jetson-embed

**An OpenAI-compatible text-embedding server in Rust, built on [candle](https://github.com/huggingface/candle), for the NVIDIA Jetson AGX Orin.**

One ~8 MB binary serves sentence-transformers models from a Hugging Face model directory, with output checked against the Python reference implementation. It doubles as a learning project: every module has a matching chapter in [`docs/`](docs/00-start-here.md) explaining the transformer, pooling, batching, and memory decisions behind it.

```bash
curl localhost:8015/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "all-MiniLM-L6-v2", "input": ["hello world", "goodbye"]}'
```

> **Status:** CPU inference is verified against sentence-transformers for three models (see [Verification](#verification)). The Metal and CUDA builds are wired up but have not been compiled or verified yet. Expect rough edges.

---

## Contents

- [Why](#why)
- [Features](#features)
- [Supported models](#supported-models)
- [Verification](#verification)
- [Quick start](#quick-start)
- [API](#api)
- [Configuration](#configuration)
- [Building for the Jetson](#building-for-the-jetson)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Learning path](#learning-path)
- [Roadmap](#roadmap)
- [License](#license)

## Why

The usual way to serve embeddings is a Python container: PyTorch, transformers, sentence-transformers, and a web framework, several gigabytes in all. On a Jetson that has three costs:

- **Image size and cold start.** A multi-GB image to run a 22M-parameter model.
- **Unpredictable memory.** On the Orin, the CPU and GPU share one memory pool. A GPU allocation spike doesn't produce a clean out-of-memory error; it can trigger the kernel OOM-killer against whatever process is largest.
- **Wedged processes.** PyTorch's out-of-memory path calls NVML, which Tegra doesn't fully support. Instead of raising an error, the process can hang.

`jetson-embed` is a single binary. It bounds memory per forward pass with a token budget, and a failed CUDA allocation comes back as an ordinary `Err`. The background is in [`docs/08-jetson-memory.md`](docs/08-jetson-memory.md).

## Features

- **OpenAI-compatible API:** `POST /v1/embeddings` and `GET /v1/models`. Works with the official OpenAI SDKs and anything built on them (LangChain, LlamaIndex, LiteLLM).
- **`float` and `base64` encodings.** The OpenAI Python SDK requests `base64` by default; both decode to bit-identical vectors.
- **Matryoshka truncation** via `dimensions`, re-normalized to unit length after truncating.
- **Model registry** that reads the model's own `config.json`, `1_Pooling/config.json`, `sentence_bert_config.json` and `modules.json`. Pooling mode, max length and post-pooling stages come from the model, not hard-coded guesses. An unknown pipeline stage is a startup error rather than silently wrong output.
- **Token-budget batching.** Inputs are sorted by length and grouped so that `inputs × padded length` stays under `MAX_BATCH_TOKENS`, which bounds peak memory.
- **Single-flight inference** behind a semaphore, on a blocking thread pool, so the async HTTP layer never stalls.
- **Structured errors** with stable machine-readable codes.
- **Built-in `verify` command** that checks output against reference vectors produced by sentence-transformers.

## Supported models

| Family | Backbone | Example models | Status |
|---|---|---|---|
| BERT | candle `bert` | `sentence-transformers/all-MiniLM-L6-v2`, `Snowflake/snowflake-arctic-embed-s`, `BAAI/bge-*-en-v1.5` | ✅ verified (MiniLM, arctic-embed-s) |
| Qwen3, bidirectional | this repo ([`qwen3_bidir.rs`](src/backbones/qwen3_bidir.rs)) | `perplexity-ai/pplx-embed-v1-0.6B` | ✅ verified |
| XLM-RoBERTa | candle `xlm_roberta` | `BAAI/bge-m3`, `Snowflake/snowflake-arctic-embed-l-v2.0`, `intfloat/multilingual-e5-*` | written, unverified |
| NomicBERT | candle `nomic_bert` | `nomic-ai/nomic-embed-text-v1`, `v1.5` | written, unverified |
| GTE | — | `Snowflake/snowflake-arctic-embed-m-v2.0`, `Alibaba-NLP/gte-multilingual-base` | not supported yet ([roadmap](docs/11-roadmap.md)) |

The Qwen3 encoder is the one backbone written from scratch: candle's Qwen3 is a causal decoder, and an embedding model needs every token to see every other token. It's a full transformer in about 260 lines and a good place to start reading.

## Verification

`jetson-embed verify` embeds a fixed set of 7 test texts and compares each vector against one produced by sentence-transformers. The texts include edge cases: an empty string, a single character, non-ASCII text, and a padding stress test. Results on macOS arm64, CPU, f32:

| Model | Min cosine | Max \|diff\| | Notes |
|---|---|---|---|
| all-MiniLM-L6-v2 | 1.00000 | 3.9e-7 | float32 rounding noise |
| snowflake-arctic-embed-s | 1.00000 | 2.4e-7 | float32 rounding noise |
| pplx-embed-v1-0.6B | 1.00000 | 1.2e-3 | exact except one component one int8 step apart (a rounding tie) |

pplx-embed ends its pipeline with an int8 quantization stage (`round(tanh(x) × 127)`), declared in its `modules.json`. Values that land exactly on a .5 boundary can round differently on different hardware, which accounts for the single 1.2e-3 difference.

To reproduce, generate the reference vectors with the Python stack, then run `verify`:

```bash
cd baseline && pip install -r requirements.txt
python embed_baseline.py --model sentence-transformers/all-MiniLM-L6-v2
cd ..
./target/release/jetson-embed verify \
  --model-dir <model-dir> \
  --reference baseline/reference/sentence-transformers__all-MiniLM-L6-v2.json
```

## Quick start

**Prerequisites:** Rust (the toolchain is pinned in `rust-toolchain.toml` and installed automatically by `rustup`), plus a model downloaded from the Hugging Face Hub.

```bash
git clone https://github.com/joshsgoldstein/jetson-embed.git
cd jetson-embed
cargo build --release

# Fetch a model (uses the `hf` CLI: pip install huggingface_hub)
scripts/download-model.sh sentence-transformers/all-MiniLM-L6-v2

./target/release/jetson-embed serve \
  --model-dir models/sentence-transformers__all-MiniLM-L6-v2 \
  --model-name all-MiniLM-L6-v2
```

A Hugging Face cache snapshot also works as `--model-dir`, e.g. `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>`.

The server listens on `0.0.0.0:8015` by default.

## API

### `POST /v1/embeddings`

| Field | Type | Description |
|---|---|---|
| `input` | string or string[] | Text(s) to embed. Required; no empty strings. |
| `model` | string | Optional. If set, must match the served model name. |
| `encoding_format` | `"float"` \| `"base64"` | Default `float`. `base64` is the vector's little-endian f32 bytes, base64-encoded. |
| `dimensions` | integer | Truncate to the first N dimensions and re-normalize (Matryoshka). |
| `prefix` | string | *Extension.* Task prefix prepended to each input, overriding `DEFAULT_PREFIX` (e.g. nomic's `search_query: `). |

Response:

```json
{
  "object": "list",
  "data": [{ "object": "embedding", "index": 0, "embedding": [0.0123, -0.0441, ...] }],
  "model": "all-MiniLM-L6-v2",
  "usage": { "prompt_tokens": 4, "total_tokens": 4 }
}
```

All vectors are L2-normalized, so the dot product equals cosine similarity.

### `GET /v1/models`

Lists the served model.

### `GET /health`

Model, backbone, pooling, device, max sequence length and in-flight request count.

### Errors

Errors use a stable machine-readable code:

```json
{ "error": { "code": "TOO_MANY_INPUTS", "message": "300 inputs, limit is 256; split into several requests ..." } }
```

| Code | Status | Meaning |
|---|---|---|
| `EMPTY_INPUT` | 400 | `input` is an empty array |
| `EMPTY_STRING` | 400 | An input is `""`, which has no meaningful embedding |
| `TOKEN_INPUT_UNSUPPORTED` | 400 | Pre-tokenized input (integer arrays). Send text instead. |
| `UNSUPPORTED_ENCODING` | 400 | `encoding_format` is not `float` or `base64` |
| `MODEL_NOT_FOUND` | 404 | `model` doesn't match the served model |
| `TOO_MANY_INPUTS` | 413 | More than `MAX_INPUTS` inputs in one request |
| `INFERENCE_FAILED` | 500 | The forward pass returned an error |

### Client examples

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8015/v1", api_key="unused")
resp = client.embeddings.create(model="all-MiniLM-L6-v2", input=["hello world"])
vector = resp.data[0].embedding
```

**LangChain.** By default, `OpenAIEmbeddings` tokenizes with OpenAI's tokenizer and sends integer token IDs, which don't correspond to this model's vocabulary. Turn that off so it sends text:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="all-MiniLM-L6-v2",
    base_url="http://localhost:8015/v1",
    api_key="unused",
    check_embedding_ctx_length=False,
)
```

## Configuration

Every flag can also be set as an environment variable (`--max-batch-tokens` → `MAX_BATCH_TOKENS`).

| Setting | Default | Description |
|---|---|---|
| `MODEL_DIR` | *required* | Directory with `config.json`, `tokenizer.json`, `model.safetensors` (+ `1_Pooling/`, `modules.json` if present) |
| `MODEL_NAME` | directory name | Name reported by `/v1/models` and accepted in `model` |
| `DTYPE` | `f16` | `f32`, `f16` or `bf16`. Forced to `f32` on CPU. |
| `DEVICE` | best available | `cpu`, `cuda` or `metal`. Requesting a device that wasn't compiled in is an error, never a silent CPU fallback. |
| `MAX_SEQ_LEN` | model's trained length | Inputs longer than this are truncated |
| `MAX_BATCH_TOKENS` | `16384` | Upper bound on `inputs × padded length` per forward pass. The main memory knob. |
| `DEFAULT_PREFIX` | `""` | Prepended to every input (nomic models need `search_document: `) |
| `MAX_INPUTS` | `256` | Maximum inputs per request |
| `HOST` / `PORT` | `0.0.0.0` / `8015` | Listen address |

Only `.safetensors` weights are supported. Convert a `pytorch_model.bin` with:

```bash
python -c "from safetensors.torch import save_file; import torch; save_file({k: v.contiguous() for k, v in torch.load('pytorch_model.bin', weights_only=True).items()}, 'model.safetensors')"
```

## Building for the Jetson

candle's CUDA backend needs `nvcc`, which JetPack 7.2 hosts don't ship. Build inside the JetPack 6 container instead; it runs unchanged on JetPack 7 hosts. Run this on the Jetson itself:

```bash
docker build -f Dockerfile.jetson -t jetson-embed:cuda .
docker run --rm --runtime nvidia -p 8015:8015 \
  -v /path/to/model:/model -e MODEL_DIR=/model \
  jetson-embed:cuda
```

The build targets compute capability 8.7 (`CUDA_COMPUTE_CAP=87`, the Orin's GPU). Don't use the `sbsa` / `cu130` toolchains or wheels on an Orin: those target Jetson Thor and data-center Arm GPUs (sm_90 / sm_100), not sm_87.

For Apple Silicon GPU development: `cargo build --release --features metal`.

## How it works

```
POST /v1/embeddings
  │
  ├─ server.rs     parse + validate, acquire the inference permit
  │
  └─ embedder.rs   (on a blocking thread)
       tokenize ─► zero-token inputs answered directly (no model run)
         │
         └─ sort longest-first ─► batch by token budget
              │
              └─ per batch:  pad + attention mask
                               ─► backbone forward pass     backbones/*.rs
                               ─► pooling: cls|mean|last    pooling.rs
                               ─► post stages (e.g. int8)   from modules.json
                               ─► L2 normalize ─► optional Matryoshka truncation
```

`registry.rs` reads the model directory once at startup and decides the backbone, pooling, max length and post-pooling stages. None of it touches tensors.

## Project layout

```
src/
  main.rs            CLI: serve | embed | verify
  settings.rs        flags and environment variables
  registry.rs        model directory → backbone, pooling, pipeline stages, limits
  backbones/         Encoder trait; bert, xlm_roberta, nomic (candle), qwen3_bidir (this repo)
  pooling.rs         cls / mean / last-token pooling, int8 stage, L2 norm, Matryoshka
  embedder.rs        tokenize → sort → batch by token budget → encode → pool
  server.rs          axum HTTP API
  verify.rs          cosine check against reference vectors
baseline/            Python script that generates reference vectors with sentence-transformers
docs/                the learning path, chapters 00–11
scripts/             model download and candle-source helpers
Dockerfile.jetson    CUDA build inside the l4t-jetpack container
```

## Learning path

The `docs/` folder is a short course that follows the code, from what an embedding is through to writing a first CUDA kernel:

| | Chapter |
|---|---|
| 00 | [Start here](docs/00-start-here.md) |
| 01 | [Embeddings 101](docs/01-embeddings-101.md) |
| 02 | [The model registry](docs/02-model-registry.md) |
| 03 | [Transformer anatomy](docs/03-transformer-anatomy.md) |
| 04 | [Bidirectional attention](docs/04-bidirectional-attention.md) |
| 05 | [Pooling and normalization](docs/05-pooling-and-normalization.md) |
| 06 | [Tokenization and batching](docs/06-tokenization-and-batching.md) |
| 07 | [Serving](docs/07-serving.md) |
| 08 | [Memory on the Jetson](docs/08-jetson-memory.md) |
| 09 | [Verification](docs/09-verification.md) |
| 10 | [A first CUDA kernel](docs/10-first-cuda-kernel.md) |
| 11 | [Roadmap](docs/11-roadmap.md) |

## Roadmap

- [ ] Verify the Metal build (f16) on Apple Silicon
- [ ] Verify the CUDA build on the AGX Orin, and measure memory and throughput
- [ ] Verify the XLM-RoBERTa and NomicBERT backbones
- [ ] GTE backbone (`arctic-embed-m-v2.0`, `gte-multilingual`)
- [ ] Prometheus `/metrics`
- [ ] A custom CUDA kernel for masked mean pooling ([chapter 10](docs/10-first-cuda-kernel.md))

## License

[MIT](LICENSE)
