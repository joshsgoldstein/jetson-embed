# 07 -- Serving (`src/server.rs`)

## The API

`POST /v1/embeddings` accepts OpenAI's request shape, so existing clients
(the `openai` Python package, LangChain, LlamaIndex) work by changing the
base URL:

```json
{"input": "one text"  |  ["many", "texts"],
 "model": "optional; must match the served model",
 "dimensions": 256,                      // Matryoshka truncation, optional
 "prefix": "search_query: "}             // non-standard: nomic task prefix
```

and returns

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [ ... ]}],
 "model": "bge-base-en-v1.5",
 "usage": {"prompt_tokens": 12, "total_tokens": 12}}
```

`GET /v1/models` lists the one model; `GET /health` reports backbone,
pooling, device, max length and whether a request is in flight.

## One request at a time, on purpose

```rust
let _permit = st.gate.acquire().await?;                 // Semaphore::new(1)
tokio::task::spawn_blocking(move || embedder.embed(...))
```

The GPU is a single resource. Two forward passes at once do not finish
sooner -- they interleave on the same cores -- but they *do* both allocate
their attention scratch at the same time, doubling peak memory. On a
unified-memory Jetson that is RAM stolen from every other service, and it is
exactly how the GLiNER server ended up holding 12 GB (docs 08). So requests
queue in the semaphore and run one by one; throughput comes from batching
*within* a request, not from concurrency across them.

`spawn_blocking` moves the CPU/GPU-bound forward pass off the async
executor's threads so health checks and new connections stay responsive
while a big batch runs.

## Errors an agent can act on

Every error is `{"error": {"code": "...", "message": "..."}}` with a stable
code (`TOO_MANY_INPUTS`, `MODEL_NOT_FOUND`, `INFERENCE_FAILED`, ...) and a
message that says what to do. This mirrors the structured 413s in the
GLiNER API: a calling agent should be able to split its request and retry
from the response alone, without reading documentation.

## What is deliberately missing

* **Multiple models per process.** One binary, one model, one port. Run two
  processes for two models; each declares its own memory budget (docs 08).
* **Auth and TLS.** This sits behind Tailscale on a LAN. Put a reverse proxy
  in front if that changes.
* **Streaming.** Embeddings are not streamed by anyone's API.
* **`encoding_format: base64`.** Rejected with `UNSUPPORTED_ENCODING`.

## Exercise

Add request timing to the log line (queue wait vs inference, the way the
GLiNER server does) and a `/metrics` endpoint that exposes the last N. Then
send 50 concurrent requests and watch the queue wait grow while inference
time stays flat -- that is the semaphore doing its job.
