# 11 -- Roadmap

Ordered by value to the lab, not by difficulty.

## Ship

1. **Build on the Jetson with `--features cuda`** via `Dockerfile.jetson`,
   run `verify` in f16 for each model you intend to serve, record peak memory
   for the worst realistic batch, and add the service to
   `expected-services.conf` with its budget. Docs 08.
2. **Pick the default model.** For the harness's retrieval, bge-base
   (768-dim, CLS, fast) or nomic v1.5 (Matryoshka, prefixes) are the
   pragmatic choices; pplx is the strongest but 5x the compute per token.

## Extend the registry

3. **GTE (`model_type: gte`)** -- unlocks Snowflake arctic-embed-m-v2.0 and
   Alibaba's gte-multilingual-base. It is a RoPE BERT with a fused qkv,
   structurally between `nomic_bert` and `qwen3_bidir`. Port from the
   model's `modeling_hf_alibaba_nlp_gte.py`; start by copying
   `qwen3_bidir.rs` and swapping in learned LayerNorm, biases and the GTE
   weight names. This is the best "second backbone" exercise in the repo.
4. **nomic-embed-text-v2-moe** -- adds a mixture-of-experts MLP (8 experts,
   top-2). candle has MoE routing code in `qwen3_moe.rs` to crib from.
5. **ModernBERT** (candle has it) -- for the newer BERT-family embedders
   built on it.

## Faster

6. **Flash attention** -- candle's `flash-attn` feature (needs the
   `candle-flash-attn` build, ~20 min on the Orin) removes the `(l, l)`
   scores tensor entirely. Attention memory becomes linear in `seq`, which
   changes the `MAX_BATCH_TOKENS` math in docs 08 completely. Bidirectional
   flash attention is `causal = false`.
7. **Quantised weights** -- candle's GGUF path (`quantized_qwen3.rs`) would
   put pplx at ~0.4 GB in Q4. Quality check with `verify` at a relaxed
   threshold and an MTEB subset.
8. **Your kernels** -- docs 10. Mean pool, then RMSNorm, then fused
   mask+softmax.

## Operate

9. **Chunk-and-merge for long documents** -- average chunk vectors, or return
   one vector per chunk (late chunking). Same shape as the GLiNER change.
10. **Batch coalescing across requests** -- collect requests for a few ms and
    run one forward pass. Only worth it once several agents hit the endpoint
    at once.
11. **Add the service to the health check** with a deep probe that embeds a
    fixed string and checks cosine against a stored vector -- the embedding
    equivalent of GLiNER's `/health/deep`.
