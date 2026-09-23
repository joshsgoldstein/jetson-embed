# 05 -- Pooling and normalisation (`src/pooling.rs`)

The backbone returns one vector per token: `(batch, seq, hidden)`. The API
promises one vector per text. Pooling closes that gap, and it is a
*checkpoint* property -- two BERTs with identical architecture can pool
differently, and swapping them silently corrupts every vector.

## The three modes

**CLS -- take token 0.** BERT-family models prepend a special `[CLS]` token
(`<s>` for RoBERTa). Attention lets that token gather information from the
whole sequence, and contrastive training teaches the model to put the summary
there. BGE, arctic, bge-m3 use it. Cheap: a slice.

**Mean -- average the real tokens.** Sum the token vectors where the mask is
1 and divide by the count. Robust, order-insensitive, and the natural choice
for models without a trained summary token. MiniLM, nomic, pplx use it. The
mask matters: padding tokens carry the embedding of `[PAD]` and would pull
the average toward it.

```rust
let summed = (hidden * mask3)?.sum(1)?;               // (batch, hidden)
let counts = mask.sum(1)?.clamp(1.0, ..)              // avoid /0 on an all-pad row
summed / counts
```

**Last token -- take position `len - 1`.** For decoders kept causal: only the
final token has attended to everything. Not used by the models in the
registry, implemented so the causal exercise in docs 04 has somewhere to go.

Two more exist in sentence-transformers (`max`, `weighted mean`); the
registry refuses them with a clear error rather than guess.

## Normalisation

`l2_normalize` divides each vector by its length so `|v| = 1`. After this,
dot product equals cosine similarity, which is what every vector store and
every `verify` check computes. All models in the registry are used with
`normalize_embeddings=True` in Python; the server always normalises.

Order matters: pool, then cast to f32, then normalise. Normalising in f16
loses precision in the sum of squares over 1024 dimensions.

## Matryoshka truncation (`truncate_dims`)

Some models (nomic v1.5, pplx) are trained with an extra loss on prefixes of
the vector, so the first 256 of 768 dimensions are *themselves* a usable
embedding. Truncate, then re-normalise (the prefix alone is no longer unit
length). OpenAI's API exposes this as `dimensions`; so does ours. On a
non-Matryoshka model the truncated vectors are meaningless, so check the
model card before using it.

## Why pooling lives outside the backbone

`Encoder::encode` returns the full token matrix. The embedder applies
`pool(spec.pooling, ...)` afterwards. This keeps the backbone
architecture-only and lets the registry drive pooling from `1_Pooling/config.json`.
It also means the token matrix is available for future features (token-level
embeddings, late interaction / ColBERT-style scoring) without touching the
models.

## Exercise

`mean_pool` is three tensor ops that each allocate a `(batch, seq, hidden)`
intermediate. Docs 10 replaces it with one fused CUDA kernel. Before that,
measure: time `mean_pool` alone on a `(8, 512, 1024)` f16 tensor on the
Jetson. That number is your baseline.
