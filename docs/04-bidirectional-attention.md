# 04 -- Bidirectional attention: the one change that makes an embedding model

## Decoder vs encoder

Qwen3 was built to generate text. When it predicts token 7 it must not see
token 8 -- otherwise training would be trivial (copy the answer) and
inference would be impossible (token 8 does not exist yet). So a decoder adds
a **causal mask** to the attention scores: for query position `i`, every key
position `j > i` gets `-infinity` before the softmax, so its weight becomes 0.

```
        keys ->   t0   t1   t2   t3
query t0        [ 0   -inf -inf -inf ]
      t1        [ 0    0   -inf -inf ]     causal: lower triangle only
      t2        [ 0    0    0   -inf ]
      t3        [ 0    0    0    0   ]
```

An embedding model has no such constraint: the whole text exists up front,
and the *point* is for every token's representation to reflect its full
context. "bank" in "river bank" should know about "river" even though
"river" comes first. So an encoder uses a **bidirectional** mask: nothing is
hidden except padding.

```
        keys ->   t0   t1   t2   PAD
query t0        [ 0    0    0   -inf ]
      t1        [ 0    0    0   -inf ]     bidirectional: only padding masked
      t2        [ 0    0    0   -inf ]
      PAD       [ 0    0    0   -inf ]
```

## Where it lives in the code

`Qwen3Bidirectional::attention_bias_mask` builds the second matrix from the
`(batch, seq)` 0/1 attention mask:

```rust
let keep = attention_mask.to_dtype(self.dtype)?;          // 1 real, 0 pad
let bias = ((keep.ones_like()? - keep)? * -1e4)?;         // 0 real, -1e4 pad
Ok(bias.reshape((b, 1, 1, l))?)                           // broadcast over heads and queries
```

and `Attention::forward` step 5 adds it to the scores. That is all. Compare
with candle's `qwen3.rs::causal_mask`, which builds the first matrix with
`j <= i + offset`: the same tensor, a different predicate.

perplexity's `modeling.py` makes exactly this change to Hugging Face's
`Qwen3Model`: it sets `is_causal = False` on every attention layer and
supplies a mask function that returns `True` for every (query, key) pair. 91
lines. That, plus mean pooling and a fine-tuning run on paired data, is how a
600M-parameter chat model became an embedding model.

## Why `-1e4` and not `-inf`

`-inf + finite = -inf` and `exp(-inf) = 0`, so `-inf` works in f32. In f16
the largest finite value is 65504, and some kernels produce NaN when a whole
row is `-inf`. `-1e4` is representable in f16, is 10,000 below any real
score, and rounds to weight 0 after softmax. Same result, no NaN risk.

## What you lose by going bidirectional

The KV cache. A decoder caches k and v for past tokens so generating token
N+1 costs one token's work, not N+1. An encoder processes the whole input in
one pass and never generates, so there is nothing to cache. That is why
`qwen3_bidir.rs` has no `kv_cache` field and no `offset` argument -- and why
it is 260 lines instead of candle's 519.

## Exercise

Add a `--causal` debug flag that swaps in a causal mask, run `verify`, and
watch cosine fall. Then try `last_token` pooling with the causal mask: that
combination is how some decoder-based embedders (e.g. those trained from
Mistral) actually work, since only the last token has seen everything.
