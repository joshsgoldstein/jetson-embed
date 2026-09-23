# 01 -- Embeddings 101

## The idea

An embedding model is a function from text to a point in a high-dimensional
space, trained so that *meaning* becomes *geometry*: paraphrases land near
each other, unrelated sentences land far apart. "Near" is measured by the
angle between the vectors, not the distance -- which is why every model here
normalises its output to length 1 (docs 05). Once vectors have length 1, the
dot product *is* the cosine of the angle:

```
cos(a, b) = (a . b) / (|a| |b|)   -->  with |a| = |b| = 1, just  a . b
```

That is the number a vector database ranks by. A score of 1.0 is the same
direction, 0 is unrelated, negative is (rarely) opposed.

## Dimensions

| Model | dim | Why it matters |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | Small index, fast search, decent English quality |
| bge-base-en-v1.5 | 768 | The BERT-base default |
| nomic-embed-text-v1.5 | 768, truncatable to 64 | Matryoshka: the first N dims are themselves a good embedding |
| bge-m3 / arctic-l-v2 | 1024 | Larger, multilingual |
| pplx-embed-v1-0.6B | 1024 | Qwen3-based, 32k context |

The vector's length is fixed by the model; storage and search cost scale with
it. Matryoshka-trained models (docs 05) let you pick a smaller prefix at
query time, which this server exposes as OpenAI's `dimensions` field.

## Symmetric vs asymmetric

Some models embed a *query* and a *document* the same way (symmetric). Others
were trained with a task prefix so the two sides land in slightly different
places (asymmetric) -- nomic wants `search_query: ` on questions and
`search_document: ` on passages, and its quality drops noticeably without
them. BGE v1.5 has an optional query instruction. pplx and MiniLM want nothing.
This is a training-time decision, so the server has to know it: that is what
`DEFAULT_PREFIX` and the per-request `prefix` field are for (docs 02, 07).

## Where the cost goes

Embedding one text means running a transformer over its tokens once. There is
no generation loop, no KV cache, no sampling -- a forward pass and a pool.
Which is why an embedding server is the ideal first Rust/candle project: the
whole model runs in `encode()` and returns, and everything is deterministic.

The cost of that forward pass is dominated by attention, which compares every
token with every other token: doubling the text length quadruples the work
and, on a unified-memory Jetson, quadruples the scratch memory (docs 08).
