# 03 -- Transformer anatomy (`src/backbones/qwen3_bidir.rs`)

Read this chapter with `qwen3_bidir.rs` open. It is 260 lines and it is the
*entire* model: nothing is hidden in a library call. When you can explain each
function in that file, you understand what every transformer since 2017 does.

## The shape of the data

Every tensor in the forward pass is one of these:

```
input_ids        (batch, seq)                integers, one per token
hidden           (batch, seq, hidden)        one vector per token; hidden = 1024 for pplx
q, k, v          (batch, heads, seq, head_dim)   the same vectors, split into heads
scores           (batch, heads, seq, seq)    every token vs every token -- THE quadratic term
```

`batch` is how many texts, `seq` the padded token length, `hidden` the model
width. Attention's `(seq, seq)` matrix is why memory grows with the square of
input length (docs 08).

## Walkthrough, top to bottom

### 1. Embedding table -- `embed_tokens`

A lookup: token id 42 -> row 42 of a `(vocab, hidden)` matrix. For pplx that
matrix is 151,936 x 1024 floats -- 620 MB in f32, the largest single tensor,
and a reason half precision matters on the Jetson.

### 2. The block, repeated N times -- `Layer::forward`

```
x  --> RMSNorm --> Attention --> (+) --> RMSNorm --> MLP --> (+) --> x'
 \_________________________________/  \____________________________/
          residual connection                residual connection
```

Two sub-layers, each wrapped the same way: normalise the input, transform it,
*add the result back to the input*. The residual add is what lets 28 layers
train: each layer only has to learn a correction, and gradients flow straight
through the `+`. Qwen3 (and Llama, and nomic) normalise *before* the sub-layer
("pre-norm"); the original BERT normalises *after* the add ("post-norm"), which
is the main structural difference between the two families.

### 3. RMSNorm -- `rms_norm(...)`

Divides each token's vector by its root-mean-square and multiplies by a
learned per-dimension weight. Keeps activations in a sane range so the next
matmul is well-conditioned. BERT's LayerNorm additionally subtracts the mean
and has a bias; RMSNorm dropped both and nobody noticed a quality loss.

### 4. Attention -- `Attention::forward`

This is the one to slow down for. Six numbered steps in the code:

1. **Project.** Three matrices turn each token vector into a *query* ("what am
   I looking for"), a *key* ("what do I contain") and a *value* ("what I will
   contribute if picked"). Then reshape so each of 16 heads gets its own
   128-dim slice and works independently.
2. **Per-head norm.** Qwen3 RMS-normalises q and k. A stability trick; no
   conceptual weight.
3. **RoPE.** Rotate q and k by a position-dependent angle so that q.k depends
   on how far apart the tokens are. Explained below.
4. **GQA repeat.** pplx has 16 query heads but only 8 key/value heads
   (grouped-query attention: halves the k/v projection cost). `repeat_kv`
   duplicates each k/v head so the shapes line up.
5. **Scores and softmax.** `q @ k^T / sqrt(d)` gives a `(seq, seq)` grid of
   "how much does token i care about token j". Add the mask (docs 04),
   softmax each row into weights that sum to 1, multiply by `v`: every token
   becomes a weighted blend of the values it attended to.
6. **Merge heads** back to `(batch, seq, hidden)` and project with `o_proj`.

The `1/sqrt(d)` keeps the dot products from growing with head size so softmax
does not saturate. It is the "scaled" in "scaled dot-product attention".

### 5. The MLP -- `Mlp::forward`

Attention *moves* information between tokens; the MLP *transforms* each token
on its own. Qwen3 uses SwiGLU: `down( silu(gate(x)) * up(x) )`. The gate
branch, squashed through SiLU, decides how much of each `up` feature passes.
Three matrices of size `hidden x 3072` per layer -- in most transformers the
MLP holds two thirds of the parameters.

BERT's MLP is the older two-matrix form with GELU: `down( gelu(up(x)) )`.

### 6. RoPE -- `RotaryEmbedding`

A transformer is a set function: shuffle the tokens and attention gives the
same answer. Position has to be injected somewhere.

* **BERT**: add a learned vector for "position 17" to the token embedding.
  Simple, but the table has a fixed size -- 512 rows for BERT -- and the model
  never sees position 513.
* **RoPE** (Qwen, Llama, nomic): rotate the q and k vectors, two dimensions at
  a time, by an angle proportional to the position. The dot product of a
  query at position `m` and a key at position `n` then depends only on
  `m - n`. Relative position falls out of the geometry, and the model
  extrapolates somewhat beyond its training length.

`RotaryEmbedding::new` precomputes `sin` and `cos` tables for every position
up to `max_position_embeddings`; `apply` slices the first `seq` rows and
calls candle's fused `rope` kernel. The frequencies run from fast (first
dimension pairs, sensitive to nearby positions) to slow (last pairs, coarse).

## How the four backbones differ

| | BERT (bge, MiniLM) | XLM-R (bge-m3, arctic-l) | NomicBERT | Qwen3 (pplx) |
|---|---|---|---|---|
| Positions | learned table | learned table (offset by pad id) | RoPE | RoPE |
| Norm | LayerNorm, post | LayerNorm, post | LayerNorm, pre | RMSNorm, pre |
| MLP | GELU, 2 mats | GELU, 2 mats | SwiGLU, fused fc11/fc12 | SwiGLU |
| Biases | yes | yes | no | no |
| Segment embedding | yes | yes (unused) | yes | no |
| Attention | full (MHA) | full (MHA) | full (MHA) | GQA 16/8 + q/k norm |
| Max length | 512 | 8192 | 8192 | 32768 |

The candle-backed three are glue files; you can read candle's `bert.rs`,
`xlm_roberta.rs` and `nomic_bert.rs` (fetched into the `references/`
folder by `scripts/fetch-candle-sources.sh`) and map every line to a row of
this table.

## Exercises

1. Print the attention `probs` for one head on a short sentence and check each
   row sums to 1. Then set the mask to causal (docs 04) and watch the upper
   triangle go to zero.
2. Remove `q_norm`/`k_norm` and run `verify`. How far does cosine drop? (It
   should collapse -- the weights were trained with them.)
3. Replace SiLU with GELU in the MLP. Same question.
