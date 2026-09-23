# 09 -- Verification (`src/verify.rs`, `baseline/embed_baseline.py`)

A port that compiles and returns unit vectors can still be completely wrong.
The only proof is agreement with the original implementation on the same
inputs. This chapter is the test protocol.

## The protocol

1. `baseline/embed_baseline.py --model X` runs the model through
   sentence-transformers -- the reference everyone benchmarks against -- on a
   fixed list of texts, and writes the vectors to `baseline/reference/X.json`.
   It records the prefix used, the dimension and `max_seq_length`.
2. `jetson-embed verify --reference that.json` embeds the same texts with
   the same prefix and prints, per text, the cosine similarity and the
   largest absolute difference.
3. Threshold: **cosine >= 0.999** on every text. f16 vs f32 rounding over
   1,024 dimensions costs about 1e-4; anything below 0.999 is a logic error.

Run it on the Mac in f32 first (exact arithmetic, no GPU questions), then on
the Jetson in f16 (the deployed configuration).

## The fixed texts, and what each one catches

| Text | Catches |
|---|---|
| A plain English sentence | Everything basic |
| A sentence with digits and a proper noun | Tokenizer normalisation, casing |
| A non-English sentence | Byte-level vs word-piece handling, `unk` tokens |
| `"a"` | Single-token input; CLS-vs-mean confusion is obvious here |
| `""` | Empty input: special-token-only sequences, empty tensors |
| 300 repeated words | Truncation at `max_seq_length`; must truncate *identically* |

They are embedded as one batch of very different lengths, so padding and the
attention mask are exercised on every run (docs 06).

## Reading a failure

| Symptom | Usual cause |
|---|---|
| All cosines ~0.3-0.7, roughly equal | Wrong pooling (CLS vs mean) |
| Short texts fine, long text off | Truncation length differs; check `max_seq_length` vs `max_position_embeddings` |
| Single-text fine, batch off | Attention mask or pooling mask not applied -- padding is leaking in |
| Everything ~0.95-0.99 | Missing a norm, wrong `eps`, wrong activation (GELU vs SiLU) |
| Nomic ~0.8 | Prefix missing on one side |
| pplx exactly right on short, drifts on long | RoPE `theta` or `head_dim` wrong -- error grows with position |
| Cosine 1.0 but `max|diff|` ~1e-2 | f16 on one side, f32 on the other; fine |
| DIM MISMATCH | `dimensions` truncation applied on one side only |

## What this does not prove

That the model is *good* -- only that the Rust port computes what the Python
one computes. Quality is the model authors' problem; MTEB scores are in the
model cards. It also does not prove the Jetson build is using the GPU:
`/health` reports the device, and `tegrastats` should show GR3D climbing
during a request. A model quietly running on the ARM cores passes `verify`
and is ten times too slow.

## Exercise

Add a `--dtype f16` run on the Mac (Metal) to the CI script and record the
max cosine drop per model. That number is your half-precision error budget
and belongs in the README.
