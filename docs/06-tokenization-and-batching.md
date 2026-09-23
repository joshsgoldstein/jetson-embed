# 06 -- Tokenization and batching (`src/embedder.rs`)

## Tokenization

A tokenizer maps text to integer ids from a fixed vocabulary, using
sub-word pieces so that unknown words decompose into known parts
(`"jetson"` -> `["jet", "##son"]`). `tokenizer.json` is a complete,
self-describing spec: the vocabulary, the merge rules, the normaliser
(lower-casing for uncased BERTs), the pre-tokeniser (how to split on
whitespace and punctuation), and the *post-processor* that adds special tokens.
The `tokenizers` crate is the same Rust code Python's `transformers` calls
underneath, so byte-for-byte identical ids are the expected outcome, not a
hope.

`encode_batch(inputs, true)` -- the `true` is `add_special_tokens`. It is what
turns `"hello"` into `[CLS] hello [SEP]` for BERT or `<s> hello </s>` for
RoBERTa, and does nothing for Qwen, whose tokenizer has no template. Get this
wrong and CLS pooling reads a real word instead of the summary token.

**Truncation** is configured once on the tokenizer, at `max_seq_len`
(docs 02). This silently drops the tail of long inputs, exactly as
sentence-transformers does. A future improvement is the chunk-and-merge
strategy the GLiNER server uses; for embeddings the merge is easy (average
the chunk vectors) but changes the semantics, so it is deliberately not on
by default.

**Empty strings.** BERT tokenizers still emit `[CLS] [SEP]`; Qwen's emits
nothing, and an empty tensor would crash the model. The embedder substitutes
a single pad token so the request succeeds with a (meaningless but finite)
vector, matching what Python returns.

## Batching

One forward pass handles a `(batch, seq)` rectangle, so every text in a batch
is padded to the longest one. Two costs follow:

* **Wasted compute** on pad positions, proportional to the padding.
* **Peak memory** `~ batch x seq^2` from attention scores, set by the
  *longest* member regardless of the others.

The embedder does two things about it:

1. **Sort by length, longest first.** Neighbours in the sorted order have
   similar lengths, so padding is minimal. Output is un-sorted at the end so
   callers see their original order.
2. **Cut batches by a token budget, not a count.** `MAX_BATCH_TOKENS`
   bounds `count x longest_len`. Sixty-four 20-token sentences fit in one
   pass; two 8,000-token documents go one at a time. This is the single knob
   that bounds peak memory on the Jetson (docs 08), and it is why the server
   never needs a "max batch size".

Padding is done by hand rather than by the tokenizer's padding option so the
same code path builds `input_ids`, `attention_mask` and `token_type_ids`
together and the pad id is unambiguous.

## The attention mask is used twice

Once inside the backbone, to stop real tokens attending to padding
(docs 04), and once in the pooler, to exclude padding from the mean
(docs 05). Both get the same `(batch, seq)` 0/1 tensor. Forgetting either
one produces vectors that are subtly wrong only for texts that got padded --
i.e. only in batches, never in single-text tests. `verify` embeds its
reference texts as one batch of mixed lengths for exactly this reason.

## Exercise

Log `(batch_count, longest, padded_tokens, real_tokens)` per forward pass and
embed a 200-line file. Compute the padding ratio with and without sorting.
