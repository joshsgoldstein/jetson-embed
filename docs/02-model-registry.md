# 02 -- The model registry (`src/registry.rs`)

## The problem

"Support nomic, snowflake, BGE, etc." sounds like a list of models. It is
really a list of *architectures* plus a list of *training-time decisions*.
The server has to reconstruct both from the files a Hub repository ships.

| Model | Architecture (`config.json` `model_type`) | Pooling | Prefix needed | Weights |
|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | `bert` | CLS | optional query instruction | safetensors |
| sentence-transformers/all-MiniLM-L6-v2 | `bert` | mean | none | safetensors |
| Snowflake/snowflake-arctic-embed-s / -m (v1) | `bert` | CLS | query prefix | safetensors |
| nomic-ai/nomic-embed-text-v1.5 | `nomic_bert` | mean | `search_document: ` etc. | safetensors |
| BAAI/bge-m3 | `xlm-roberta` | CLS | none | pytorch_model.bin (convert) |
| Snowflake/snowflake-arctic-embed-l-v2.0 | `xlm-roberta` | CLS | query prefix | safetensors |
| perplexity-ai/pplx-embed-v1-0.6b | `bidirectional_pplx_qwen3` | mean | none | safetensors |
| Snowflake/snowflake-arctic-embed-m-v2.0 | `gte` | CLS | query prefix | **not yet: GTE port needed** |
| nomic-ai/nomic-embed-text-v2-moe | `nomic_bert` (MoE) | mean | prefixes | **not yet: MoE layers** |

Four architectures cover seven of the nine. The registry's job is to read the
JSON and say which of the four to build, and how to pool.

## The three files it reads

**`config.json`** -- the architecture and its hyper-parameters. The registry
reads only `model_type` to route; the chosen backbone then parses the full
file into its own strongly-typed `Config` (each family names things
differently: BERT says `hidden_size`, nomic says `n_embd`).

**`1_Pooling/config.json`** -- written by sentence-transformers, and the
single most important file for correctness. It holds flags like
`pooling_mode_cls_token` / `pooling_mode_mean_tokens`. Use the wrong one and
every vector is still unit length, still looks plausible, and is wrong.
If the file is missing the registry guesses from the family and logs a
warning; treat that warning as a bug to fix, not noise.

**`sentence_bert_config.json`** -- `max_seq_length`, the length the authors
trained at. Often shorter than what the architecture allows (nomic's
architecture allows 8192; some BERTs allow 512 but were trained at 256). We
take the smaller of the two, and let `MAX_SEQ_LEN` cap it further on a
memory-constrained box.

Plus **`tokenizer_config.json`** for the pad token's *string*, which the
tokenizer turns into an id. Every family pads with something different
(`[PAD]`, `<pad>`, `<|endoftext|>`).

## Why an enum and a trait, not a config string

`Backbone` is a Rust enum, so adding GTE means the compiler lists every place
that must learn about it. `Encoder` is a trait with one method, so the
embedder never knows which family it is running: it hands over
`(input_ids, attention_mask, token_type_ids)` and gets `(batch, seq, hidden)`
back. Pooling is applied outside the backbone for the same reason -- it is a
property of the *checkpoint*, not the architecture (two BERTs above pool
differently).

## Weights: memory-mapped safetensors

`VarBuilder::from_mmaped_safetensors` maps the file and pulls tensors by name
(`layers.0.self_attn.q_proj.weight`), converting to the requested dtype on the
way in. Two consequences you will hit:

* Tensor names must match what the Rust model asks for. pplx saves a bare
  `Qwen3Model` (`embed_tokens.weight`); a checkpoint saved from
  `Qwen3ForCausalLM` nests everything under `model.`. `qwen3_bidir.rs` checks
  which layout it has. candle's BERT does the same dance for a `bert.` prefix.
* Only `.safetensors` is read. bge-m3 ships `pytorch_model.bin`; the error
  message in `registry.rs` includes the one-line conversion.

## Exercise

Add a `--list-models DIR` command that walks a directory of downloaded
models and prints this table from their JSON files, flagging any without
`1_Pooling/`. You will touch only `registry.rs` and `main.rs`.
