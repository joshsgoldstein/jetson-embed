//! Model discovery: given a directory downloaded from the Hub, work out which
//! backbone to build, how to pool, how long inputs may be, and which prefix
//! the model expects. Nothing here touches tensors; it only reads JSON.
//!
//! The Hub gives us three sources of truth, and we read all three:
//!   config.json                    -> architecture + hyper-parameters
//!   1_Pooling/config.json          -> pooling mode (sentence-transformers)
//!   sentence_bert_config.json      -> max_seq_length the authors trained for
//! See docs/02-model-registry.md.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::pooling::{Pooling, PoolingConfig};

/// The transformer families this server knows how to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backbone {
    /// Classic BERT: absolute positions, GELU, post-LayerNorm. BGE v1.5, e5, MiniLM.
    Bert,
    /// RoBERTa with a 250k multilingual vocab. bge-m3, arctic-embed-l-v2.0, e5-multilingual.
    XlmRoberta,
    /// BERT re-architected with RoPE + SwiGLU, no biases. nomic-embed-text v1 / v1.5.
    NomicBert,
    /// Qwen3 decoder run with bidirectional attention (no causal mask). pplx-embed.
    Qwen3Bidirectional,
}

impl Backbone {
    /// Map Hugging Face `model_type` strings to a backbone.
    pub fn from_model_type(model_type: &str) -> Result<Self> {
        Ok(match model_type {
            "bert" => Backbone::Bert,
            "xlm-roberta" | "roberta" => Backbone::XlmRoberta,
            "nomic_bert" => Backbone::NomicBert,
            "bidirectional_pplx_qwen3" | "qwen3" => Backbone::Qwen3Bidirectional,
            "gte" | "new" => bail!(
                "model_type '{model_type}' (Alibaba GTE / arctic-embed-m-v2.0) is not in candle yet. \
                 It is a RoPE BERT close to nomic_bert; porting it is a good exercise -- see docs/11-roadmap.md"
            ),
            other => bail!("unsupported model_type '{other}'; supported: bert, xlm-roberta, nomic_bert, bidirectional_pplx_qwen3"),
        })
    }

    /// Some backbones use token_type_ids (segment embeddings); decoders do not.
    pub fn uses_token_type_ids(self) -> bool {
        matches!(self, Backbone::Bert | Backbone::XlmRoberta | Backbone::NomicBert)
    }
}

/// Just enough of config.json to route. The backbone parses the full config itself.
#[derive(Debug, Deserialize)]
struct RoutingConfig {
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    n_positions: Option<usize>, // nomic_bert's name for the same thing
}

#[derive(Debug, Deserialize, Default)]
struct SentenceBertConfig {
    max_seq_length: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    pad_token: Option<serde_json::Value>, // string, or {"content": "..."} in older files
}

/// Everything the embedder needs to know about one model directory.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub dir: PathBuf,
    pub backbone: Backbone,
    pub pooling: Pooling,
    /// Longest input in tokens. Inputs are truncated to this.
    pub max_seq_len: usize,
    /// The string the tokenizer pads with ("[PAD]", "<pad>", "<|endoftext|>").
    pub pad_token: Option<String>,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_path: PathBuf,
}

impl ModelSpec {
    pub fn discover(dir: &Path, name_override: Option<String>, max_seq_override: Option<usize>) -> Result<Self> {
        let config_path = dir.join("config.json");
        let cfg: RoutingConfig = read_json(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let model_type = cfg
            .model_type
            .clone()
            .or_else(|| cfg.architectures.first().map(|a| a.to_lowercase()))
            .context("config.json has neither model_type nor architectures")?;
        let backbone = Backbone::from_model_type(&model_type)?;

        // Pooling: the sentence-transformers file if present, else a per-family default
        // with a loud warning, because guessing wrong silently produces garbage vectors.
        let pooling_path = dir.join("1_Pooling").join("config.json");
        let pooling = if pooling_path.exists() {
            read_json::<PoolingConfig>(&pooling_path)?.to_pooling()?
        } else {
            let guess = match backbone {
                Backbone::Bert | Backbone::XlmRoberta => Pooling::Cls,
                Backbone::NomicBert | Backbone::Qwen3Bidirectional => Pooling::Mean,
            };
            tracing::warn!("no 1_Pooling/config.json; guessing {guess:?} pooling for {backbone:?}");
            guess
        };

        // Max length: what the authors trained at beats what the architecture allows.
        let sbert: SentenceBertConfig = read_json(&dir.join("sentence_bert_config.json")).unwrap_or_default();
        let arch_max = cfg.max_position_embeddings.or(cfg.n_positions).unwrap_or(512);
        let trained_max = sbert.max_seq_length.unwrap_or(arch_max).min(arch_max);
        let max_seq_len = max_seq_override.unwrap_or(trained_max).min(arch_max);

        let tok_cfg: TokenizerConfig = read_json(&dir.join("tokenizer_config.json")).unwrap_or_default();
        let pad_token = tok_cfg.pad_token.and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str().map(String::from)),
            _ => None,
        });

        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            bail!(
                "{} not found. This loader reads safetensors only. Convert a pytorch_model.bin with:\n  \
                 python -c \"from safetensors.torch import save_file; import torch; \
                 save_file({{k: v.contiguous() for k, v in torch.load('pytorch_model.bin').items()}}, 'model.safetensors')\"",
                weights_path.display()
            );
        }
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            bail!("{} not found (slow tokenizers without tokenizer.json are not supported)", tokenizer_path.display());
        }

        let name = name_override.unwrap_or_else(|| {
            dir.file_name().map(|s| s.to_string_lossy().replace("__", "/")).unwrap_or_else(|| "model".into())
        });

        Ok(Self { name, dir: dir.to_path_buf(), backbone, pooling, max_seq_len, pad_token, config_path, tokenizer_path, weights_path })
    }
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}
