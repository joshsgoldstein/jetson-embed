//! BGE-base/large, e5, MiniLM: the original encoder-only transformer.
//! candle ships the implementation; this file is only glue.
//!
//! What makes BERT "BERT" (docs/03-transformer-anatomy.md has the diagrams):
//!   * learned absolute position embeddings (so a hard 512-token limit)
//!   * post-LayerNorm blocks (norm after the residual add)
//!   * a segment ("token type") embedding, always 0 for single-text embedding
//!   * a [CLS] token at position 0 that the embedding models train as the summary

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

use super::Encoder;
use crate::registry::{read_json, ModelSpec};

pub struct Bert {
    model: BertModel,
    hidden: usize,
}

impl Bert {
    pub fn load(spec: &ModelSpec, vb: VarBuilder) -> Result<Self> {
        let cfg: Config = read_json(&spec.config_path).context("parsing BERT config")?;
        let hidden = cfg.hidden_size;
        // candle's loader tries the root prefix, then falls back to "bert." if the
        // checkpoint came from BertForMaskedLM instead of a bare BertModel.
        let model = BertModel::load(vb, &cfg).context("loading BERT weights")?;
        Ok(Self { model, hidden })
    }
}

impl Encoder for Bert {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(input_ids, token_type_ids, Some(attention_mask))?)
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
}
