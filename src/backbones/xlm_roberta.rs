//! bge-m3, arctic-embed-l-v2.0, multilingual-e5: RoBERTa with a 250k-piece
//! SentencePiece vocabulary. Structurally BERT, with two differences that
//! matter for embeddings:
//!   * positions start at pad_token_id + 1 (RoBERTa's quirk), and
//!   * there is no next-sentence objective, so token_type is always 0 -- but
//!     the embedding table for it still exists and must be added.
//! Note: bge-m3 on the Hub ships pytorch_model.bin, not safetensors. See the
//! conversion one-liner in registry.rs.

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};

use super::Encoder;
use crate::registry::{read_json, ModelSpec};

pub struct XlmRoberta {
    model: XLMRobertaModel,
    hidden: usize,
}

impl XlmRoberta {
    pub fn load(spec: &ModelSpec, vb: VarBuilder) -> Result<Self> {
        let cfg: Config = read_json(&spec.config_path).context("parsing XLM-RoBERTa config")?;
        let hidden = cfg.hidden_size;
        let model = XLMRobertaModel::new(&cfg, vb).context("loading XLM-RoBERTa weights")?;
        Ok(Self { model, hidden })
    }
}

impl Encoder for XlmRoberta {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(input_ids, attention_mask, token_type_ids, None, None, None)?)
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
}
