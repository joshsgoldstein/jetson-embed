//! nomic-embed-text v1 / v1.5. A BERT that borrowed the good ideas from
//! Llama: rotary position embeddings instead of a learned table (so 8192
//! tokens without retraining), SwiGLU instead of GELU, and no bias terms.
//! The three "attn.Wqkv" / "mlp.fc11, fc12, fc2" weight names you will see in
//! the checkpoint are the fused query-key-value projection and the two halves
//! of SwiGLU. docs/03-transformer-anatomy.md walks through both.
//!
//! Nomic models REQUIRE a task prefix on every input or quality collapses:
//!   "search_document: " for things you index, "search_query: " for queries,
//!   "classification: ", "clustering: ". Set DEFAULT_PREFIX or pass "prefix".

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel};

use super::Encoder;
use crate::registry::{read_json, ModelSpec};

pub struct Nomic {
    model: NomicBertModel,
    hidden: usize,
}

impl Nomic {
    pub fn load(spec: &ModelSpec, vb: VarBuilder) -> Result<Self> {
        let cfg: Config = read_json(&spec.config_path).context("parsing nomic_bert config")?;
        let hidden = cfg.n_embd;
        let model = NomicBertModel::load(vb, &cfg).context("loading nomic_bert weights")?;
        Ok(Self { model, hidden })
    }
}

impl Encoder for Nomic {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        Ok(self.model.forward(input_ids, Some(token_type_ids), Some(attention_mask))?)
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
}
