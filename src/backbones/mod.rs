//! One trait, four implementations. Each backbone turns token ids into
//! per-token hidden states of shape (batch, seq, hidden). Pooling happens
//! outside, in `pooling.rs`, so a backbone never needs to know how its output
//! will be summarised.
//!
//! Three of the four wrap models that ship inside candle-transformers; the
//! fourth (Qwen3 bidirectional) is written out in full in this crate because
//! candle's Qwen3 is a causal decoder and its internals are private. Reading
//! qwen3_bidir.rs top to bottom is the fastest way to learn what a
//! transformer block actually does -- see docs/03-transformer-anatomy.md.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::registry::{Backbone, ModelSpec};

pub mod bert;
pub mod nomic;
pub mod qwen3_bidir;
pub mod xlm_roberta;

pub trait Encoder: Send + Sync {
    /// input_ids, attention_mask, token_type_ids: all (batch, seq), u32.
    /// Returns hidden states (batch, seq, hidden) in the model dtype.
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, token_type_ids: &Tensor) -> Result<Tensor>;
    fn hidden_size(&self) -> usize;
}

/// Build the right encoder for a spec, with the weights memory-mapped from
/// safetensors straight into the requested dtype on the requested device.
pub fn load(spec: &ModelSpec, device: &Device, dtype: DType) -> Result<Box<dyn Encoder>> {
    // SAFETY (candle's contract): the file must not be modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&spec.weights_path], dtype, device)? };
    Ok(match spec.backbone {
        Backbone::Bert => Box::new(bert::Bert::load(spec, vb)?),
        Backbone::XlmRoberta => Box::new(xlm_roberta::XlmRoberta::load(spec, vb)?),
        Backbone::NomicBert => Box::new(nomic::Nomic::load(spec, vb)?),
        Backbone::Qwen3Bidirectional => Box::new(qwen3_bidir::Qwen3Bidirectional::load(spec, vb)?),
    })
}
