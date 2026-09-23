//! Turning per-token hidden states into one vector per input.
//!
//! A transformer encoder gives back a matrix: one row per token. An embedding
//! is one row per text. "Pooling" is how you collapse the matrix, and every
//! model family made a different choice at training time -- so you must use
//! the same one at inference or the vectors are meaningless. See
//! docs/05-pooling-and-normalization.md for why each family chose what it did.

use candle_core::{DType, Result, Tensor, D};
use serde::Deserialize;

/// Which token(s) represent the whole text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The first token. BERT-style models (BGE, arctic, bge-m3) train a special
    /// [CLS] token to summarise the sequence.
    Cls,
    /// Average of all real tokens (padding excluded). Nomic, pplx, most
    /// sentence-transformers models. Robust and order-insensitive.
    Mean,
    /// The final real token. Decoder-derived embedders sometimes use this
    /// because in a causal model only the last token has seen everything.
    LastToken,
}

/// sentence-transformers writes its pooling choice to `1_Pooling/config.json`.
/// We read that file so the same code serves any model from the Hub.
#[derive(Debug, Deserialize, Default)]
pub struct PoolingConfig {
    #[serde(default)]
    pub pooling_mode_cls_token: bool,
    #[serde(default)]
    pub pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pub pooling_mode_lasttoken: bool,
    #[serde(default)]
    pub pooling_mode_max_tokens: bool,
    #[serde(default)]
    pub pooling_mode_weightedmean_tokens: bool,
}

impl PoolingConfig {
    pub fn to_pooling(&self) -> anyhow::Result<Pooling> {
        match (
            self.pooling_mode_cls_token,
            self.pooling_mode_mean_tokens,
            self.pooling_mode_lasttoken,
        ) {
            (true, false, false) => Ok(Pooling::Cls),
            (false, true, false) => Ok(Pooling::Mean),
            (false, false, true) => Ok(Pooling::LastToken),
            _ if self.pooling_mode_max_tokens || self.pooling_mode_weightedmean_tokens => {
                anyhow::bail!("max / weighted-mean pooling not implemented yet")
            }
            _ => anyhow::bail!("1_Pooling/config.json selects no supported pooling mode"),
        }
    }
}

/// hidden: (batch, seq, dim). mask: (batch, seq) with 1 for real tokens.
/// Returns (batch, dim).
pub fn pool(pooling: Pooling, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    match pooling {
        Pooling::Cls => hidden.i_cls(),
        Pooling::Mean => mean_pool(hidden, mask),
        Pooling::LastToken => last_token_pool(hidden, mask),
    }
}

trait ClsIndex {
    fn i_cls(&self) -> Result<Tensor>;
}
impl ClsIndex for Tensor {
    /// hidden[:, 0, :]
    fn i_cls(&self) -> Result<Tensor> {
        self.narrow(1, 0, 1)?.squeeze(1)
    }
}

/// Masked average: sum(hidden * mask) / sum(mask). The mask keeps padding
/// tokens from dragging the average toward whatever the pad embedding is.
/// This is the candidate for your first custom CUDA kernel -- see
/// docs/10-first-cuda-kernel.md.
pub fn mean_pool(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, l, d) = hidden.dims3()?;
    let mask = mask.to_dtype(hidden.dtype())?;
    let mask3 = mask.unsqueeze(2)?.broadcast_as((b, l, d))?;
    let summed = (hidden * &mask3)?.sum(1)?; // (b, d)
    let counts = mask
        .sum(1)?
        .clamp(1.0, f64::MAX)? // an all-padding row would divide by zero
        .unsqueeze(1)?
        .broadcast_as((b, d))?;
    summed / counts
}

/// Pick hidden[i, len_i - 1, :] for each row, where len_i = mask.sum().
pub fn last_token_pool(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, _l, d) = hidden.dims3()?;
    let lengths = mask.to_dtype(DType::F32)?.sum(1)?; // (b)
    let idx = (lengths - 1.0)?.clamp(0.0, f64::MAX)?.to_dtype(DType::U32)?; // (b)
    let idx = idx.unsqueeze(1)?.unsqueeze(2)?.broadcast_as((b, 1, d))?;
    hidden.gather(&idx, 1)?.squeeze(1)
}

/// Scale each row to unit length so that dot product == cosine similarity.
/// Every model here ships with `normalize_embeddings=True` semantics.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
    x.broadcast_div(&norm)
}

/// Matryoshka models (nomic v1.5, pplx) are trained so the first `dim` values
/// are themselves a good embedding. Truncate, then re-normalise.
pub fn truncate_dims(x: &Tensor, dim: usize) -> Result<Tensor> {
    let full = x.dim(D::Minus1)?;
    if dim >= full {
        return Ok(x.clone());
    }
    l2_normalize(&x.narrow(D::Minus1, 0, dim)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn mean_pool_ignores_padding() -> Result<()> {
        let dev = Device::Cpu;
        // batch=1, seq=3, dim=2 ; third token is padding with garbage values
        let h = Tensor::new(&[[[1f32, 2.], [3., 4.], [100., 100.]]], &dev)?;
        let m = Tensor::new(&[[1u32, 1, 0]], &dev)?;
        let out = mean_pool(&h, &m)?.to_vec2::<f32>()?;
        assert_eq!(out, vec![vec![2.0, 3.0]]);
        Ok(())
    }

    #[test]
    fn last_token_uses_mask_length() -> Result<()> {
        let dev = Device::Cpu;
        let h = Tensor::new(&[[[1f32, 1.], [2., 2.], [9., 9.]]], &dev)?;
        let m = Tensor::new(&[[1u32, 1, 0]], &dev)?;
        let out = last_token_pool(&h, &m)?.to_vec2::<f32>()?;
        assert_eq!(out, vec![vec![2.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn normalize_gives_unit_length() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[3f32, 4.]], &dev)?;
        let out = l2_normalize(&x)?.to_vec2::<f32>()?;
        assert!((out[0][0] - 0.6).abs() < 1e-6 && (out[0][1] - 0.8).abs() < 1e-6);
        Ok(())
    }
}
