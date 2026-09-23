//! Qwen3 as a bidirectional encoder -- the backbone behind pplx-embed.
//!
//! This is the one backbone written out in full, on purpose. candle's own
//! Qwen3 is a causal decoder with a KV cache and private internals; an
//! embedding model needs neither the cache nor the causal mask. So this file
//! is the whole transformer, top to bottom, in ~250 lines:
//!
//!   token ids --> embedding table --> N x [ RMSNorm -> attention -> add
//!                                            RMSNorm -> SwiGLU MLP -> add ]
//!             --> final RMSNorm --> hidden states (batch, seq, hidden)
//!
//! The ONLY difference from the text-generation model is in `attention_bias_mask`:
//! a decoder masks out future tokens (position j > i); an encoder lets every
//! token see every other token and masks out padding only. perplexity's
//! modeling.py is 91 lines that make exactly that change to HF's Qwen3Model.
//! docs/03-transformer-anatomy.md and docs/04-bidirectional-attention.md
//! walk through each function below.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, linear_b, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv;
use serde::Deserialize;
use std::sync::Arc;

use super::Encoder;
use crate::registry::{read_json, ModelSpec};

/// The subset of config.json a Qwen3 encoder needs. Defaults match Qwen3-0.6B
/// so older configs that omit a field still load.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
}
fn default_head_dim() -> usize { 128 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_eps() -> f64 { 1e-6 }
fn default_act() -> Activation { Activation::Silu }

/// Rotary position embedding (RoPE). Instead of adding a position vector to
/// each token (BERT), RoPE rotates the query and key vectors by an angle that
/// depends on the position. Two tokens' dot product then depends only on
/// their *distance*, which is why RoPE models extrapolate to longer inputs.
/// The angle tables are computed once for max_position_embeddings positions.
struct RotaryEmbedding {
    sin: Tensor, // (max_seq, head_dim / 2)
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dtype: DType, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        // One frequency per pair of dimensions: high frequencies for the first
        // pairs (fine-grained nearby positions), low for the last (coarse, far).
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let angles = positions.matmul(&inv_freq)?; // (max_seq, dim/2)
        Ok(Self { sin: angles.sin()?.to_dtype(dtype)?, cos: angles.cos()?.to_dtype(dtype)? })
    }

    /// q, k: (batch, heads, seq, head_dim). Positions always start at 0 here
    /// because an encoder processes the whole input at once (no KV-cache offset).
    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

/// SwiGLU feed-forward: down( silu(gate(x)) * up(x) ). The "gate" branch
/// decides how much of each "up" feature to let through. Three matrices
/// instead of BERT's two, and no biases.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

/// Grouped-query attention with per-head RMSNorm on q and k (a Qwen3 detail
/// that stabilises training; at inference it is just two extra norms).
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(cfg: &Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let (h, kvh, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        Ok(Self {
            q_proj: linear_b(cfg.hidden_size, h * d, cfg.attention_bias, vb.pp("q_proj"))?,
            k_proj: linear_b(cfg.hidden_size, kvh * d, cfg.attention_bias, vb.pp("k_proj"))?,
            v_proj: linear_b(cfg.hidden_size, kvh * d, cfg.attention_bias, vb.pp("v_proj"))?,
            o_proj: linear_b(h * d, cfg.hidden_size, cfg.attention_bias, vb.pp("o_proj"))?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: h,
            num_kv_heads: kvh,
            head_dim: d,
            rotary,
        })
    }

    /// x: (batch, seq, hidden). bias: (batch, 1, 1, seq) additive mask,
    /// 0 for real key positions and a large negative for padding.
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kvh, d) = (self.num_heads, self.num_kv_heads, self.head_dim);

        // 1. Project to queries, keys, values and split into heads: (b, heads, l, d)
        let q = self.q_proj.forward(x)?.reshape((b, l, h, d))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?.reshape((b, l, kvh, d))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, l, kvh, d))?.transpose(1, 2)?;

        // 2. Qwen3's per-head norm on q and k (over head_dim)
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((b, h, l, d))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((b, kvh, l, d))?;

        // 3. Encode position by rotating q and k
        let (q, k) = self.rotary.apply(&q, &k)?;

        // 4. GQA: 16 query heads share 8 key/value heads; repeat k, v to match
        let k = repeat_kv(k, h / kvh)?.contiguous()?;
        let v = repeat_kv(v, h / kvh)?.contiguous()?;

        // 5. Scaled dot-product attention. THIS is where bidirectional differs
        //    from causal: `bias` masks padding only, never the future.
        let scale = 1.0 / (d as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // (b, h, l, l)
        let scores = scores.broadcast_add(bias)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (b, h, l, d)

        // 6. Merge heads and project back to hidden size
        Ok(self.o_proj.forward(&ctx.transpose(1, 2)?.reshape((b, l, h * d))?)?)
    }
}

/// One transformer block, pre-norm style: normalise, transform, add back.
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn new(cfg: &Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }
    fn forward(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let h = self.self_attn.forward(&self.input_layernorm.forward(x)?, bias)?;
        let x = (x + h)?; // residual connection: the block adds a correction to its input
        let h = self.mlp.forward(&self.post_attention_layernorm.forward(&x)?)?;
        Ok((x + h)?)
    }
}

pub struct Qwen3Bidirectional {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    hidden: usize,
    dtype: DType,
}

impl Qwen3Bidirectional {
    pub fn load(spec: &ModelSpec, vb: VarBuilder) -> Result<Self> {
        let cfg: Config = read_json(&spec.config_path).context("parsing Qwen3 config")?;
        // pplx-embed saves a bare Qwen3Model ("embed_tokens.weight"); a checkpoint
        // exported from Qwen3ForCausalLM nests everything under "model.".
        let vb = if vb.contains_tensor("embed_tokens.weight") { vb } else { vb.pp("model") };
        let rotary = Arc::new(RotaryEmbedding::new(&cfg, vb.dtype(), vb.device())?);
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(&cfg, rotary.clone(), vb.pp("layers").pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self { embed_tokens, layers, norm, hidden: cfg.hidden_size, dtype: vb.dtype() })
    }

    /// Turn a (batch, seq) 0/1 padding mask into the additive bias attention
    /// expects: 0 where a key may be attended to, -1e4 where it is padding.
    /// -1e4 rather than -inf so f16 stays finite and softmax never sees NaN.
    /// A causal model would ALSO put -1e4 wherever key position > query position.
    fn attention_bias_mask(&self, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, l) = attention_mask.dims2()?;
        let keep = attention_mask.to_dtype(self.dtype)?; // 1 real, 0 pad
        let bias = ((keep.ones_like()? - keep)? * -1e4)?; // 0 real, -1e4 pad
        Ok(bias.reshape((b, 1, 1, l))?)
    }
}

impl Encoder for Qwen3Bidirectional {
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor, _token_type_ids: &Tensor) -> Result<Tensor> {
        let bias = self.attention_bias_mask(attention_mask)?;
        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            h = layer.forward(&h, &bias)?;
        }
        Ok(self.norm.forward(&h)?)
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
}
