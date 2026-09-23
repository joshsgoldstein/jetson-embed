//! The pipeline: text -> tokens -> padded batches -> encoder -> pooling ->
//! unit vectors. This is the file to read to see how the pieces connect.
//! docs/06-tokenization-and-batching.md explains the batching policy.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::backbones::{self, Encoder};
use crate::pooling::{l2_normalize, pool, truncate_dims};
use crate::registry::ModelSpec;
use crate::settings::ModelArgs;

pub struct Embedder {
    pub spec: ModelSpec,
    encoder: Box<dyn Encoder>,
    tokenizer: Tokenizer,
    device: Device,
    pad_id: u32,
    max_batch_tokens: usize,
    pub default_prefix: String,
}

pub struct EmbedOutput {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
}

impl Embedder {
    pub fn load(args: &ModelArgs) -> Result<Self> {
        let spec = ModelSpec::discover(&args.model_dir, args.model_name.clone(), args.max_seq_len)?;
        let device = select_device(args.device.as_deref())?;
        let mut dtype = parse_dtype(&args.dtype)?;
        if device.is_cpu() && dtype != DType::F32 {
            tracing::warn!("{dtype:?} requested on CPU; using f32 (half precision on CPU is slow and some ops lack kernels)");
            dtype = DType::F32;
        }
        tracing::info!(
            "loading {} ({:?}, {:?} pooling, max_seq_len {}) as {dtype:?} on {device:?}",
            spec.name, spec.backbone, spec.pooling, spec.max_seq_len
        );
        let t0 = std::time::Instant::now();
        let encoder = backbones::load(&spec, &device, dtype)?;
        tracing::info!("weights loaded in {:.1}s, hidden size {}", t0.elapsed().as_secs_f32(), encoder.hidden_size());

        let mut tokenizer = Tokenizer::from_file(&spec.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        // Truncate here, once, instead of in every caller. This mirrors
        // sentence-transformers, which silently truncates to max_seq_length.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: spec.max_seq_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("setting truncation: {e}"))?;
        // We pad ourselves per batch (see `encode_batch`), so no tokenizer padding.
        tokenizer.with_padding(None);
        let pad_id = spec
            .pad_token
            .as_deref()
            .and_then(|t| tokenizer.token_to_id(t))
            .unwrap_or_else(|| {
                tracing::warn!("no pad token found in tokenizer; using id 0");
                0
            });

        Ok(Self {
            spec,
            encoder,
            tokenizer,
            device,
            pad_id,
            max_batch_tokens: args.max_batch_tokens.max(1),
            default_prefix: args.default_prefix.clone(),
        })
    }

    /// Embed `texts`. `prefix` overrides the server default (nomic task
    /// prefixes); `dims` truncates Matryoshka models. Output order == input order.
    pub fn embed(&self, texts: &[String], prefix: Option<&str>, dims: Option<usize>) -> Result<EmbedOutput> {
        if texts.is_empty() {
            return Ok(EmbedOutput { vectors: vec![], prompt_tokens: 0 });
        }
        let prefix = prefix.unwrap_or(&self.default_prefix);
        let inputs: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();

        // 1. Tokenize everything (special tokens on: [CLS]/[SEP], <s>/</s>, etc.)
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let mut items: Vec<(usize, Vec<u32>, Vec<u32>)> = encodings
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                let mut ids = e.get_ids().to_vec();
                let mut types = e.get_type_ids().to_vec();
                if ids.is_empty() {
                    // "" tokenizes to nothing for decoder tokenizers; give the
                    // model one token to attend to rather than an empty tensor.
                    ids.push(self.pad_id);
                    types.push(0);
                }
                (i, ids, types)
            })
            .collect();
        let prompt_tokens: usize = items.iter().map(|(_, ids, _)| ids.len()).sum();

        // 2. Sort longest-first so each batch pads as little as possible, then
        //    cut batches where (count x longest) would exceed max_batch_tokens.
        items.sort_by_key(|(_, ids, _)| std::cmp::Reverse(ids.len()));
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut start = 0;
        while start < items.len() {
            let longest = items[start].1.len();
            let per_batch = (self.max_batch_tokens / longest.max(1)).max(1);
            let end = (start + per_batch).min(items.len());
            let batch = &items[start..end];
            let out = self.encode_batch(batch, dims)?;
            for ((orig, _, _), v) in batch.iter().zip(out) {
                vectors[*orig] = Some(v);
            }
            start = end;
        }
        Ok(EmbedOutput { vectors: vectors.into_iter().map(|v| v.expect("every input embedded")).collect(), prompt_tokens })
    }

    /// One forward pass. Pads to the longest member; the attention mask tells
    /// the model (and the pooler) which positions are real.
    fn encode_batch(&self, batch: &[(usize, Vec<u32>, Vec<u32>)], dims: Option<usize>) -> Result<Vec<Vec<f32>>> {
        let b = batch.len();
        let l = batch.iter().map(|(_, ids, _)| ids.len()).max().unwrap_or(1);
        let mut ids = Vec::with_capacity(b * l);
        let mut mask = Vec::with_capacity(b * l);
        let mut types = Vec::with_capacity(b * l);
        for (_, seq, ty) in batch {
            ids.extend_from_slice(seq);
            ids.extend(std::iter::repeat(self.pad_id).take(l - seq.len()));
            mask.extend(std::iter::repeat(1u32).take(seq.len()));
            mask.extend(std::iter::repeat(0u32).take(l - seq.len()));
            types.extend_from_slice(ty);
            types.extend(std::iter::repeat(0u32).take(l - ty.len()));
        }
        let input_ids = Tensor::from_vec(ids, (b, l), &self.device)?;
        let attention_mask = Tensor::from_vec(mask, (b, l), &self.device)?;
        let token_type_ids = Tensor::from_vec(types, (b, l), &self.device)?;

        let hidden = self.encoder.encode(&input_ids, &attention_mask, &token_type_ids)?;
        let pooled = pool(self.spec.pooling, &hidden, &attention_mask)?;
        let mut vec = l2_normalize(&pooled.to_dtype(DType::F32)?)?;
        if let Some(d) = dims {
            vec = truncate_dims(&vec, d)?;
        }
        Ok(vec.to_vec2::<f32>().context("copying embeddings to host")?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        other => bail!("unknown dtype '{other}' (use f32, f16 or bf16)"),
    })
}

/// cuda > metal > cpu unless the user forces one. The feature flags decide
/// what was compiled in; a "cuda" request on a CPU-only build is an error, not
/// a silent fallback -- silent CPU fallback is how a Jetson ends up running a
/// model on its ARM cores and nobody notices for a week.
fn select_device(forced: Option<&str>) -> Result<Device> {
    match forced.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda") => cuda_device().context("cuda requested but unavailable (build with --features cuda on a CUDA host)"),
        Some("metal") => metal_device().context("metal requested but unavailable (build with --features metal on macOS)"),
        Some(other) => bail!("unknown device '{other}'"),
        None => Ok(cuda_device().or_else(|_| metal_device()).unwrap_or(Device::Cpu)),
    }
}

fn cuda_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[allow(unreachable_code)]
    bail!("not built with the cuda feature")
}

fn metal_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        return Ok(Device::new_metal(0)?);
    }
    #[allow(unreachable_code)]
    bail!("not built with the metal feature")
}
