//! Runtime settings. Everything is an environment variable with a CLI override,
//! so the same binary runs on the Mac (dev) and the Jetson (prod) unchanged.
//!
//! Read docs/08-jetson-memory.md before touching the batch limits: on a
//! unified-memory Jetson, GPU memory is RAM, and attention cost is quadratic in
//! sequence length, so `max_batch_tokens` is the knob that bounds peak memory.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(name = "jetson-embed", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum Command {
    /// Run the OpenAI-compatible HTTP server.
    Serve(ServeArgs),
    /// Embed texts from the command line and print the vectors (debugging).
    Embed(EmbedArgs),
    /// Compare this implementation against baseline/reference/*.json.
    Verify(VerifyArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct ModelArgs {
    /// Directory holding config.json, tokenizer.json, model.safetensors and
    /// (optionally) the sentence-transformers 1_Pooling/config.json.
    #[arg(long, env = "MODEL_DIR")]
    pub model_dir: PathBuf,

    /// Name reported by /v1/models and accepted in the "model" field.
    /// Defaults to the directory name.
    #[arg(long, env = "MODEL_NAME")]
    pub model_name: Option<String>,

    /// Weight precision: f32 (exact, 2x memory) or f16 / bf16 (Jetson default).
    #[arg(long, env = "DTYPE", default_value = "f16")]
    pub dtype: String,

    /// Force a device: cpu | cuda | metal. Default: best available.
    #[arg(long, env = "DEVICE")]
    pub device: Option<String>,

    /// Hard cap on tokens per input. Longer inputs are truncated (this is what
    /// sentence-transformers does too). Defaults to the model's own limit.
    #[arg(long, env = "MAX_SEQ_LEN")]
    pub max_seq_len: Option<usize>,

    /// Upper bound on (inputs x padded length) per forward pass. Bounds peak
    /// memory: a batch is padded to its longest member, so cost is
    /// roughly inputs x longest^2. See docs/08-jetson-memory.md.
    #[arg(long, env = "MAX_BATCH_TOKENS", default_value_t = 16_384)]
    pub max_batch_tokens: usize,

    /// Prefix prepended to every input (nomic needs "search_document: ").
    /// Callers can override per request with the non-standard "prefix" field.
    #[arg(long, env = "DEFAULT_PREFIX", default_value = "")]
    pub default_prefix: String,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ServeArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: String,
    #[arg(long, env = "PORT", default_value_t = 8015)]
    pub port: u16,
    /// Inputs per request above which the request is rejected with 413.
    #[arg(long, env = "MAX_INPUTS", default_value_t = 256)]
    pub max_inputs: usize,
}

#[derive(clap::Args, Debug, Clone)]
pub struct EmbedArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Texts to embed.
    #[arg(required = true)]
    pub texts: Vec<String>,
    /// Print only the first N dimensions of each vector.
    #[arg(long, default_value_t = 8)]
    pub head: usize,
}

#[derive(clap::Args, Debug, Clone)]
pub struct VerifyArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Reference JSON produced by baseline/embed_baseline.py.
    #[arg(long)]
    pub reference: PathBuf,
    /// Fail if any cosine similarity is below this.
    #[arg(long, default_value_t = 0.999)]
    pub min_cosine: f32,
}
