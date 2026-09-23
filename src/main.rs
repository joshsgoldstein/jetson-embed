//! jetson-embed: an OpenAI-compatible embedding server in Rust, on candle.
//! Start with docs/00-start-here.md. Module map:
//!   settings   -- CLI / env configuration
//!   registry   -- read a Hub model directory, decide backbone + pooling
//!   backbones  -- the transformers (three from candle, one written here)
//!   pooling    -- token matrix -> one vector
//!   embedder   -- tokenize, batch, encode, pool
//!   server     -- axum HTTP API
//!   verify     -- compare against the Python reference

mod backbones;
mod embedder;
mod pooling;
mod registry;
mod server;
mod settings;
mod verify;

use anyhow::Result;
use clap::Parser;

use settings::{Cli, Command};

fn init_logging() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .init();
}

fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.cmd {
        Command::Serve(args) => {
            let embedder = embedder::Embedder::load(&args.model)?;
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(server::serve(args, embedder))
        }
        Command::Embed(args) => {
            let embedder = embedder::Embedder::load(&args.model)?;
            let out = embedder.embed(&args.texts, None, None)?;
            for (text, v) in args.texts.iter().zip(&out.vectors) {
                let head: Vec<String> = v.iter().take(args.head).map(|x| format!("{x:+.4}")).collect();
                println!("{text:?}\n  dim={} [{} ...]", v.len(), head.join(", "));
            }
            println!("prompt_tokens={}", out.prompt_tokens);
            Ok(())
        }
        Command::Verify(args) => {
            let embedder = embedder::Embedder::load(&args.model)?;
            let ok = verify::run(&args, &embedder)?;
            if !ok {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
