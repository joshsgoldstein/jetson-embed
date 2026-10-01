//! Is the port correct? Embed the same texts the Python baseline did and
//! compare vector by vector. Cosine similarity > 0.999 means the two stacks
//! agree to well within half-precision rounding. Anything lower is a bug in
//! *this* code: wrong pooling, wrong mask, wrong prefix, wrong special tokens.
//! docs/09-verification.md lists what each failure pattern usually means.

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::embedder::Embedder;
use crate::settings::VerifyArgs;

#[derive(Deserialize)]
struct Reference {
    model: String,
    #[serde(default)]
    prefix: String,
    dim: usize,
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Two zero vectors (e.g. "" under mean pooling) have no direction, so 0/0.
    // They are still identical outputs: count that as a match, not a failure.
    if na < 1e-12 && nb < 1e-12 {
        return 1.0;
    }
    dot / (na * nb).max(1e-12)
}

pub fn run(args: &VerifyArgs, embedder: &Embedder) -> Result<bool> {
    let text = std::fs::read_to_string(&args.reference).with_context(|| format!("reading {}", args.reference.display()))?;
    let reference: Reference = serde_json::from_str(&text).context("parsing reference JSON")?;
    println!("reference: {} (dim {}, prefix {:?}, {} texts)", reference.model, reference.dim, reference.prefix, reference.texts.len());

    let t0 = std::time::Instant::now();
    let out = embedder.embed(&reference.texts, Some(&reference.prefix), None)?;
    println!("embedded {} texts in {:.3}s on {:?}", reference.texts.len(), t0.elapsed().as_secs_f32(), embedder.device());

    let mut all_ok = true;
    println!("{:<6} {:>8}  {:>10}  text", "idx", "cosine", "max|diff|");
    for (i, (ours, theirs)) in out.vectors.iter().zip(&reference.embeddings).enumerate() {
        if ours.len() != theirs.len() {
            println!("{i:<6} DIM MISMATCH ours={} theirs={}", ours.len(), theirs.len());
            all_ok = false;
            continue;
        }
        let cos = cosine(ours, theirs);
        let max_diff = ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ok = cos >= args.min_cosine;
        all_ok &= ok;
        let preview: String = reference.texts[i].chars().take(48).collect();
        println!("{i:<6} {cos:>8.5}  {max_diff:>10.2e}  {}{preview:?}", if ok { "" } else { "FAIL " });
    }
    println!("{}", if all_ok { "PASS: all cosines >= threshold" } else { "FAIL: see rows above" });
    Ok(all_ok)
}
