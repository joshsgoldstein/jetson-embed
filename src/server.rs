//! OpenAI-compatible HTTP API on top of `Embedder`.
//!
//!   POST /v1/embeddings   {"input": "text" | ["..."], "model": "...",
//!                          "dimensions": 256, "encoding_format": "float" | "base64",
//!                          "prefix": "search_query: "}
//!   GET  /v1/models
//!   GET  /health
//!
//! Two decisions worth understanding (docs/07-serving.md):
//!   * Inference runs under `spawn_blocking` behind a 1-permit semaphore. The
//!     GPU is one resource; two concurrent forwards would double peak memory
//!     for zero throughput gain -- the exact lesson GLiNER taught on the AGX.
//!   * Errors are structured JSON with a stable `code`, so an agent calling
//!     this can act on them without parsing prose.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::embedder::Embedder;
use crate::settings::ServeArgs;

pub struct AppState {
    embedder: Arc<Embedder>,
    gate: Semaphore,
    max_inputs: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
    /// OpenAI also allows pre-tokenized input. We parse it only to reject it
    /// clearly: tiktoken ids mean nothing to this model's vocabulary.
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    input: Input,
    #[serde(default)]
    model: Option<String>,
    /// Matryoshka truncation (OpenAI's field name).
    #[serde(default)]
    dimensions: Option<usize>,
    /// Non-standard: task prefix for nomic-style models.
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(Serialize)]
struct EmbeddingObject {
    object: &'static str,
    index: usize,
    embedding: Embedding,
}

/// `untagged` makes serde write the inner value directly: a JSON array of
/// numbers for Float, a JSON string for Base64.
#[derive(Serialize)]
#[serde(untagged)]
enum Embedding {
    Float(Vec<f32>),
    Base64(String),
}

/// OpenAI's base64 format: the vector's raw little-endian f32 bytes, base64'd.
/// The OpenAI Python SDK asks for this by default and decodes it with
/// numpy.frombuffer(..., dtype="<f4"), so this is not optional.
fn to_base64(v: &[f32]) -> String {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    B64.encode(bytes)
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: Usage,
}

#[derive(Serialize)]
struct ApiError {
    code: &'static str,
    message: String,
}

fn err(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": ApiError { code, message: message.into() } }))).into_response()
}

async fn embeddings(State(st): State<Arc<AppState>>, Json(req): Json<EmbeddingsRequest>) -> Response {
    let texts = match req.input {
        Input::One(s) => vec![s],
        Input::Many(v) => v,
        Input::Tokens(_) | Input::TokenBatches(_) => {
            return err(
                StatusCode::BAD_REQUEST,
                "TOKEN_INPUT_UNSUPPORTED",
                "send text, not token ids (LangChain: OpenAIEmbeddings(..., check_embedding_ctx_length=False))",
            )
        }
    };
    if texts.is_empty() {
        return err(StatusCode::BAD_REQUEST, "EMPTY_INPUT", "'input' must be a non-empty string or array");
    }
    if let Some(i) = texts.iter().position(|t| t.is_empty()) {
        return err(
            StatusCode::BAD_REQUEST,
            "EMPTY_STRING",
            format!("input[{i}] is an empty string; an empty input has no meaningful embedding"),
        );
    }
    if texts.len() > st.max_inputs {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "TOO_MANY_INPUTS",
            format!("{} inputs, limit is {}; split into several requests and concatenate 'data' in order", texts.len(), st.max_inputs),
        );
    }
    if let Some(m) = &req.model {
        if m != &st.embedder.spec.name {
            return err(StatusCode::NOT_FOUND, "MODEL_NOT_FOUND", format!("this server serves '{}', not '{m}'", st.embedder.spec.name));
        }
    }
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return err(StatusCode::BAD_REQUEST, "UNSUPPORTED_ENCODING", format!("encoding_format '{other}': use float or base64")),
    };

    let _permit = match st.gate.acquire().await {
        Ok(p) => p,
        Err(_) => return err(StatusCode::SERVICE_UNAVAILABLE, "SHUTTING_DOWN", "server is shutting down"),
    };
    let embedder = st.embedder.clone();
    let prefix = req.prefix.clone();
    let dims = req.dimensions;
    let result = tokio::task::spawn_blocking(move || embedder.embed(&texts, prefix.as_deref(), dims)).await;

    match result {
        Ok(Ok(out)) => {
            let data = out
                .vectors
                .into_iter()
                .enumerate()
                .map(|(index, v)| EmbeddingObject {
                    object: "embedding",
                    index,
                    embedding: if base64 { Embedding::Base64(to_base64(&v)) } else { Embedding::Float(v) },
                })
                .collect();
            Json(EmbeddingsResponse {
                object: "list",
                data,
                model: st.embedder.spec.name.clone(),
                usage: Usage { prompt_tokens: out.prompt_tokens, total_tokens: out.prompt_tokens },
            })
            .into_response()
        }
        Ok(Err(e)) => {
            tracing::error!("embedding failed: {e:#}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_FAILED", format!("{e:#}"))
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "WORKER_PANICKED", e.to_string()),
    }
}

async fn models(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{ "id": st.embedder.spec.name, "object": "model", "owned_by": "local" }]
    }))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let s = &st.embedder.spec;
    Json(serde_json::json!({
        "status": "ok",
        "model": s.name,
        "backbone": format!("{:?}", s.backbone),
        "pooling": format!("{:?}", s.pooling),
        "max_seq_len": s.max_seq_len,
        "device": format!("{:?}", st.embedder.device()),
        "default_prefix": st.embedder.default_prefix,
        "inflight": 1 - st.gate.available_permits(),
    }))
}

pub async fn serve(args: ServeArgs, embedder: Embedder) -> anyhow::Result<()> {
    let state = Arc::new(AppState { embedder: Arc::new(embedder), gate: Semaphore::new(1), max_inputs: args.max_inputs });
    let app = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}  (POST /v1/embeddings)");
    axum::serve(listener, app).await?;
    Ok(())
}
