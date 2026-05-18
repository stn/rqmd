//! Spike #3 — verify critical assumption #2 (rerank path):
//! `rankAll` does not exist; reranking is the same code path as embeddings,
//! but with `LlamaPoolingType::Rank`. The relevance score is the first f32
//! of the per-sequence "embedding" output.
//!
//! Pass criteria:
//!   * Three documents score relative to the query in an intuitive order
//!     (the most relevant document should get the highest score).
//!   * Record how to extract the scalar score and whether the same embedding
//!     model can serve as a reranker (Qwen3-Embedding) or whether a dedicated
//!     reranker (Qwen3-Reranker) is required.
//!
//! Override the model path with `SPIKE_MODEL_PATH=/local/path/model.gguf`.
//! Override the reranker repo/file with `SPIKE_RERANK_REPO` /
//! `SPIKE_RERANK_FILE` if you want to test the dedicated reranker.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

// Default to the dedicated reranker. Override via env vars if you want to
// probe whether the embedding model behaves sanely under pooling=Rank.
const HF_REPO: &str = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF";
const HF_FILE: &str = "qwen3-reranker-0.6b-q8_0.gguf";

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("backend init"))
}

fn resolve_model_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("SPIKE_MODEL_PATH") {
        let p = PathBuf::from(path);
        anyhow::ensure!(
            p.exists(),
            "SPIKE_MODEL_PATH does not exist: {}",
            p.display()
        );
        return Ok(p);
    }
    let repo = std::env::var("SPIKE_RERANK_REPO").unwrap_or_else(|_| HF_REPO.to_string());
    let file = std::env::var("SPIKE_RERANK_FILE").unwrap_or_else(|_| HF_FILE.to_string());
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to build hf-hub API")?;
    api.model(repo.clone())
        .get(&file)
        .with_context(|| format!("failed to download {repo}/{file}"))
}

/// Build the Qwen3-Reranker chat-formatted prompt for one (query, document) pair.
///
/// The Qwen3-Reranker model card prescribes a system prompt that asks the
/// model to judge yes/no whether the document satisfies the query, plus a
/// user message that wraps the query and document in `<Instruct>:`,
/// `<Query>:` and `<Document>:` tags. The yes/no judgment isn't used
/// directly under `pooling=Rank` — the rank head produces a single scalar
/// — but the prompt shape is what the rank head was trained on, and a
/// generic separator-based prompt gives meaningless relative scores
/// (`json` > `tokio` in the first spike run with `{query}</s><s>{doc}`).
fn qwen3_rerank_prompt(query: &str, document: &str) -> String {
    let instruct = "Given a web search query, retrieve relevant passages that answer the query";
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
         Query and the Instruct provided. Note that the answer can only be \"yes\" or \
         \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruct}\n<Query>: {query}\n\
         <Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
}

/// Score a single (query, document) pair under pooling=Rank.
///
/// Returns the first f32 of the pooled output, which is the relevance score.
fn score_one(
    backend: &LlamaBackend,
    model: &LlamaModel,
    query: &str,
    document: &str,
) -> Result<f32> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).expect("non-zero")))
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("model.new_context")?;

    let prompt = qwen3_rerank_prompt(query, document);
    // AddBos::Never because the chat template's <|im_start|> already
    // brackets the conversation; we don't want a leading BOS to confuse
    // the rank head.
    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .with_context(|| format!("failed to tokenize prompt of length {}", prompt.len()))?;

    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    // logits_all=true silences llama.cpp's "some input tokens were not marked
    // as outputs -> overriding" message when embeddings are required. For a
    // single sequence it's a no-op functionally; for v2's parallel embed path
    // we'll still want this.
    batch
        .add_sequence(&tokens, 0, true)
        .context("batch.add_sequence")?;

    ctx.clear_kv_cache();
    ctx.decode(&mut batch).context("ctx.decode")?;

    let pooled = ctx
        .embeddings_seq_ith(0)
        .context("ctx.embeddings_seq_ith(0)")?;
    anyhow::ensure!(
        !pooled.is_empty(),
        "pooled output must contain at least one value"
    );
    // The upstream reranker example reads `embeddings[0]` as the score for
    // pooling=Rank. See examples/reranker/src/main.rs:192-194.
    Ok(pooled[0])
}

fn main() -> Result<()> {
    let backend = backend();
    let model_path = resolve_model_path()?;
    eprintln!("model path = {}", model_path.display());

    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
        .context("LlamaModel::load_from_file")?;

    let query = "rust async runtime";
    let docs = [
        (
            "tokio",
            "Tokio is an asynchronous runtime for the Rust programming language.",
        ),
        (
            "rails",
            "Ruby on Rails is a server-side web application framework written in Ruby.",
        ),
        (
            "json",
            "JSON Schema is a vocabulary for annotating and validating JSON documents.",
        ),
    ];

    let mut scored: Vec<(&str, f32)> = Vec::new();
    for (name, text) in docs.iter() {
        let s = score_one(backend, &model, query, text)?;
        eprintln!("score[{name}] = {s:.4}");
        scored.push((name, s));
    }

    // Sort descending by score and report the order.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let order: Vec<&str> = scored.iter().map(|(n, _)| *n).collect();
    println!("ranked order = {order:?}");

    // Soft assertion: with a real reranker, `tokio` should win. We only warn
    // if not, because (a) the spike may be run against the embedding model
    // (env override), and (b) score signs/scales differ by model.
    if order.first().copied() != Some("tokio") {
        eprintln!("WARNING: expected `tokio` to rank first; got {order:?}");
        eprintln!("  This may indicate a different prompt template is required,");
        eprintln!("  or that this model isn't suitable as a reranker.");
    } else {
        println!("OK: `tokio` ranked first as expected.");
    }

    println!();
    println!("Notes for v2 plan:");
    println!("  - Score extraction: ctx.embeddings_seq_ith(seq_id)?[0] under pooling=Rank.");
    println!("  - Same API path as embeddings — only the pooling type changes.");
    println!("  - PROMPT FORMAT MATTERS: the upstream `{{query}}</s><s>{{doc}}` produced");
    println!("    nonsense ordering (json > tokio). Use the model's own chat template");
    println!("    (Qwen3-Reranker: yes/no judgment system prompt + <Instruct>/<Query>/");
    println!("    <Document> wrappers) for sensible scores. AddBos::Never after the");
    println!("    template since <|im_start|> already brackets the conversation.");
    println!("  - One context per call here for clarity; the real impl should");
    println!("    reuse contexts across sequences via batch.add_sequence with");
    println!("    multiple seq_ids (see upstream examples/reranker batching).");
    println!("  - logits_all=true on add_sequence silences the llama.cpp \"overriding\"");
    println!("    message when embeddings are required.");
    Ok(())
}
