//! Spike #2 — verify critical assumption #2 (embed path):
//! `model.embed_text` / `getEmbeddingFor` do not exist; embeddings come from
//! a `LlamaContext` configured with `with_embeddings(true)` + a pooling type,
//! by decoding a batch and reading `embeddings_seq_ith(seq_id)`.
//!
//! Pass criteria:
//!   * Returned `&[f32]` is non-empty and contains non-zero values.
//!   * Dimension matches the model's hidden size (Qwen3-Embedding-0.6B = 1024).
//!   * Record exact signatures (return type lifetime, error type) for v2 plan.
//!
//! Override the model path with `SPIKE_MODEL_PATH=/local/path/model.gguf` to
//! skip the HuggingFace download.

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

const HF_REPO: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";
const HF_FILE: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";

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
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to build hf-hub API")?;
    api.model(HF_REPO.to_string())
        .get(HF_FILE)
        .with_context(|| format!("failed to download {HF_REPO}/{HF_FILE}"))
}

fn main() -> Result<()> {
    let backend = backend();
    let model_path = resolve_model_path()?;
    eprintln!("model path = {}", model_path.display());

    // CPU-only is fine for a spike. GPU offload requires the corresponding
    // llama-cpp-2 Cargo feature, which is intentionally not wired up here.
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
        .context("LlamaModel::load_from_file")?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).expect("non-zero")))
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("model.new_context")?;

    let prompt = "task: search result | query: rust async runtime";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt:?}"))?;
    eprintln!("tokenized {} tokens", tokens.len());

    // Allocate a batch large enough for our prompt. n_seq_max=1 because we
    // only need one sequence at a time for this spike.
    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    batch
        .add_sequence(&tokens, 0, false)
        .context("batch.add_sequence")?;

    // Clear any state from a prior call (mirrors the upstream reranker example).
    ctx.clear_kv_cache();
    ctx.decode(&mut batch).context("ctx.decode")?;

    let emb: &[f32] = ctx
        .embeddings_seq_ith(0)
        .context("ctx.embeddings_seq_ith(0)")?;

    let nonzero = emb.iter().filter(|v| **v != 0.0).count();
    assert!(!emb.is_empty(), "embedding must be non-empty");
    assert!(nonzero > 0, "embedding must contain non-zero values");

    println!(
        "OK: dim={} nonzero={} first_4={:?}",
        emb.len(),
        nonzero,
        &emb[..4.min(emb.len())]
    );

    println!();
    println!("Notes for v2 plan:");
    println!("  - Return type: ctx.embeddings_seq_ith(i32) -> Result<&[f32], EmbeddingsError>");
    println!("  - The &[f32] borrows from `ctx`, so the caller must copy before");
    println!("    releasing the context. This is the lifetime constraint that");
    println!("    feeds into the LlamaContext<'a> !Send story (spike_05).");
    println!("  - clear_kv_cache() between sequences is required to avoid state bleed.");
    println!("  - with_pooling_type takes LlamaPoolingType by value, not by ref.");
    Ok(())
}
