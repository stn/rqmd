//! Spike #5 — verify critical assumption #1 (LlamaContext Send/lifetime):
//! `LlamaContext<'a>` is `!Send` and borrows from `&'a LlamaModel`. The
//! naive plan ("hold a `Mutex<Vec<Context>>` and dispatch from tokio") is
//! impossible. This spike sketches the two designs that actually work and
//! demonstrates each one compiles + produces a non-empty embedding.
//!
//! Pass criteria:
//!   * Both Approach A (per-call context) and Approach B (dedicated thread
//!     + channel) compile and run.
//!   * Each produces an embedding of the expected dimension.
//!   * Notes capture the trade-offs so the v2 plan can pick.
//!
//! Approach C (`ouroboros::self_referencing`) is described in comments but
//! not implemented, because it would add another dev-dep just for a sketch.
//! Whichever of A or B we pick will determine if C is even needed.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{Context as _, Result};
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

// ============================================================================
// Approach A — per-call context
// ============================================================================
//
// The simplest design. We hold `Arc<LlamaModel>` and create a fresh
// `LlamaContext<'_>` for each embed call. The context's lifetime is bounded
// by the function scope, so the `!Send` problem never escapes.
//
// Trade-off: pays the context-creation cost (KV cache alloc, etc.) on every
// call. No warmup benefit between sequences. Fine for low-frequency rerank
// but wasteful for batch embedding.
fn approach_a_embed(model: &LlamaModel, text: &str) -> Result<Vec<f32>> {
    let backend = backend();
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).expect("non-zero")))
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("new_context")?;
    let tokens = model.str_to_token(text, AddBos::Always)?;
    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    batch.add_sequence(&tokens, 0, false)?;
    ctx.clear_kv_cache();
    ctx.decode(&mut batch)?;
    Ok(ctx.embeddings_seq_ith(0)?.to_vec())
}

// ============================================================================
// Approach B — dedicated thread + mpsc channel
// ============================================================================
//
// One OS thread owns the model and a long-lived `LlamaContext<'_>`. Other
// threads submit jobs over a channel and receive results on a per-job
// oneshot. The context never crosses thread boundaries, so its `!Send` is
// irrelevant. The handle (`EmbedWorker`) is `Send + Sync` and can live in a
// shared struct.
//
// Trade-off: more code, but reuses the context (and KV cache, when
// applicable). One worker = serialized embeddings; a worker pool gives
// parallelism that the upstream `Promise.all` design relies on.

struct EmbedJob {
    text: String,
    reply: mpsc::Sender<Result<Vec<f32>>>,
}

struct EmbedWorker {
    sender: mpsc::Sender<EmbedJob>,
    _join: thread::JoinHandle<()>,
}

impl EmbedWorker {
    fn spawn(model: Arc<LlamaModel>) -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<EmbedJob>();
        let join = thread::Builder::new()
            .name("rmd-llm-embed-worker".into())
            .spawn(move || {
                let backend = backend();
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(Some(NonZeroU32::new(2048).expect("non-zero")))
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Mean);
                let mut ctx = match model.new_context(backend, ctx_params) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("worker: failed to create context: {e}");
                        return;
                    }
                };
                while let Ok(job) = receiver.recv() {
                    let result = (|| -> Result<Vec<f32>> {
                        let tokens = model.str_to_token(&job.text, AddBos::Always)?;
                        let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
                        batch.add_sequence(&tokens, 0, false)?;
                        ctx.clear_kv_cache();
                        ctx.decode(&mut batch)?;
                        Ok(ctx.embeddings_seq_ith(0)?.to_vec())
                    })();
                    let _ = job.reply.send(result);
                }
            })
            .context("spawn worker thread")?;
        Ok(Self {
            sender,
            _join: join,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.sender
            .send(EmbedJob {
                text: text.to_owned(),
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed"))?;
        reply_rx.recv()?
    }
}

// ============================================================================
// Approach C — ouroboros::self_referencing
// ============================================================================
//
// Sketch only (not compiled here to avoid pulling in `ouroboros`):
//
//   #[ouroboros::self_referencing]
//   struct OwnedContext {
//       model: Arc<LlamaModel>,
//       #[borrows(model)]
//       ctx: LlamaContext<'this>,
//   }
//
// Pros: holds model + context together in a single struct that can live in
//       a Vec<OwnedContext>. With a manual `unsafe impl Send`, it can cross
//       threads (every operation goes through `with_ctx_mut(|ctx| ...)`).
// Cons: adds a macro-heavy dep, the resulting type is awkward, and `Send`
//       requires an `unsafe impl` that's tricky to justify since llama.cpp
//       contexts are not safe to call from multiple threads concurrently.
//       Most ergonomic only if we want to put many contexts in a single
//       Vec and round-robin over them from a single thread.
//
// Recommendation: do NOT use Approach C unless A and B both prove
// inadequate during real implementation.

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let model_path = resolve_model_path()?;
    eprintln!("model path = {}", model_path.display());

    let backend = backend();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = Arc::new(
        LlamaModel::load_from_file(backend, &model_path, &model_params)
            .context("LlamaModel::load_from_file")?,
    );

    // Approach A
    {
        let v = approach_a_embed(&model, "hello from approach A")?;
        let nz = v.iter().filter(|x| **x != 0.0).count();
        assert!(!v.is_empty() && nz > 0);
        println!("OK: approach_a dim={} nonzero={}", v.len(), nz);
    }

    // Approach B
    {
        let worker = EmbedWorker::spawn(Arc::clone(&model))?;
        let v1 = worker.embed("hello from approach B (1)")?;
        let v2 = worker.embed("hello from approach B (2)")?;
        assert_eq!(
            v1.len(),
            v2.len(),
            "dims must match across calls on same worker"
        );
        let nz = v1.iter().filter(|x| **x != 0.0).count();
        assert!(nz > 0);
        println!(
            "OK: approach_b dim={} nonzero={} two_calls_same_worker=ok",
            v1.len(),
            nz
        );
    }

    println!();
    println!("Notes for v2 plan:");
    println!("  - Approach A: simplest, no Send headaches, no warmup. Use for");
    println!("    low-frequency operations like one-off rerank/generate.");
    println!("  - Approach B: one worker per parallel slot. Channel handle is");
    println!("    Send+Sync and lives in the LlamaCpp struct. Use for batched");
    println!("    embed/rerank where TS `Promise.all(chunks.map(...))` matters.");
    println!("  - Tokio integration for B: replace `mpsc::Sender<Reply>` with");
    println!("    `tokio::sync::oneshot::Sender<Reply>` and have the worker");
    println!("    block on a `std::sync::mpsc::Receiver<EmbedJob>` (which is");
    println!("    perfectly fine inside a dedicated OS thread).");
    println!("  - Cancellation: dropping the worker handle closes the channel,");
    println!("    so the worker exits between jobs. In-flight C++ decode is");
    println!("    NOT cancelable (same limitation noted in review item #7).");
    Ok(())
}
