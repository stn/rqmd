//! Dedicated OS-thread workers that own a `LlamaContext<'_>`.
//!
//! Why this module exists: `llama_cpp_2::context::LlamaContext<'a>` is
//! `!Send + !Sync` and borrows from `&'a LlamaModel`. We can't hold one
//! across an `.await` point, and we can't park one in an `async` field
//! or a `tokio::task::spawn_blocking` closure that wants to be reused.
//!
//! Approach B from `spike_05_context_send.rs`: each worker owns a real
//! OS thread (`std::thread`). The thread loads its context once at
//! startup, then loops reading [`EmbedJob`] / [`RerankJob`] off a
//! `std::sync::mpsc::Receiver`. Callers from the async side submit
//! jobs paired with a `tokio::sync::oneshot::Sender` for the reply, so
//! they can `.await` results without blocking the runtime.
//!
//! The `Sender` half is `Send + Sync` and lives in the worker handle.
//! The handle itself is therefore `Send + Sync`, which lets [`Pool`]
//! hold a `Vec<Worker>` in an `ArcSwapOption` field on `LlamaCpp`.
//!
//! Per-job failure (one bad text in a batch) is reported as a `None`
//! slot in the reply vec. Worker-level failure (context creation, the
//! mpsc channel closing) propagates as `Err(_)`.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use futures::future::join_all;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::llm::backend;
use crate::llm::error::{Error, Result};

// =============================================================================
// Job + reply types
// =============================================================================

/// One batch of texts to embed. The worker returns one slot per input;
/// `None` means that specific text errored (matches TS soft-failure).
struct EmbedJob {
    texts: Vec<String>,
    reply: tokio::sync::oneshot::Sender<Vec<Option<Vec<f32>>>>,
}

/// One batch of pre-formatted reranker prompts (see
/// [`crate::llm::prompt::build_qwen3_rerank_prompt`]). The worker returns
/// one `f32` per input; `None` means that prompt errored.
struct RerankJob {
    prompts: Vec<String>,
    reply: tokio::sync::oneshot::Sender<Vec<Option<f32>>>,
}

// =============================================================================
// Embed worker
// =============================================================================

/// Handle to one embedding worker thread.
pub struct EmbedWorker {
    sender: mpsc::Sender<EmbedJob>,
    join: Option<thread::JoinHandle<()>>,
}

impl EmbedWorker {
    /// Spawn a worker, block briefly until it confirms it could create
    /// its `LlamaContext`. Failure to create the context returns
    /// [`Error::Llama`]; failure to spawn the OS thread returns
    /// [`Error::Io`].
    pub fn spawn(
        model: Arc<LlamaModel>,
        pooling: LlamaPoolingType,
        n_ctx: usize,
    ) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::channel::<EmbedJob>();
        let (ack_tx, ack_rx) = mpsc::sync_channel::<Result<()>>(1);

        let join = thread::Builder::new()
            .name("rqmd-embed".into())
            .spawn(move || {
                run_embed_worker(model, pooling, n_ctx, job_rx, ack_tx);
            })
            .map_err(|e| Error::Io {
                path: std::path::PathBuf::from("<embed-worker-thread>"),
                source: e,
            })?;

        // Wait for worker to confirm context creation, or report failure.
        match ack_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: job_tx,
                join: Some(join),
            }),
            Ok(Err(e)) => {
                // Worker failed to init; let the thread finish, then propagate.
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                // Worker panicked before sending ack.
                let _ = join.join();
                Err(Error::WorkerClosed)
            }
        }
    }

    /// Submit a batch of texts; await one `f32` vec per input.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Option<Vec<f32>>>> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(EmbedJob { texts, reply: reply_tx })
            .map_err(|_| Error::WorkerClosed)?;
        reply_rx.await.map_err(|_| Error::WorkerClosed)
    }

    /// Synchronously join the worker thread, swallowing any panic.
    /// Returns once the channel has been closed AND the thread has
    /// finished. Used by [`crate::llm::llama_cpp::LlamaCpp::dispose`].
    pub fn join_blocking(mut self) {
        // Dropping the sender closes the channel; the worker exits next
        // loop iteration.
        drop(std::mem::replace(
            &mut self.sender,
            mpsc::channel().0, // unconnected dummy sender
        ));
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for EmbedWorker {
    fn drop(&mut self) {
        // If join_blocking wasn't called, do a best-effort shutdown:
        // close the channel so the worker can exit. We do NOT block on
        // join here because Drop runs in arbitrary contexts (including
        // potentially under a tokio runtime where blocking is illegal).
        drop(std::mem::replace(
            &mut self.sender,
            mpsc::channel().0,
        ));
        // self.join intentionally not awaited — leaked threads are
        // preferable to a deadlocked Drop. Callers wanting deterministic
        // shutdown should call join_blocking explicitly.
    }
}

fn run_embed_worker(
    model: Arc<LlamaModel>,
    pooling: LlamaPoolingType,
    n_ctx: usize,
    rx: mpsc::Receiver<EmbedJob>,
    ack: mpsc::SyncSender<Result<()>>,
) {
    let backend = match backend::try_get() {
        Ok(b) => b,
        Err(e) => {
            let _ = ack.send(Err(e));
            return;
        }
    };

    let ctx_params = make_pool_ctx_params(n_ctx, pooling);
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ack.send(Err(Error::Llama(format!("context init: {e}"))));
            return;
        }
    };

    // Init success.
    if ack.send(Ok(())).is_err() {
        // Caller went away; nothing to do.
        return;
    }

    while let Ok(job) = rx.recv() {
        let out = process_embed_batch(&mut ctx, &model, &job.texts);
        // Drop the reply if the caller already abandoned it.
        let _ = job.reply.send(out);
    }
}

/// Build `LlamaContextParams` for an encoder-style pool (embedding or
/// pooled-rank rerank).
///
/// llama.cpp asserts `n_ubatch >= n_tokens` for any single sequence in
/// a single forward pass — required by encoder-only / pooled models.
/// Defaults of `n_batch = n_ubatch = 512` are too small for any
/// non-trivial document. Pinning both to `n_ctx` is the natural ceiling:
/// anything that fits in the context window also fits in a single
/// ubatch (the pattern used in llama.cpp's own `examples/embedding`).
///
/// `n_ctx` is resolved from `QMD_{EMBED,RERANK}_CONTEXT_SIZE` / config /
/// `DEFAULT_*_CONTEXT_SIZE`. `CHUNK_SIZE_TOKENS = 900` is the chunker
/// invariant this ceiling must accommodate (see the warning in
/// `store_ops::embed::generate_embeddings`).
fn make_pool_ctx_params(n_ctx: usize, pooling: LlamaPoolingType) -> LlamaContextParams {
    let n_ctx_u32 = n_ctx.max(1) as u32;
    LlamaContextParams::default()
        .with_n_ctx(Some(non_zero_ctx(n_ctx)))
        .with_n_batch(n_ctx_u32)
        .with_n_ubatch(n_ctx_u32)
        .with_embeddings(true)
        .with_pooling_type(pooling)
}

fn process_embed_batch(
    ctx: &mut LlamaContext<'_>,
    model: &LlamaModel,
    texts: &[String],
) -> Vec<Option<Vec<f32>>> {
    texts
        .iter()
        .map(|text| process_one_embed(ctx, model, text).ok())
        .collect()
}

fn process_one_embed(
    ctx: &mut LlamaContext<'_>,
    model: &LlamaModel,
    text: &str,
) -> Result<Vec<f32>> {
    let tokens = model
        .str_to_token(text, AddBos::Always)
        .map_err(|e| Error::Tokenize(format!("str_to_token: {e}")))?;
    if tokens.is_empty() {
        return Err(Error::Tokenize("empty token sequence".into()));
    }
    // n_seq_max = 1 because we process one text at a time. logits_all =
    // true silences llama.cpp's "embeddings required but some input
    // tokens were not marked as outputs" warning that spike #2 surfaced.
    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    batch
        .add_sequence(&tokens, 0, true)
        .map_err(|e| Error::Llama(format!("batch.add_sequence: {e}")))?;
    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .map_err(|e| Error::Llama(format!("ctx.decode: {e}")))?;
    let pooled = ctx
        .embeddings_seq_ith(0)
        .map_err(|e| Error::Llama(format!("embeddings_seq_ith: {e}")))?;
    Ok(pooled.to_vec())
}

// =============================================================================
// Rerank worker
// =============================================================================

/// Handle to one reranker worker thread. Same shape as
/// [`EmbedWorker`], but pooling=Rank and we read the scalar score from
/// `embeddings_seq_ith(0)[0]` (see `spike_03_rerank.rs`).
pub struct RerankWorker {
    sender: mpsc::Sender<RerankJob>,
    join: Option<thread::JoinHandle<()>>,
}

impl RerankWorker {
    pub fn spawn(model: Arc<LlamaModel>, n_ctx: usize) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::channel::<RerankJob>();
        let (ack_tx, ack_rx) = mpsc::sync_channel::<Result<()>>(1);

        let join = thread::Builder::new()
            .name("rqmd-rerank".into())
            .spawn(move || {
                run_rerank_worker(model, n_ctx, job_rx, ack_tx);
            })
            .map_err(|e| Error::Io {
                path: std::path::PathBuf::from("<rerank-worker-thread>"),
                source: e,
            })?;

        match ack_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: job_tx,
                join: Some(join),
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(Error::WorkerClosed)
            }
        }
    }

    pub async fn rerank_batch(&self, prompts: Vec<String>) -> Result<Vec<Option<f32>>> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(RerankJob { prompts, reply: reply_tx })
            .map_err(|_| Error::WorkerClosed)?;
        reply_rx.await.map_err(|_| Error::WorkerClosed)
    }

    pub fn join_blocking(mut self) {
        drop(std::mem::replace(&mut self.sender, mpsc::channel().0));
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for RerankWorker {
    fn drop(&mut self) {
        drop(std::mem::replace(&mut self.sender, mpsc::channel().0));
    }
}

fn run_rerank_worker(
    model: Arc<LlamaModel>,
    n_ctx: usize,
    rx: mpsc::Receiver<RerankJob>,
    ack: mpsc::SyncSender<Result<()>>,
) {
    let backend = match backend::try_get() {
        Ok(b) => b,
        Err(e) => {
            let _ = ack.send(Err(e));
            return;
        }
    };

    let ctx_params = make_pool_ctx_params(n_ctx, LlamaPoolingType::Rank);
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ack.send(Err(Error::Llama(format!("context init: {e}"))));
            return;
        }
    };
    if ack.send(Ok(())).is_err() {
        return;
    }

    while let Ok(job) = rx.recv() {
        let scores = job
            .prompts
            .iter()
            .map(|p| process_one_rerank(&mut ctx, &model, p).ok())
            .collect();
        let _ = job.reply.send(scores);
    }
}

fn process_one_rerank(
    ctx: &mut LlamaContext<'_>,
    model: &LlamaModel,
    prompt: &str,
) -> Result<f32> {
    // AddBos::Never — the chat-template prompt already brackets with
    // <|im_start|> tags (spike #3 finding).
    let tokens = model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|e| Error::Tokenize(format!("str_to_token: {e}")))?;
    if tokens.is_empty() {
        return Err(Error::Tokenize("empty token sequence".into()));
    }
    let mut batch = LlamaBatch::new(tokens.len().max(64), 1);
    batch
        .add_sequence(&tokens, 0, true)
        .map_err(|e| Error::Llama(format!("batch.add_sequence: {e}")))?;
    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .map_err(|e| Error::Llama(format!("ctx.decode: {e}")))?;
    let pooled = ctx
        .embeddings_seq_ith(0)
        .map_err(|e| Error::Llama(format!("embeddings_seq_ith: {e}")))?;
    pooled
        .first()
        .copied()
        .ok_or_else(|| Error::Llama("rerank pooled output is empty".into()))
}

// =============================================================================
// Pool
// =============================================================================

/// Pool of [`EmbedWorker`]s.
///
/// Two ways to submit work:
///
/// * [`scatter`](Self::scatter) fans batches across workers by index
///   (`batches[i]` → `workers[i % n]`) and awaits all replies
///   concurrently via `join_all`. Per-chunk failures surface as
///   `Err(_)` slots so the caller can convert them to `None`
///   per-input (matching the `embed_batch` soft-failure contract).
/// * [`submit_to_next`](Self::submit_to_next) sends one batch to the
///   "next" worker in a round-robin (atomic counter). Used by
///   `embed()` single-text calls so concurrent callers don't all
///   serialize on `workers[0]`.
pub struct EmbedPool {
    workers: Vec<EmbedWorker>,
    next: AtomicUsize,
}

impl EmbedPool {
    pub fn new(workers: Vec<EmbedWorker>) -> Self {
        Self {
            workers,
            next: AtomicUsize::new(0),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Hand back the underlying workers (used by `dispose` to call
    /// `join_blocking` on each in turn).
    pub fn into_workers(self) -> Vec<EmbedWorker> {
        self.workers
    }

    /// Send `batches[i]` to `workers[i % n]` and await all results.
    /// Returns one outer slot per input batch (each carrying its own
    /// `Result`) so the caller can convert per-chunk failures to
    /// `None`-padded slots in input position — empty `batches` yields
    /// `Ok(vec![])`.
    pub async fn scatter(
        &self,
        batches: Vec<Vec<String>>,
    ) -> Result<Vec<Result<Vec<Option<Vec<f32>>>>>> {
        if self.workers.is_empty() {
            return Err(Error::Llama("embed pool is empty".into()));
        }
        let n = self.workers.len();
        let futures: Vec<_> = batches
            .into_iter()
            .enumerate()
            .map(|(i, batch)| self.workers[i % n].embed_batch(batch))
            .collect();
        Ok(join_all(futures).await)
    }

    /// Submit one batch to the round-robin-next worker. Used for
    /// single-text `embed()` calls so concurrent callers spread
    /// across workers instead of all hitting worker 0.
    pub async fn submit_to_next(&self, texts: Vec<String>) -> Result<Vec<Option<Vec<f32>>>> {
        if self.workers.is_empty() {
            return Err(Error::Llama("embed pool is empty".into()));
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        self.workers[idx].embed_batch(texts).await
    }
}

/// Pool of [`RerankWorker`]s. Same semantics as [`EmbedPool`].
///
/// Rerank doesn't currently expose a `submit_to_next` because the
/// only call site (`LlamaCpp::rerank`) always operates on a full
/// batch and benefits from the fan-out across workers.
pub struct RerankPool {
    workers: Vec<RerankWorker>,
}

impl RerankPool {
    pub fn new(workers: Vec<RerankWorker>) -> Self {
        Self { workers }
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn into_workers(self) -> Vec<RerankWorker> {
        self.workers
    }

    /// As [`EmbedPool::scatter`], returns per-chunk `Result` slots so
    /// the caller can decide whether to soft-fail individual chunks.
    pub async fn scatter(
        &self,
        batches: Vec<Vec<String>>,
    ) -> Result<Vec<Result<Vec<Option<f32>>>>> {
        if self.workers.is_empty() {
            return Err(Error::Llama("rerank pool is empty".into()));
        }
        let n = self.workers.len();
        let futures: Vec<_> = batches
            .into_iter()
            .enumerate()
            .map(|(i, batch)| self.workers[i % n].rerank_batch(batch))
            .collect();
        Ok(join_all(futures).await)
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert `usize` to `NonZeroU32`, clamping to at least 1. Used for
/// `with_n_ctx`. We trust `n_ctx` to fit in `u32` — context sizes are
/// in the thousands, not billions.
fn non_zero_ctx(n: usize) -> NonZeroU32 {
    NonZeroU32::new(n.max(1) as u32).expect("non-zero after max(1)")
}

/// Split `items` into approximately equal chunks of size `n_chunks`.
/// Used to slice an input list into per-worker batches before
/// [`EmbedPool::scatter`] / [`RerankPool::scatter`]. The last chunk
/// absorbs the remainder.
pub fn split_into_chunks<T: Clone>(items: Vec<T>, n_chunks: usize) -> Vec<Vec<T>> {
    if n_chunks == 0 || items.is_empty() {
        return Vec::new();
    }
    let n = n_chunks.min(items.len());
    let chunk_size = items.len().div_ceil(n);
    items
        .chunks(chunk_size)
        .map(<[T]>::to_vec)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_into_chunks_distributes_evenly() {
        let chunks = split_into_chunks(vec![1, 2, 3, 4, 5, 6], 3);
        assert_eq!(chunks, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
    }

    #[test]
    fn split_into_chunks_remainder_goes_in_last_chunk() {
        let chunks = split_into_chunks(vec![1, 2, 3, 4, 5, 6, 7], 3);
        // div_ceil(7, 3) = 3, so chunks of size 3: [1,2,3], [4,5,6], [7]
        assert_eq!(chunks, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn split_into_chunks_clamps_at_input_len() {
        let chunks = split_into_chunks(vec![1, 2], 5);
        // n = min(5, 2) = 2, chunk_size = div_ceil(2, 2) = 1
        assert_eq!(chunks, vec![vec![1], vec![2]]);
    }

    #[test]
    fn split_into_chunks_zero_chunks_is_empty() {
        let chunks: Vec<Vec<i32>> = split_into_chunks(vec![1, 2, 3], 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn split_into_chunks_empty_input_is_empty() {
        let chunks: Vec<Vec<i32>> = split_into_chunks(vec![], 3);
        assert!(chunks.is_empty());
    }
}
