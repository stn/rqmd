//! Core `LlamaCpp` struct and `Llm` impl.
//!
//! Wires together everything PR1 set up (config, format, gpu, pull,
//! prompt) with PR2's worker pool. The struct holds all lazily-loaded
//! state (models + pools) behind `ArcSwapOption` so reads are
//! lock-free, and only the load-path takes a `tokio::sync::Mutex` for
//! double-load prevention.
//!
//! Lifecycle rules (mirrors `tobi/qmd/src/llm.ts` `class LlamaCpp`):
//!
//! * `LlamaCppConfig` is the only construction surface. It centralizes
//!   env reads at struct creation time — in particular `ci_mode` is
//!   captured here so that parallel tests can inject `ci_mode: true`
//!   without racing on a shared `CI` env var (Rust 2024 edition makes
//!   `std::env::set_var` `unsafe` precisely because of that race).
//! * Method entry: `ensure_alive()` → `ensure_not_ci(op)` (when
//!   applicable) → `in_flight.acquire()` → re-`ensure_alive()` to close
//!   the dispose race window.
//! * Synchronous FFI work goes through `tokio::task::spawn_blocking`.
//!   `LlamaContext<'_>` is created inside the closure (or inside a
//!   worker thread) so it never crosses an `.await`.
//! * `dispose()` is idempotent: swaps the `disposed` flag, drains
//!   in-flight up to 30s, then joins worker threads with per-resource
//!   timeouts.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tokio::sync::Mutex as AsyncMutex;

use crate::llm::backend;
use crate::llm::config;
use crate::llm::error::{Error, Result};
use crate::llm::format;
use crate::llm::gpu;
use crate::llm::prompt;
use crate::llm::pull;
use crate::llm::traits::{Llm, LlamaToken};
use crate::llm::types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateResult, GenerateOptions, ModelInfo,
    ModelResolutionConfig, PullOptions, QueryType, Queryable, RerankDocument, RerankDocumentResult,
    RerankOptions, RerankResult,
};
use crate::llm::worker::{EmbedPool, EmbedWorker, RerankPool, RerankWorker, split_into_chunks};

// =============================================================================
// Config
// =============================================================================

/// Construction-time configuration for [`LlamaCpp`].
///
/// All fields are optional; `None` means "fall back to the env var or
/// crate default". `ci_mode` is explicitly NOT optional — call sites
/// must opt in to CI guards consciously (mainly via [`Self::from_env`]
/// for production binaries, or via a literal `true` in tests).
#[derive(Debug, Clone, Default)]
pub struct LlamaCppConfig {
    pub embed_model: Option<String>,
    pub generate_model: Option<String>,
    pub rerank_model: Option<String>,
    pub model_cache_dir: Option<PathBuf>,
    pub expand_context_size: Option<usize>,
    pub embed_context_size: Option<usize>,
    pub rerank_context_size: Option<usize>,
    /// Number of dedicated worker threads for the embed pool. Default
    /// 2; clamped at the env level by [`gpu::resolve_safe_parallelism`].
    pub embed_parallelism: Option<usize>,
    /// Number of dedicated worker threads for the rerank pool. Default 2.
    pub rerank_parallelism: Option<usize>,
    /// When true, `embed_batch` / `generate` / `expand_query` / `rerank`
    /// short-circuit to [`Error::CiDisabled`] without loading any
    /// model. Tests should set this explicitly; production binaries
    /// can use [`Self::from_env`] to honor `$CI`.
    pub ci_mode: bool,
}

impl LlamaCppConfig {
    /// Construct a config from the environment. Currently only reads
    /// `$CI` (any non-empty value → `ci_mode = true`); all other
    /// fields use crate defaults / per-method env reads.
    pub fn from_env() -> Self {
        Self {
            ci_mode: std::env::var("CI").map(|v| !v.is_empty()).unwrap_or(false),
            ..Default::default()
        }
    }
}

// =============================================================================
// In-flight counter (RAII guard)
// =============================================================================

/// Counts in-flight operations on [`LlamaCpp`]. `dispose()` waits on
/// this to reach zero before tearing down workers, closing the
/// otherwise-racy window where a load could resurrect a freshly-dropped
/// model.
#[derive(Default)]
struct InFlightCounter {
    count: AtomicUsize,
}

impl InFlightCounter {
    fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    fn acquire(self: &Arc<Self>) -> InFlightGuard {
        self.count.fetch_add(1, Ordering::AcqRel);
        InFlightGuard {
            counter: self.clone(),
        }
    }
}

struct InFlightGuard {
    counter: Arc<InFlightCounter>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.count.fetch_sub(1, Ordering::AcqRel);
    }
}

// =============================================================================
// LlamaCpp
// =============================================================================

pub struct LlamaCpp {
    embed_model_uri: String,
    generate_model_uri: String,
    rerank_model_uri: String,
    model_cache_dir: Option<PathBuf>,
    expand_context_size: usize,
    embed_context_size: usize,
    rerank_context_size: usize,
    embed_parallelism: usize,
    rerank_parallelism: usize,
    ci_mode: bool,

    embed_model: ArcSwapOption<LlamaModel>,
    generate_model: ArcSwapOption<LlamaModel>,
    rerank_model: ArcSwapOption<LlamaModel>,

    embed_pool: ArcSwapOption<EmbedPool>,
    rerank_pool: ArcSwapOption<RerankPool>,

    // Double-load prevention. We hold these across .await intentionally
    // — tokio::sync::Mutex is the right primitive (std::sync::Mutex
    // would trip clippy::await_holding_lock).
    embed_model_load: AsyncMutex<()>,
    generate_model_load: AsyncMutex<()>,
    rerank_model_load: AsyncMutex<()>,
    embed_pool_load: AsyncMutex<()>,
    rerank_pool_load: AsyncMutex<()>,

    disposed: AtomicBool,
    in_flight: Arc<InFlightCounter>,
}

impl LlamaCpp {
    pub fn new(config: LlamaCppConfig) -> Self {
        let model_res = ModelResolutionConfig {
            embed: config.embed_model,
            generate: config.generate_model,
            rerank: config.rerank_model,
        };
        let expand_size = config::resolve_expand_context_size(config.expand_context_size)
            .unwrap_or(config::DEFAULT_EXPAND_CONTEXT_SIZE);
        let embed_ctx = config
            .embed_context_size
            .unwrap_or_else(config::resolve_embed_context_size);
        let rerank_ctx = config
            .rerank_context_size
            .unwrap_or_else(config::resolve_rerank_context_size);

        let embed_par = resolve_pool_parallelism(
            config.embed_parallelism,
            "QMD_EMBED_PARALLELISM",
            DEFAULT_EMBED_PARALLELISM,
        );
        let rerank_par = resolve_pool_parallelism(
            config.rerank_parallelism,
            "QMD_RERANK_PARALLELISM",
            DEFAULT_RERANK_PARALLELISM,
        );

        Self {
            embed_model_uri: config::resolve_embed_model(Some(&model_res)),
            generate_model_uri: config::resolve_generate_model(Some(&model_res)),
            rerank_model_uri: config::resolve_rerank_model(Some(&model_res)),
            model_cache_dir: config.model_cache_dir,
            expand_context_size: expand_size,
            embed_context_size: embed_ctx,
            rerank_context_size: rerank_ctx,
            embed_parallelism: embed_par,
            rerank_parallelism: rerank_par,
            ci_mode: config.ci_mode,

            embed_model: ArcSwapOption::const_empty(),
            generate_model: ArcSwapOption::const_empty(),
            rerank_model: ArcSwapOption::const_empty(),
            embed_pool: ArcSwapOption::const_empty(),
            rerank_pool: ArcSwapOption::const_empty(),

            embed_model_load: AsyncMutex::new(()),
            generate_model_load: AsyncMutex::new(()),
            rerank_model_load: AsyncMutex::new(()),
            embed_pool_load: AsyncMutex::new(()),
            rerank_pool_load: AsyncMutex::new(()),

            disposed: AtomicBool::new(false),
            in_flight: Arc::new(InFlightCounter::default()),
        }
    }

    /// Construct from the environment (`LlamaCppConfig::from_env`).
    pub fn with_env() -> Self {
        Self::new(LlamaCppConfig::from_env())
    }

    // ---- accessors used by tests / external callers ------------------------

    pub fn embed_model_uri(&self) -> &str {
        &self.embed_model_uri
    }

    pub fn generate_model_uri(&self) -> &str {
        &self.generate_model_uri
    }

    pub fn rerank_model_uri(&self) -> &str {
        &self.rerank_model_uri
    }

    /// Resolved embed-pool context size (env / config / default).
    /// `store_ops::embed` uses this to warn if it's smaller than
    /// `CHUNK_SIZE_TOKENS`.
    pub fn embed_context_size(&self) -> usize {
        self.embed_context_size
    }

    /// Resolved rerank-pool context size.
    pub fn rerank_context_size(&self) -> usize {
        self.rerank_context_size
    }

    pub fn ci_mode(&self) -> bool {
        self.ci_mode
    }

    pub fn is_disposed(&self) -> bool {
        self.disposed.load(Ordering::Acquire)
    }

    /// Number of `LlamaCpp` methods currently mid-flight. Exposed
    /// for tests; production callers should not depend on the exact
    /// value (operations come and go quickly).
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.count()
    }

    // ---- internal helpers --------------------------------------------------

    fn ensure_alive(&self) -> Result<()> {
        if self.is_disposed() {
            Err(Error::Disposed)
        } else {
            Ok(())
        }
    }

    fn ensure_not_ci(&self) -> Result<()> {
        if self.ci_mode {
            Err(Error::CiDisabled)
        } else {
            Ok(())
        }
    }

    async fn load_model(&self, uri: &str) -> Result<Arc<LlamaModel>> {
        let uri_owned = uri.to_owned();
        let cache_dir = self.model_cache_dir.clone();

        // 1. Download (if needed) on a blocking thread.
        let path = tokio::task::spawn_blocking({
            let uri = uri_owned.clone();
            move || -> Result<PathBuf> {
                let results = pull::pull_models(
                    &[uri],
                    &PullOptions {
                        refresh: false,
                        cache_dir,
                    },
                )?;
                results
                    .into_iter()
                    .next()
                    .map(|r| r.path)
                    .ok_or_else(|| Error::Llama("pull_models returned no result".into()))
            }
        })
        .await??;

        // 2. Load the GGUF.
        let uri_for_err = uri_owned.clone();
        let model = tokio::task::spawn_blocking(move || -> Result<LlamaModel> {
            let backend = backend::try_get()?;
            let params = make_model_params();
            LlamaModel::load_from_file(backend, &path, &params).map_err(|e| Error::ModelLoad {
                uri: uri_for_err,
                source: Box::new(e),
            })
        })
        .await??;

        Ok(Arc::new(model))
    }

    async fn ensure_embed_model(&self) -> Result<Arc<LlamaModel>> {
        if let Some(m) = self.embed_model.load_full() {
            return Ok(m);
        }
        let _lock = self.embed_model_load.lock().await;
        if let Some(m) = self.embed_model.load_full() {
            return Ok(m);
        }
        let model = self.load_model(&self.embed_model_uri).await?;
        self.embed_model.store(Some(model.clone()));
        Ok(model)
    }

    async fn ensure_generate_model(&self) -> Result<Arc<LlamaModel>> {
        if let Some(m) = self.generate_model.load_full() {
            return Ok(m);
        }
        let _lock = self.generate_model_load.lock().await;
        if let Some(m) = self.generate_model.load_full() {
            return Ok(m);
        }
        let model = self.load_model(&self.generate_model_uri).await?;
        self.generate_model.store(Some(model.clone()));
        Ok(model)
    }

    async fn ensure_rerank_model(&self) -> Result<Arc<LlamaModel>> {
        if let Some(m) = self.rerank_model.load_full() {
            return Ok(m);
        }
        let _lock = self.rerank_model_load.lock().await;
        if let Some(m) = self.rerank_model.load_full() {
            return Ok(m);
        }
        let model = self.load_model(&self.rerank_model_uri).await?;
        self.rerank_model.store(Some(model.clone()));
        Ok(model)
    }

    async fn ensure_embed_pool(&self) -> Result<Arc<EmbedPool>> {
        if let Some(p) = self.embed_pool.load_full() {
            return Ok(p);
        }
        let _lock = self.embed_pool_load.lock().await;
        if let Some(p) = self.embed_pool.load_full() {
            return Ok(p);
        }
        let model = self.ensure_embed_model().await?;
        let n = self.embed_parallelism;
        let ctx = self.embed_context_size;
        let pool = tokio::task::spawn_blocking(move || -> Result<EmbedPool> {
            let mut workers = Vec::with_capacity(n);
            for _ in 0..n {
                workers.push(EmbedWorker::spawn(
                    model.clone(),
                    LlamaPoolingType::Mean,
                    ctx,
                )?);
            }
            Ok(EmbedPool::new(workers))
        })
        .await??;
        let arc = Arc::new(pool);
        self.embed_pool.store(Some(arc.clone()));
        Ok(arc)
    }

    async fn ensure_rerank_pool(&self) -> Result<Arc<RerankPool>> {
        if let Some(p) = self.rerank_pool.load_full() {
            return Ok(p);
        }
        let _lock = self.rerank_pool_load.lock().await;
        if let Some(p) = self.rerank_pool.load_full() {
            return Ok(p);
        }
        let model = self.ensure_rerank_model().await?;
        let n = self.rerank_parallelism;
        let ctx = self.rerank_context_size;
        let pool = tokio::task::spawn_blocking(move || -> Result<RerankPool> {
            let mut workers = Vec::with_capacity(n);
            for _ in 0..n {
                workers.push(RerankWorker::spawn(model.clone(), ctx)?);
            }
            Ok(RerankPool::new(workers))
        })
        .await??;
        let arc = Arc::new(pool);
        self.rerank_pool.store(Some(arc.clone()));
        Ok(arc)
    }
}

// =============================================================================
// Llm impl
// =============================================================================

#[async_trait]
impl Llm for LlamaCpp {
    async fn embed(&self, text: &str, opts: EmbedOptions) -> Result<Option<EmbeddingResult>> {
        // TS `embed()` does NOT check ciMode (only embedBatch / generate /
        // expandQuery / rerank do). Mirror that.
        self.ensure_alive()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let formatted = format_for_embedding(text, &opts, &self.embed_model_uri);
        let pool = self.ensure_embed_pool().await?;
        // Single-text path uses round-robin so concurrent `embed()` calls
        // spread across workers instead of all serializing on workers[0].
        let mut raw = pool.submit_to_next(vec![formatted]).await?;
        let model_name = opts
            .model
            .unwrap_or_else(|| self.embed_model_uri.clone());
        Ok(raw
            .pop()
            .flatten()
            .map(|embedding| EmbeddingResult {
                embedding,
                model: model_name,
            }))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        opts: EmbedOptions,
    ) -> Result<Vec<Option<EmbeddingResult>>> {
        self.ensure_alive()?;
        self.ensure_not_ci()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let pool = self.ensure_embed_pool().await?;
        let n_workers = pool.worker_count();

        let formatted: Vec<String> = texts
            .iter()
            .map(|t| format_for_embedding(t, &opts, &self.embed_model_uri))
            .collect();
        let chunks = split_into_chunks(formatted, n_workers);
        let chunk_sizes: Vec<usize> = chunks.iter().map(Vec::len).collect();
        let raw_results = pool.scatter(chunks).await?;

        let model_name = opts
            .model
            .unwrap_or_else(|| self.embed_model_uri.clone());
        Ok(embed_results_to_output(raw_results, &chunk_sizes, &model_name))
    }

    async fn generate(
        &self,
        prompt_str: &str,
        opts: GenerateOptions,
    ) -> Result<Option<GenerateResult>> {
        self.ensure_alive()?;
        self.ensure_not_ci()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let model = self.ensure_generate_model().await?;
        let model_uri = self.generate_model_uri.clone();
        let prompt_owned = prompt_str.to_owned();
        let max_tokens = opts.max_tokens.unwrap_or(150) as i32;
        let temperature = opts.temperature.unwrap_or(0.7);
        let n_ctx = self.expand_context_size;

        let text: String = tokio::task::spawn_blocking(move || -> Result<String> {
            let backend = backend::try_get()?;
            let messages = vec![LlamaChatMessage::new("user".into(), prompt_owned)
                .map_err(|e| Error::ChatTemplate(format!("user msg: {e}")))?];
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::top_k(20),
                LlamaSampler::top_p(0.8, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::penalties(64, 1.0, 0.0, 0.5),
                LlamaSampler::dist(1234),
            ]);
            run_chat_decode(backend, &model, &messages, n_ctx, max_tokens, &mut sampler)
        })
        .await??;

        Ok(Some(GenerateResult {
            text,
            model: model_uri,
            logprobs: None,
            done: true,
        }))
    }

    async fn model_exists(&self, model_uri: &str) -> Result<ModelInfo> {
        if model_uri.starts_with("hf:") {
            return Ok(ModelInfo {
                name: model_uri.to_owned(),
                exists: true,
                path: None,
            });
        }
        let path = PathBuf::from(model_uri);
        let exists = path.exists();
        Ok(ModelInfo {
            name: model_uri.to_owned(),
            exists,
            path: if exists { Some(path) } else { None },
        })
    }

    /// Expand a search query into `lex`/`vec`/`hyde` lines.
    ///
    /// **Context budget**: uses `self.expand_context_size` (env:
    /// `QMD_EXPAND_CONTEXT_SIZE`, default 2048) for the generation
    /// context, with a hard cap of 600 new tokens. Long queries +
    /// verbose chat templates can exceed the budget; if you see a
    /// "ctx.decode" error from this method, raise
    /// `QMD_EXPAND_CONTEXT_SIZE` or override
    /// [`LlamaCppConfig::expand_context_size`].
    async fn expand_query(
        &self,
        query: &str,
        opts: ExpandQueryOptions,
    ) -> Result<Vec<Queryable>> {
        self.ensure_alive()?;
        self.ensure_not_ci()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let model = self.ensure_generate_model().await?;
        let query_owned = query.to_owned();
        let include_lexical = opts.include_lexical.unwrap_or(true);
        let intent = opts.intent.clone();
        let n_ctx = self.expand_context_size;
        let q_for_decode = query_owned.clone();

        let raw_text: String = tokio::task::spawn_blocking(move || -> Result<String> {
            let backend = backend::try_get()?;
            let user_msg =
                prompt::build_expand_query_user_message(&q_for_decode, intent.as_deref());
            let messages = vec![
                LlamaChatMessage::new(
                    "system".into(),
                    prompt::EXPAND_QUERY_SYSTEM_PROMPT.into(),
                )
                .map_err(|e| Error::ChatTemplate(format!("system msg: {e}")))?,
                LlamaChatMessage::new("user".into(), user_msg)
                    .map_err(|e| Error::ChatTemplate(format!("user msg: {e}")))?,
            ];
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::top_k(20),
                LlamaSampler::top_p(0.8, 1),
                LlamaSampler::temp(0.7),
                LlamaSampler::penalties(64, 1.0, 0.0, 0.5),
                LlamaSampler::dist(1234),
            ]);
            run_chat_decode(backend, &model, &messages, n_ctx, 600, &mut sampler)
        })
        .await??;

        let parsed = prompt::parse_expand_query_output(&raw_text);
        let filtered = prompt::filter_with_query_terms(&query_owned, parsed);
        if filtered.is_empty() {
            return Ok(prompt::fallback_queryables(&query_owned, include_lexical));
        }
        Ok(if include_lexical {
            filtered
        } else {
            filtered
                .into_iter()
                .filter(|q| q.type_ != QueryType::Lex)
                .collect()
        })
    }

    async fn rerank(
        &self,
        query: &str,
        docs: &[RerankDocument],
        opts: RerankOptions,
    ) -> Result<RerankResult> {
        self.ensure_alive()?;
        self.ensure_not_ci()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let model_name = opts
            .model
            .unwrap_or_else(|| self.rerank_model_uri.clone());

        if docs.is_empty() {
            return Ok(RerankResult {
                results: Vec::new(),
                model: model_name,
            });
        }

        let pool = self.ensure_rerank_pool().await?;
        let n_workers = pool.worker_count();

        let prompts: Vec<String> = docs
            .iter()
            .map(|d| prompt::build_qwen3_rerank_prompt(query, &d.text))
            .collect();
        let chunks = split_into_chunks(prompts, n_workers);
        let chunk_sizes: Vec<usize> = chunks.iter().map(Vec::len).collect();
        let raw_results = pool.scatter(chunks).await?;
        let flat: Vec<Option<f32>> = flatten_with_chunk_fallback(raw_results, &chunk_sizes);

        let mut results: Vec<RerankDocumentResult> = docs
            .iter()
            .zip(flat.iter())
            .enumerate()
            .filter_map(|(index, (doc, score_opt))| {
                score_opt.map(|score| RerankDocumentResult {
                    file: doc.file.clone(),
                    score,
                    index,
                })
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(RerankResult {
            results,
            model: model_name,
        })
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        self.ensure_alive()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let model = self.ensure_embed_model().await?;
        let text_owned = text.to_owned();
        let tokens = tokio::task::spawn_blocking(move || {
            model
                .str_to_token(&text_owned, AddBos::Always)
                .map_err(|e| Error::Tokenize(format!("str_to_token: {e}")))
        })
        .await??;
        Ok(tokens)
    }

    async fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String> {
        self.ensure_alive()?;
        let _guard = self.in_flight.acquire();
        self.ensure_alive()?;

        let model = self.ensure_embed_model().await?;
        let tokens_owned: Vec<LlamaToken> = tokens.to_vec();
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut out = String::new();
            for t in tokens_owned {
                let piece = model
                    .token_to_piece(t, &mut decoder, false, None)
                    .map_err(|e| Error::Tokenize(format!("token_to_piece: {e}")))?;
                out.push_str(&piece);
            }
            Ok(out)
        })
        .await??;
        Ok(text)
    }

    /// Tear down workers and release native resources.
    ///
    /// **Concurrent dispose**: the first call does the work; any
    /// concurrent / subsequent call observes `disposed == true` and
    /// returns immediately WITHOUT waiting for the first call to
    /// finish. This matters for the singleton dispose-then-replace
    /// pattern: the new instance is independent (separate models,
    /// separate workers), but the old instance's worker join threads
    /// may still be running in the background when the second
    /// `dispose().await` returns. If you need strong sequencing,
    /// await all calls in a single task instead of racing them.
    async fn dispose(&self) {
        if self.disposed.swap(true, Ordering::AcqRel) {
            return;
        }

        // Drain in-flight ops.
        let start = Instant::now();
        let drain_timeout = Duration::from_secs(30);
        while self.in_flight.count() > 0 {
            if start.elapsed() > drain_timeout {
                tracing::warn!(
                    "llm dispose: {} in-flight op(s) did not drain in 30s; proceeding",
                    self.in_flight.count()
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Tear down pools. We need exclusive ownership to call join_blocking
        // on each worker. If something still holds an Arc<EmbedPool>, log
        // and let the OS clean up — better than blocking dispose forever.
        if let Some(pool_arc) = self.embed_pool.swap(None) {
            match Arc::try_unwrap(pool_arc) {
                Ok(pool) => {
                    for (i, worker) in pool.into_workers().into_iter().enumerate() {
                        join_embed_worker_with_timeout(
                            worker,
                            i,
                            Duration::from_secs(2),
                        )
                        .await;
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "llm dispose: embed pool still has Arc holders; workers will be \
                         cleaned up by Drop instead of explicit join"
                    );
                }
            }
        }
        if let Some(pool_arc) = self.rerank_pool.swap(None) {
            match Arc::try_unwrap(pool_arc) {
                Ok(pool) => {
                    for (i, worker) in pool.into_workers().into_iter().enumerate() {
                        join_rerank_worker_with_timeout(
                            worker,
                            i,
                            Duration::from_secs(2),
                        )
                        .await;
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "llm dispose: rerank pool still has Arc holders; workers will be \
                         cleaned up by Drop instead of explicit join"
                    );
                }
            }
        }

        // Drop model refs. Each ArcSwapOption::swap returns the previous
        // value, which we then drop. If other Arc holders exist, the
        // model lives until they drop too.
        self.embed_model.store(None);
        self.generate_model.store(None);
        self.rerank_model.store(None);
    }
}

// =============================================================================
// Helpers
// =============================================================================

const DEFAULT_EMBED_PARALLELISM: usize = 2;
const DEFAULT_RERANK_PARALLELISM: usize = 2;

fn resolve_pool_parallelism(
    config_value: Option<usize>,
    env_var: &str,
    default: usize,
) -> usize {
    if let Some(n) = config_value {
        return n.max(1);
    }
    let env_value = std::env::var(env_var).ok();
    gpu::resolve_safe_parallelism(gpu::ParallelismOptions {
        env_value: env_value.as_deref(),
        computed: default,
        serialize_windows_cuda: gpu::windows_cuda_serialization_required(),
    })
}

fn format_for_embedding(text: &str, opts: &EmbedOptions, embed_model_uri: &str) -> String {
    if opts.is_query {
        format::format_query_for_embedding(text, embed_model_uri)
    } else {
        format::format_doc_for_embedding(text, opts.title.as_deref(), embed_model_uri)
    }
}

fn make_model_params() -> LlamaModelParams {
    let mode = gpu::resolve_llama_gpu_mode_from_env();
    let n_gpu_layers = match mode {
        gpu::LlamaGpuMode::Off => 0,
        // Match upstream `simple` example: 1000 means "offload everything
        // we can". llama.cpp clamps to the real layer count internally.
        gpu::LlamaGpuMode::Auto => 1000,
    };
    LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers)
}

/// Tokenize-decode-sample loop for a chat-formatted prompt. Builds the
/// chat template from `messages`, then walks the model token-by-token
/// until EOG or `max_new_tokens`. Stays entirely synchronous — caller
/// must wrap in `tokio::task::spawn_blocking`.
fn run_chat_decode(
    backend: &'static LlamaBackend,
    model: &LlamaModel,
    messages: &[LlamaChatMessage],
    n_ctx: usize,
    max_new_tokens: i32,
    sampler: &mut LlamaSampler,
) -> Result<String> {
    let template = model
        .chat_template(None)
        .map_err(|e| Error::ChatTemplate(format!("chat_template(None): {e}")))?;
    let prompt = model
        .apply_chat_template(&template, messages, /* add_ass */ true)
        .map_err(|e| Error::ChatTemplate(format!("apply_chat_template: {e}")))?;

    // Decoder context: llama.cpp chunks prefill across multiple decodes if
    // n_ubatch < prompt length, so the encoder assertion that triggers on
    // the embed/rerank pools doesn't fire here. We still pin
    // n_batch = n_ubatch = n_ctx for two reasons: defense (if a future
    // encoder-decoder generate path is added it inherits a safe ceiling),
    // and uniform shape across all 4 context init sites in this crate.
    let n_ctx_u32 = n_ctx.max(1) as u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(n_ctx_u32).expect("non-zero")))
        .with_n_batch(n_ctx_u32)
        .with_n_ubatch(n_ctx_u32);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| Error::Llama(format!("ctx init: {e}")))?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .map_err(|e| Error::Tokenize(format!("str_to_token: {e}")))?;
    if tokens.is_empty() {
        return Err(Error::Tokenize("empty prompt tokenization".into()));
    }

    let batch_capacity = tokens.len().max(64);
    let mut batch = LlamaBatch::new(batch_capacity, 1);
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        batch
            .add(*token, i, &[0], i == last_idx)
            .map_err(|e| Error::Llama(format!("batch.add (prime): {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| Error::Llama(format!("initial ctx.decode: {e}")))?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    for _ in 0..max_new_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, /* special */ false, /* lstrip */ None)
            .unwrap_or_else(|_| String::from("\u{FFFD}"));
        output.push_str(&piece);
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| Error::Llama(format!("batch.add (loop): {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| Error::Llama(format!("loop ctx.decode: {e}")))?;
    }
    Ok(output)
}

async fn join_embed_worker_with_timeout(worker: EmbedWorker, index: usize, timeout: Duration) {
    let join = tokio::task::spawn_blocking(move || worker.join_blocking());
    if tokio::time::timeout(timeout, join).await.is_err() {
        tracing::warn!(
            "llm dispose: embed worker #{index} did not join within {timeout:?}; \
             leaking the thread"
        );
    }
}

async fn join_rerank_worker_with_timeout(worker: RerankWorker, index: usize, timeout: Duration) {
    let join = tokio::task::spawn_blocking(move || worker.join_blocking());
    if tokio::time::timeout(timeout, join).await.is_err() {
        tracing::warn!(
            "llm dispose: rerank worker #{index} did not join within {timeout:?}; \
             leaking the thread"
        );
    }
}

/// Convert per-chunk `Result<Vec<Option<T>>>` from
/// [`EmbedPool::scatter`] / [`RerankPool::scatter`] into a flat
/// `Vec<Option<T>>` in input order. A chunk-level error becomes
/// `vec![None; chunk_size]` so the output slot count matches the
/// original input count exactly — preserving the
/// `input[i] ↔ output[i]` invariant the `Llm::embed_batch` /
/// `Llm::rerank` contracts promise.
fn flatten_with_chunk_fallback<T>(
    raw: Vec<Result<Vec<Option<T>>>>,
    chunk_sizes: &[usize],
) -> Vec<Option<T>> {
    debug_assert_eq!(raw.len(), chunk_sizes.len());
    raw.into_iter()
        .zip(chunk_sizes.iter().copied())
        .flat_map(|(result, size)| match result {
            Ok(slots) => slots,
            Err(e) => {
                tracing::warn!(
                    "llm: chunk of {size} input(s) failed and will be reported as \
                     None slots: {e}"
                );
                // `vec![None; size]` would require `T: Clone` on the
                // generic; constructing via repeat-with avoids that.
                std::iter::repeat_with(|| None).take(size).collect()
            }
        })
        .collect()
}

/// Wrap the raw `Option<Vec<f32>>` slots from
/// [`EmbedPool::scatter`] into `Option<EmbeddingResult>` with the
/// model name attached. Extracted from `embed_batch` so the
/// flatten-and-wrap step can be tested in isolation without a worker
/// pool — see `tests` module below.
fn embed_results_to_output(
    raw: Vec<Result<Vec<Option<Vec<f32>>>>>,
    chunk_sizes: &[usize],
    model_name: &str,
) -> Vec<Option<EmbeddingResult>> {
    flatten_with_chunk_fallback(raw, chunk_sizes)
        .into_iter()
        .map(|opt| {
            opt.map(|embedding| EmbeddingResult {
                embedding,
                model: model_name.to_owned(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_results_to_output_preserves_input_order() {
        // Two chunks of 2 inputs each, second chunk has a None slot.
        let raw = vec![
            Ok(vec![Some(vec![1.0]), Some(vec![2.0])]),
            Ok(vec![None, Some(vec![4.0])]),
        ];
        let out = embed_results_to_output(raw, &[2, 2], "test-model");
        assert_eq!(out.len(), 4, "output slot count must match input slot count");
        assert_eq!(out[0].as_ref().unwrap().embedding, vec![1.0]);
        assert_eq!(out[1].as_ref().unwrap().embedding, vec![2.0]);
        assert!(out[2].is_none(), "None slot must survive at index 2");
        assert_eq!(out[3].as_ref().unwrap().embedding, vec![4.0]);
        for slot in out.iter().flatten() {
            assert_eq!(slot.model, "test-model");
        }
    }

    #[test]
    fn embed_results_to_output_pads_failed_chunks_with_none() {
        // First chunk succeeded; second chunk failed entirely. Failed
        // chunk's slots must become `None`s of the same count.
        let raw = vec![
            Ok(vec![Some(vec![1.0]), Some(vec![2.0])]),
            Err(Error::WorkerClosed),
        ];
        let out = embed_results_to_output(raw, &[2, 3], "m");
        assert_eq!(out.len(), 5, "padded output must equal sum of chunk_sizes");
        assert_eq!(out[0].as_ref().unwrap().embedding, vec![1.0]);
        assert_eq!(out[1].as_ref().unwrap().embedding, vec![2.0]);
        assert!(out[2].is_none(), "failed chunk slot 0");
        assert!(out[3].is_none(), "failed chunk slot 1");
        assert!(out[4].is_none(), "failed chunk slot 2");
    }

    #[test]
    fn flatten_with_chunk_fallback_handles_scores() {
        let raw: Vec<Result<Vec<Option<f32>>>> = vec![
            Ok(vec![Some(0.9), Some(0.1)]),
            Err(Error::WorkerClosed),
            Ok(vec![Some(0.5)]),
        ];
        let out = flatten_with_chunk_fallback(raw, &[2, 1, 1]);
        assert_eq!(out, vec![Some(0.9), Some(0.1), None, Some(0.5)]);
    }
}
