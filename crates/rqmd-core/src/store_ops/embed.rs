//! `generate_embeddings`: scoped LLM session, doc batching, per-doc token
//! chunking, sub-batch embed with fallback, incomplete cleanup.
//!
//! Port of `tobi/qmd/src/store.ts` lines 1511–1698. See the plan file for
//! the edge-case table.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::store::chunking::ChunkStrategy;
use crate::store::documents::extract_title;
use crate::store::embeddings::{
    PendingEmbeddingDoc, clear_all_embeddings, get_embedding_docs_for_batch,
    get_pending_embedding_docs, insert_embedding, remove_incomplete_embeddings,
};
use crate::store::path::now_rfc3339;
use crate::store::{
    CHUNK_SIZE_TOKENS, DEFAULT_EMBED_MAX_BATCH_BYTES, DEFAULT_EMBED_MAX_DOCS_PER_BATCH, Store,
};

use crate::llm::config::resolve_embed_model;
use crate::llm::format::{embedding_fingerprint, format_doc_for_embedding};
use crate::llm::session::{LlmSession, LlmSessionOptions};
use crate::llm::traits::Llm;
use crate::llm::types::EmbedOptions as LlmEmbedOptions;

use super::chunk_tokens::chunk_document_by_tokens;
use super::{Error, Result};

const SUB_BATCH: usize = 32;
const SESSION_MAX_DURATION: Duration = Duration::from_secs(30 * 60);
/// Drain the retry queue once after this many unrelated chunks succeed.
/// Mirrors qmd `RETRY_AFTER_SUCCESSFUL_CHUNKS` (`store.ts:1612`).
const RETRY_AFTER_SUCCESSFUL_CHUNKS: usize = 64;
/// Per-chunk retry cap; prevents endless loops on permanently bad chunks.
/// Mirrors qmd `MAX_RETRY_ATTEMPTS` (`store.ts:1613`).
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Caller-provided options for [`generate_embeddings`].
///
/// Default (`EmbedOptions::default()`) embeds every pending document in
/// every collection using the configured embed model, with `force=false`
/// and no progress callback.
#[derive(Clone, Default)]
pub struct EmbedOptions {
    /// Drop all existing embeddings for the scope before re-embedding.
    pub force: bool,
    /// Override the embed model URI. Default: [`resolve_embed_model`].
    pub model: Option<String>,
    /// Restrict to one collection. Default: all collections.
    pub collection: Option<String>,
    /// Document-count cap per batch. Default
    /// [`DEFAULT_EMBED_MAX_DOCS_PER_BATCH`].
    pub max_docs_per_batch: Option<usize>,
    /// Byte cap per batch. Default [`DEFAULT_EMBED_MAX_BATCH_BYTES`].
    pub max_batch_bytes: Option<usize>,
    /// Chunking strategy. Default `ChunkStrategy::Auto`.
    pub chunk_strategy: Option<ChunkStrategy>,
    /// Progress callback, invoked after each sub-batch and at end of
    /// each doc-batch.
    pub on_progress: Option<Arc<dyn Fn(EmbedProgress) + Send + Sync>>,
}

impl std::fmt::Debug for EmbedOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedOptions")
            .field("force", &self.force)
            .field("model", &self.model)
            .field("collection", &self.collection)
            .field("max_docs_per_batch", &self.max_docs_per_batch)
            .field("max_batch_bytes", &self.max_batch_bytes)
            .field("chunk_strategy", &self.chunk_strategy)
            .field("on_progress", &self.on_progress.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

/// Snapshot delivered via [`EmbedOptions::on_progress`]. `chunks_embedded`,
/// `total_chunks` and `bytes_processed` are monotonically non-decreasing
/// within a call; `errors` is the *active* failure count and may decrease
/// when a retry recovers a previously-failed chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedProgress {
    pub chunks_embedded: usize,
    pub total_chunks: usize,
    pub bytes_processed: usize,
    pub total_bytes: usize,
    /// Chunks still unresolved after retries (= `failures.len()`).
    pub errors: usize,
}

/// A chunk that did not embed successfully and is still unresolved after
/// retries. Port of qmd `EmbedFailure` (`store.ts:1374`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedFailure {
    pub path: String,
    pub hash: String,
    pub seq: i64,
    pub attempts: u32,
    pub reason: String,
}

/// Summary returned by [`generate_embeddings`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedResult {
    pub docs_processed: usize,
    pub chunks_embedded: usize,
    /// Chunks still failed after retries (= active failure count).
    pub errors: usize,
    /// Detail of each still-failed chunk (drives the CLI failure report).
    pub failures: Vec<EmbedFailure>,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
struct ChunkItem {
    hash: String,
    path: String,
    title: String,
    text: String,
    seq: i64,
    pos: i64,
    bytes: usize,
    /// Total chunks for the owning document. Carried on the item (not looked
    /// up per doc-batch) because retries can fire in a *later* doc-batch where
    /// the per-batch `expected_chunks` map no longer holds this hash. Mirrors
    /// qmd `chunk.expectedTotalChunks` (`store.ts:1434`).
    expected_total_chunks: i64,
}

#[derive(Default)]
struct Counters {
    chunks_embedded: usize,
    total_chunks: usize,
    bytes_processed: usize,
}

/// Outcome of [`run_inner`]: success count plus the unresolved-failure view.
struct RunOutcome {
    chunks_embedded: usize,
    errors: usize,
    failures: Vec<EmbedFailure>,
}

fn chunk_key(chunk: &ChunkItem) -> String {
    format!("{}:{}", chunk.hash, chunk.seq)
}

/// Stringify an error for a failure record, truncating to 180 chars
/// (177 + "...") like qmd `reasonFromError` (`store.ts:1620`). Char-based,
/// not byte- or UTF-16-based: boundary-safe (never panics on multibyte) and
/// identical to qmd for ASCII messages (the common case). The char-vs-UTF-16
/// boundary difference for non-ASCII text is a sanctioned micro-divergence.
fn reason_from_error<E: std::fmt::Display>(e: &E) -> String {
    let raw = e.to_string();
    if raw.chars().count() > 180 {
        let truncated: String = raw.chars().take(177).collect();
        format!("{truncated}...")
    } else {
        raw
    }
}

/// Per-chunk failure tracking + bounded retry queue, shared across all
/// doc-batches of a single [`generate_embeddings`] run. Port of qmd's
/// `failures`/`retryQueue` closures (`store.ts:1614-1682`).
///
/// `errors` reported to callers is `failures.len()` — the count of chunks
/// still unresolved *after* retries, not a monotonic attempt counter. A chunk
/// that fails then recovers leaves no error; a chunk that fails three times
/// counts once.
#[derive(Default)]
struct ResilienceState {
    /// key = "hash:seq"
    failures: HashMap<String, EmbedFailure>,
    /// key = "hash:seq"
    retry_queue: HashMap<String, ChunkItem>,
    successes_since_retry: usize,
}

impl ResilienceState {
    fn active_error_count(&self) -> usize {
        self.failures.len()
    }

    /// Snapshot of outstanding failures, sorted by `(path, seq)` for stable,
    /// readable output. (qmd relies on JS `Map` insertion order; rqmd's
    /// `HashMap` is unordered, so we sort to keep the CLI's "first 8" and any
    /// tests deterministic.)
    fn failure_list(&self) -> Vec<EmbedFailure> {
        let mut out: Vec<EmbedFailure> = self.failures.values().cloned().collect();
        out.sort_by(|a, b| a.path.cmp(&b.path).then(a.seq.cmp(&b.seq)));
        out
    }

    fn record_failure(&mut self, chunk: &ChunkItem, reason: String) {
        let key = chunk_key(chunk);
        let attempts = self.failures.get(&key).map(|f| f.attempts).unwrap_or(0) + 1;
        self.failures.insert(
            key.clone(),
            EmbedFailure {
                path: chunk.path.clone(),
                hash: chunk.hash.clone(),
                seq: chunk.seq,
                attempts,
                reason,
            },
        );
        self.retry_queue.insert(key, chunk.clone());
    }

    fn clear_failure(&mut self, chunk: &ChunkItem) {
        let key = chunk_key(chunk);
        self.failures.remove(&key);
        self.retry_queue.remove(&key);
    }

    /// Embed one chunk individually. On success inserts the vector and clears
    /// any prior failure; on failure records the reason (never propagates the
    /// embed error). DB-insert errors *do* propagate (`?`) — a write failure
    /// is systemic, not a transient per-chunk condition. Port of qmd
    /// `tryEmbedChunk` (`store.ts:1642`).
    #[allow(clippy::too_many_arguments)]
    async fn try_embed_chunk(
        &mut self,
        store: &mut Store,
        session: &Arc<LlmSession>,
        chunk: &ChunkItem,
        model: &str,
        fingerprint: &str,
        now: &str,
        chunks_embedded: &mut usize,
    ) -> Result<bool> {
        let text = format_doc_for_embedding(&chunk.text, Some(&chunk.title), model);
        let opts = LlmEmbedOptions {
            model: Some(model.into()),
            is_query: false,
            title: None,
        };
        match session.embed(&text, opts).await {
            Ok(Some(emb)) => {
                store.with_connection_mut(|c| {
                    insert_embedding(
                        c,
                        &chunk.hash,
                        chunk.seq,
                        chunk.pos,
                        &emb.embedding,
                        model,
                        fingerprint,
                        now,
                        chunk.expected_total_chunks,
                    )
                })?;
                *chunks_embedded += 1;
                self.successes_since_retry += 1;
                self.clear_failure(chunk);
                Ok(true)
            }
            Ok(None) => {
                self.record_failure(chunk, "embedding returned no vector".to_string());
                Ok(false)
            }
            Err(e) => {
                self.record_failure(chunk, reason_from_error(&e));
                Ok(false)
            }
        }
    }

    /// Drain the retry queue. Normal mode (`force=false`): one pass, and only
    /// once at least `RETRY_AFTER_SUCCESSFUL_CHUNKS` unrelated chunks have
    /// succeeded. Force mode: keep retrying until every outstanding failure
    /// recovers or hits `MAX_RETRY_ATTEMPTS`. Port of qmd `retryFailedChunks`
    /// (`store.ts:1660`).
    #[allow(clippy::too_many_arguments)]
    async fn retry_failed_chunks(
        &mut self,
        store: &mut Store,
        session: &Arc<LlmSession>,
        model: &str,
        fingerprint: &str,
        now: &str,
        chunks_embedded: &mut usize,
        force: bool,
    ) -> Result<()> {
        if !session.is_valid() || self.retry_queue.is_empty() {
            return Ok(());
        }
        if !force && self.successes_since_retry < RETRY_AFTER_SUCCESSFUL_CHUNKS {
            return Ok(());
        }
        self.successes_since_retry = 0;

        loop {
            // Re-snapshot each pass: try_embed_chunk mutates retry_queue, and
            // qmd re-spreads `[...retryQueue]` every iteration (store.ts:1668).
            let snapshot: Vec<(String, ChunkItem)> = self
                .retry_queue
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let mut retried = 0usize;
            for (key, chunk) in snapshot {
                let retryable = self
                    .failures
                    .get(&key)
                    .is_some_and(|f| f.attempts < MAX_RETRY_ATTEMPTS);
                if !retryable {
                    continue;
                }
                retried += 1;
                self.try_embed_chunk(
                    store,
                    session,
                    &chunk,
                    model,
                    fingerprint,
                    now,
                    chunks_embedded,
                )
                .await?;
            }
            if !force || retried == 0 || !session.is_valid() {
                break;
            }
            let any_retryable = self.retry_queue.keys().any(|key| {
                self.failures
                    .get(key)
                    .is_some_and(|f| f.attempts < MAX_RETRY_ATTEMPTS)
            });
            if !any_retryable {
                break;
            }
        }
        Ok(())
    }
}

/// Generate vector embeddings for documents that need them.
///
/// Composes:
/// * [`get_pending_embedding_docs`] → `build_embedding_batches`
/// * scoped [`LlmSession`] with 30-minute max duration
/// * [`chunk_document_by_tokens`] per doc
/// * sub-batches of 32 chunks through [`Llm::embed_batch`], falling back
///   to one-at-a-time `embed` on batch failure
/// * [`insert_embedding`] per success, [`remove_incomplete_embeddings`]
///   at end of each doc-batch
///
/// See the plan file's "edge cases" table for behaviour around empty
/// collections, `force=true`, session timeout, error-rate >80%, and
/// dimension mismatch.
pub async fn generate_embeddings(
    store: &mut Store,
    llm: Arc<dyn Llm>,
    options: EmbedOptions,
) -> Result<EmbedResult> {
    let start = Instant::now();

    // Sanity check: the chunker emits chunks up to CHUNK_SIZE_TOKENS, which
    // must fit inside a single embed-pool ubatch. The embed pool pins
    // n_ubatch = embed_context_size (see `worker::make_pool_ctx_params`), so
    // a context smaller than CHUNK_SIZE_TOKENS will either truncate inputs
    // or hit the encoder assertion. Warn instead of erroring — users may
    // intentionally shrink the context for short-text corpora.
    let embed_ctx = llm.embed_context_size();
    if embed_ctx < CHUNK_SIZE_TOKENS {
        tracing::warn!(
            "embed_context_size ({}) < CHUNK_SIZE_TOKENS ({}); large chunks will hit the \
             encoder assertion or be rejected. Set QMD_EMBED_CONTEXT_SIZE to at least {}.",
            embed_ctx,
            CHUNK_SIZE_TOKENS,
            CHUNK_SIZE_TOKENS,
        );
    }

    let model = options
        .model
        .clone()
        .unwrap_or_else(|| resolve_embed_model(None));
    // Pure over (model, templates, chunk constants) — constant for the whole
    // run, so compute once and thread it down rather than per chunk.
    let fingerprint = embedding_fingerprint(&model);
    let collection_owned = options.collection.clone();
    let collection = collection_owned.as_deref();
    let max_docs_per_batch = options
        .max_docs_per_batch
        .unwrap_or(DEFAULT_EMBED_MAX_DOCS_PER_BATCH);
    let max_batch_bytes = options
        .max_batch_bytes
        .unwrap_or(DEFAULT_EMBED_MAX_BATCH_BYTES);

    // Mirror TS `generateEmbeddings`: reject non-positive batch limits up
    // front rather than silently looping forever on a zero budget.
    if max_docs_per_batch == 0 {
        return Err(Error::EmbedFailed(
            "maxDocsPerBatch must be greater than 0".into(),
        ));
    }
    if max_batch_bytes == 0 {
        return Err(Error::EmbedFailed(
            "maxBatchBytes must be greater than 0".into(),
        ));
    }

    if options.force {
        store.with_connection_mut(|c| clear_all_embeddings(c, collection))?;
    }

    let docs = store
        .with_connection(|c| get_pending_embedding_docs(c, collection, &model, &fingerprint))?;
    if docs.is_empty() {
        return Ok(EmbedResult {
            docs_processed: 0,
            chunks_embedded: 0,
            errors: 0,
            failures: Vec::new(),
            duration_ms: start.elapsed().as_millis(),
        });
    }
    let total_bytes: usize = docs.iter().map(|d| d.bytes).sum();
    let total_docs = docs.len();
    let now = now_rfc3339();
    let chunk_strategy = options.chunk_strategy.unwrap_or(ChunkStrategy::Auto);
    let on_progress = options.on_progress.clone();

    // Manual session lifecycle (instead of `with_llm_session`) so the inner
    // closure can return `store_ops::Result` directly without an extra
    // `From` impl on `crate::llm::Error`.
    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            max_duration: Some(SESSION_MAX_DURATION),
            name: Some("generateEmbeddings".into()),
        },
    );
    let outcome = run_inner(
        store,
        session.clone(),
        &model,
        &fingerprint,
        &now,
        max_docs_per_batch,
        max_batch_bytes,
        total_bytes,
        chunk_strategy,
        on_progress.as_ref(),
        docs,
    )
    .await;
    session.release();
    let outcome = outcome?;

    Ok(EmbedResult {
        docs_processed: total_docs,
        chunks_embedded: outcome.chunks_embedded,
        errors: outcome.errors,
        failures: outcome.failures,
        duration_ms: start.elapsed().as_millis(),
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_inner(
    store: &mut Store,
    session: Arc<LlmSession>,
    model: &str,
    fingerprint: &str,
    now: &str,
    max_docs_per_batch: usize,
    max_batch_bytes: usize,
    total_bytes: usize,
    chunk_strategy: ChunkStrategy,
    on_progress: Option<&Arc<dyn Fn(EmbedProgress) + Send + Sync>>,
    docs: Vec<PendingEmbeddingDoc>,
) -> Result<RunOutcome> {
    let llm_arc: Arc<dyn Llm> = session.clone();
    let mut counters = Counters::default();
    // Per-chunk failures + retry queue, shared across all doc-batches.
    let mut state = ResilienceState::default();
    let mut vec_initialized = false;

    for batch_meta in build_embedding_batches(&docs, max_docs_per_batch, max_batch_bytes) {
        if !session.is_valid() {
            tracing::warn!("Session expired — skipping remaining document batches");
            break;
        }
        let batch_bytes: usize = batch_meta.iter().map(|d| d.bytes).sum();
        let batch_docs = store.with_connection(|c| get_embedding_docs_for_batch(c, &batch_meta))?;

        // Build chunk items.
        let mut chunk_items: Vec<ChunkItem> = Vec::new();
        let mut expected_chunks: HashMap<String, i64> = HashMap::new();

        for doc in &batch_docs {
            if doc.body.trim().is_empty() {
                continue;
            }
            let title = extract_title(&doc.body, &doc.path);
            let chunks = chunk_document_by_tokens(
                llm_arc.clone(),
                &doc.body,
                None,
                None,
                None,
                Some(&doc.path),
                chunk_strategy,
                Some(session.signal()),
            )
            .await?;
            let n_chunks = chunks.len() as i64;
            expected_chunks.insert(doc.hash.clone(), n_chunks);
            for (seq, chunk) in chunks.into_iter().enumerate() {
                let bytes = chunk.text.len();
                chunk_items.push(ChunkItem {
                    hash: doc.hash.clone(),
                    path: doc.path.clone(),
                    title: title.clone(),
                    text: chunk.text,
                    seq: seq as i64,
                    pos: chunk.pos as i64,
                    bytes,
                    expected_total_chunks: n_chunks,
                });
            }
        }
        counters.total_chunks += chunk_items.len();

        if chunk_items.is_empty() {
            counters.bytes_processed += batch_bytes;
            fire_progress(
                on_progress,
                &counters,
                state.active_error_count(),
                total_bytes,
            );
            continue;
        }

        // First-time vec table provisioning: probe dimension via a single
        // embed of chunk[0]. The result is NOT inserted — the inner sub-batch
        // loop below starts at index 0 and re-embeds it. Matches TS exactly;
        // the duplication is deliberate so we avoid a special insert path.
        if !vec_initialized {
            let first = &chunk_items[0];
            let formatted = format_doc_for_embedding(&first.text, Some(&first.title), model);
            let probe = session
                .embed(
                    &formatted,
                    LlmEmbedOptions {
                        model: Some(model.into()),
                        is_query: false,
                        title: Some(first.title.clone()),
                    },
                )
                .await?
                .ok_or_else(|| Error::EmbedFailed("first chunk returned None".into()))?;
            store.ensure_vec_table(probe.embedding.len())?;
            vec_initialized = true;
        }

        let total_batch_chunk_bytes: usize = chunk_items.iter().map(|c| c.bytes).sum();
        let mut batch_chunk_bytes_processed = 0usize;

        let mut batch_start = 0usize;
        while batch_start < chunk_items.len() {
            if !session.is_valid() {
                let remaining = &chunk_items[batch_start..];
                tracing::warn!(
                    "Session expired — skipping {} remaining chunks",
                    remaining.len()
                );
                for chunk in remaining {
                    state.record_failure(
                        chunk,
                        "LLM session expired before embedding chunk".to_string(),
                    );
                }
                break;
            }
            let processed = counters.chunks_embedded + state.active_error_count();
            if processed >= SUB_BATCH
                && (state.active_error_count() as f64) > (processed as f64) * 0.8
            {
                // Record-then-warn order mirrors qmd (`store.ts:1756`): the
                // logged count includes the just-aborted remaining chunks.
                for chunk in &chunk_items[batch_start..] {
                    state.record_failure(
                        chunk,
                        "embedding aborted because error rate was too high".to_string(),
                    );
                }
                tracing::warn!(
                    "Error rate too high ({}/{}) — aborting sub-batch",
                    state.active_error_count(),
                    processed
                );
                break;
            }
            let batch_end = (batch_start + SUB_BATCH).min(chunk_items.len());
            let sub = &chunk_items[batch_start..batch_end];
            let texts: Vec<String> = sub
                .iter()
                .map(|c| format_doc_for_embedding(&c.text, Some(&c.title), model))
                .collect();
            let opts = LlmEmbedOptions {
                model: Some(model.into()),
                is_query: false,
                title: None,
            };

            match session.embed_batch(&texts, opts).await {
                Ok(embeddings) => {
                    for (i, chunk) in sub.iter().enumerate() {
                        if let Some(Some(emb)) = embeddings.get(i) {
                            store.with_connection_mut(|c| {
                                insert_embedding(
                                    c,
                                    &chunk.hash,
                                    chunk.seq,
                                    chunk.pos,
                                    &emb.embedding,
                                    model,
                                    fingerprint,
                                    now,
                                    chunk.expected_total_chunks,
                                )
                            })?;
                            counters.chunks_embedded += 1;
                            state.successes_since_retry += 1;
                            state.clear_failure(chunk);
                        } else {
                            state.record_failure(
                                chunk,
                                "batch embedding returned no vector".to_string(),
                            );
                        }
                        batch_chunk_bytes_processed += chunk.bytes;
                    }
                    state
                        .retry_failed_chunks(
                            store,
                            &session,
                            model,
                            fingerprint,
                            now,
                            &mut counters.chunks_embedded,
                            false,
                        )
                        .await?;
                }
                Err(e) => {
                    // Batch failed — fall back to individual embeds. A recovered
                    // chunk clears its prior failure, so the error count reflects
                    // only outstanding failures.
                    if !session.is_valid() {
                        let reason = reason_from_error(&e);
                        for chunk in sub {
                            state.record_failure(
                                chunk,
                                format!("batch failed and session expired: {reason}"),
                            );
                        }
                        batch_chunk_bytes_processed += sub.iter().map(|c| c.bytes).sum::<usize>();
                    } else {
                        for chunk in sub {
                            state
                                .try_embed_chunk(
                                    store,
                                    &session,
                                    chunk,
                                    model,
                                    fingerprint,
                                    now,
                                    &mut counters.chunks_embedded,
                                )
                                .await?;
                            batch_chunk_bytes_processed += chunk.bytes;
                            state
                                .retry_failed_chunks(
                                    store,
                                    &session,
                                    model,
                                    fingerprint,
                                    now,
                                    &mut counters.chunks_embedded,
                                    false,
                                )
                                .await?;
                        }
                    }
                }
            }

            let proportional = if total_batch_chunk_bytes == 0 {
                batch_bytes
            } else {
                ((batch_chunk_bytes_processed as f64 / total_batch_chunk_bytes as f64)
                    * batch_bytes as f64)
                    .round() as usize
            };
            let snapshot_bytes = counters.bytes_processed + proportional.min(batch_bytes);
            let snapshot = EmbedProgress {
                chunks_embedded: counters.chunks_embedded,
                total_chunks: counters.total_chunks,
                bytes_processed: snapshot_bytes,
                total_bytes,
                errors: state.active_error_count(),
            };
            if let Some(cb) = on_progress {
                cb(snapshot);
            }
            batch_start = batch_end;
        }

        // Forced drain of the retry queue before cleanup, so chunks that recover
        // here complete their document and survive remove_incomplete_embeddings.
        state
            .retry_failed_chunks(
                store,
                &session,
                model,
                fingerprint,
                now,
                &mut counters.chunks_embedded,
                true,
            )
            .await?;

        let removed = store
            .with_connection_mut(|c| remove_incomplete_embeddings(c, &expected_chunks, model))?;
        if removed > 0 {
            counters.chunks_embedded = counters.chunks_embedded.saturating_sub(removed as usize);
        }
        counters.bytes_processed += batch_bytes;
        fire_progress(
            on_progress,
            &counters,
            state.active_error_count(),
            total_bytes,
        );
    }

    Ok(RunOutcome {
        chunks_embedded: counters.chunks_embedded,
        errors: state.active_error_count(),
        failures: state.failure_list(),
    })
}

fn fire_progress(
    on_progress: Option<&Arc<dyn Fn(EmbedProgress) + Send + Sync>>,
    counters: &Counters,
    errors: usize,
    total_bytes: usize,
) {
    if let Some(cb) = on_progress {
        cb(EmbedProgress {
            chunks_embedded: counters.chunks_embedded,
            total_chunks: counters.total_chunks,
            bytes_processed: counters.bytes_processed,
            total_bytes,
            errors,
        });
    }
}

/// Group pending docs into batches bounded by both doc-count and byte
/// totals. Mirrors TS `buildEmbeddingBatches` (`store.ts:1458`).
fn build_embedding_batches(
    docs: &[PendingEmbeddingDoc],
    max_docs_per_batch: usize,
    max_batch_bytes: usize,
) -> Vec<Vec<PendingEmbeddingDoc>> {
    let mut batches: Vec<Vec<PendingEmbeddingDoc>> = Vec::new();
    let mut current: Vec<PendingEmbeddingDoc> = Vec::new();
    let mut current_bytes = 0usize;

    for doc in docs {
        let doc_bytes = doc.bytes;
        let would_exceed_docs = current.len() >= max_docs_per_batch;
        let would_exceed_bytes =
            !current.is_empty() && current_bytes.saturating_add(doc_bytes) > max_batch_bytes;
        if would_exceed_docs || would_exceed_bytes {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(doc.clone());
        current_bytes += doc_bytes;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_embedding_batches_respects_doc_count() {
        let docs: Vec<PendingEmbeddingDoc> = (0..5)
            .map(|i| PendingEmbeddingDoc {
                hash: format!("h{i}"),
                path: format!("p{i}"),
                bytes: 10,
            })
            .collect();
        let b = build_embedding_batches(&docs, 2, 1_000_000);
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].len(), 2);
        assert_eq!(b[1].len(), 2);
        assert_eq!(b[2].len(), 1);
    }

    #[test]
    fn build_embedding_batches_respects_byte_budget() {
        let docs: Vec<PendingEmbeddingDoc> = (0..3)
            .map(|i| PendingEmbeddingDoc {
                hash: format!("h{i}"),
                path: format!("p{i}"),
                bytes: 100,
            })
            .collect();
        let b = build_embedding_batches(&docs, 64, 150);
        // First doc fits, second pushes a new batch (would exceed 150B).
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn build_embedding_batches_keeps_one_oversized_doc_alone() {
        let docs = vec![PendingEmbeddingDoc {
            hash: "h0".into(),
            path: "p0".into(),
            bytes: 1_000,
        }];
        let b = build_embedding_batches(&docs, 64, 100);
        // `current.is_empty()` short-circuit ensures we always place at least
        // one doc per batch — single oversized doc gets its own batch.
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].len(), 1);
    }

    fn chunk(hash: &str, seq: i64) -> ChunkItem {
        ChunkItem {
            hash: hash.into(),
            path: format!("docs/{hash}.md"),
            title: "T".into(),
            text: "body".into(),
            seq,
            pos: 0,
            bytes: 4,
            expected_total_chunks: 1,
        }
    }

    #[test]
    fn chunk_key_is_hash_colon_seq() {
        assert_eq!(chunk_key(&chunk("abc", 2)), "abc:2");
    }

    #[test]
    fn record_failure_tracks_attempts_and_queues_retry() {
        let mut s = ResilienceState::default();
        let c = chunk("h1", 0);

        s.record_failure(&c, "boom".into());
        assert_eq!(s.active_error_count(), 1);
        assert!(s.retry_queue.contains_key("h1:0"));
        let f = &s.failures["h1:0"];
        assert_eq!(f.attempts, 1);
        assert_eq!(f.path, "docs/h1.md");
        assert_eq!(f.hash, "h1");
        assert_eq!(f.seq, 0);
        assert_eq!(f.reason, "boom");

        // Re-recording the same chunk bumps attempts, keeps a single entry.
        s.record_failure(&c, "boom again".into());
        assert_eq!(s.active_error_count(), 1);
        assert_eq!(s.failures["h1:0"].attempts, 2);
        assert_eq!(s.failures["h1:0"].reason, "boom again");
    }

    #[test]
    fn clear_failure_removes_from_both_maps() {
        let mut s = ResilienceState::default();
        let c = chunk("h1", 0);
        s.record_failure(&c, "boom".into());
        s.clear_failure(&c);
        assert_eq!(s.active_error_count(), 0);
        assert!(s.retry_queue.is_empty());
        assert!(s.failure_list().is_empty());
    }

    #[test]
    fn failure_list_is_sorted_by_path_then_seq() {
        let mut s = ResilienceState::default();
        s.record_failure(&chunk("b", 1), "x".into());
        s.record_failure(&chunk("a", 5), "y".into());
        s.record_failure(&chunk("a", 2), "z".into());
        let list = s.failure_list();
        let keys: Vec<(String, i64)> = list.iter().map(|f| (f.path.clone(), f.seq)).collect();
        assert_eq!(
            keys,
            vec![
                ("docs/a.md".into(), 2),
                ("docs/a.md".into(), 5),
                ("docs/b.md".into(), 1),
            ]
        );
    }

    #[test]
    fn reason_from_error_passes_through_short_messages() {
        let r = reason_from_error(&"short message");
        assert_eq!(r, "short message");
    }

    #[test]
    fn reason_from_error_truncates_long_messages_to_180() {
        let long = "x".repeat(500);
        let r = reason_from_error(&long);
        assert_eq!(r.chars().count(), 180);
        assert!(r.ends_with("..."));
        assert_eq!(r, format!("{}...", "x".repeat(177)));
    }

    #[test]
    fn reason_from_error_at_boundary_180_is_untouched() {
        let exactly = "y".repeat(180);
        assert_eq!(reason_from_error(&exactly), exactly);
    }

    #[test]
    fn reason_from_error_is_char_boundary_safe_on_multibyte() {
        // 200 multibyte chars: must not panic on a non-UTF-8-boundary slice.
        let multibyte = "あ".repeat(200);
        let r = reason_from_error(&multibyte);
        assert_eq!(r.chars().count(), 180);
        assert!(r.ends_with("..."));
    }
}
