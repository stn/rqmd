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
use crate::llm::format::format_doc_for_embedding;
use crate::llm::llama_cpp::LlamaCpp;
use crate::llm::session::{LlmSession, LlmSessionOptions};
use crate::llm::traits::Llm;
use crate::llm::types::EmbedOptions as LlmEmbedOptions;

use super::chunk_tokens::chunk_document_by_tokens;
use super::{Error, Result};

const SUB_BATCH: usize = 32;
const SESSION_MAX_DURATION: Duration = Duration::from_secs(30 * 60);

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

/// Snapshot delivered via [`EmbedOptions::on_progress`]. Counters are
/// monotonically non-decreasing within a single call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedProgress {
    pub chunks_embedded: usize,
    pub total_chunks: usize,
    pub bytes_processed: usize,
    pub total_bytes: usize,
    pub errors: usize,
}

/// Summary returned by [`generate_embeddings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedResult {
    pub docs_processed: usize,
    pub chunks_embedded: usize,
    pub errors: usize,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
struct ChunkItem {
    hash: String,
    title: String,
    text: String,
    seq: i64,
    pos: i64,
    bytes: usize,
}

#[derive(Default)]
struct Counters {
    chunks_embedded: usize,
    errors: usize,
    total_chunks: usize,
    bytes_processed: usize,
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
    llm: Arc<LlamaCpp>,
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

    let docs = store.with_connection(|c| get_pending_embedding_docs(c, collection, &model))?;
    if docs.is_empty() {
        return Ok(EmbedResult {
            docs_processed: 0,
            chunks_embedded: 0,
            errors: 0,
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
        duration_ms: start.elapsed().as_millis(),
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_inner(
    store: &mut Store,
    session: Arc<LlmSession>,
    model: &str,
    now: &str,
    max_docs_per_batch: usize,
    max_batch_bytes: usize,
    total_bytes: usize,
    chunk_strategy: ChunkStrategy,
    on_progress: Option<&Arc<dyn Fn(EmbedProgress) + Send + Sync>>,
    docs: Vec<PendingEmbeddingDoc>,
) -> Result<Counters> {
    let llm_arc: Arc<dyn Llm> = session.clone();
    let mut counters = Counters::default();
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
            expected_chunks.insert(doc.hash.clone(), chunks.len() as i64);
            for (seq, chunk) in chunks.into_iter().enumerate() {
                let bytes = chunk.text.len();
                chunk_items.push(ChunkItem {
                    hash: doc.hash.clone(),
                    title: title.clone(),
                    text: chunk.text,
                    seq: seq as i64,
                    pos: chunk.pos as i64,
                    bytes,
                });
            }
        }
        counters.total_chunks += chunk_items.len();

        if chunk_items.is_empty() {
            counters.bytes_processed += batch_bytes;
            fire_progress(on_progress, &counters, total_bytes);
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
                let remaining = chunk_items.len() - batch_start;
                counters.errors += remaining;
                tracing::warn!("Session expired — skipping {remaining} remaining chunks");
                break;
            }
            let processed = counters.chunks_embedded + counters.errors;
            if processed >= SUB_BATCH && (counters.errors as f64) > (processed as f64) * 0.8 {
                let remaining = chunk_items.len() - batch_start;
                counters.errors += remaining;
                tracing::warn!(
                    "Error rate too high ({}/{}) — aborting sub-batch",
                    counters.errors,
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

            match session.embed_batch(&texts, opts.clone()).await {
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
                                    now,
                                    *expected_chunks.get(&chunk.hash).unwrap_or(&1),
                                )
                            })?;
                            counters.chunks_embedded += 1;
                        } else {
                            counters.errors += 1;
                        }
                        batch_chunk_bytes_processed += chunk.bytes;
                    }
                }
                Err(_) => {
                    if !session.is_valid() {
                        counters.errors += sub.len();
                        batch_chunk_bytes_processed += sub.iter().map(|c| c.bytes).sum::<usize>();
                    } else {
                        for chunk in sub {
                            let text =
                                format_doc_for_embedding(&chunk.text, Some(&chunk.title), model);
                            match session.embed(&text, opts.clone()).await {
                                Ok(Some(emb)) => {
                                    store.with_connection_mut(|c| {
                                        insert_embedding(
                                            c,
                                            &chunk.hash,
                                            chunk.seq,
                                            chunk.pos,
                                            &emb.embedding,
                                            model,
                                            now,
                                            *expected_chunks.get(&chunk.hash).unwrap_or(&1),
                                        )
                                    })?;
                                    counters.chunks_embedded += 1;
                                }
                                _ => counters.errors += 1,
                            }
                            batch_chunk_bytes_processed += chunk.bytes;
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
                errors: counters.errors,
            };
            if let Some(cb) = on_progress {
                cb(snapshot);
            }
            batch_start = batch_end;
        }

        let removed = store
            .with_connection_mut(|c| remove_incomplete_embeddings(c, &expected_chunks, model))?;
        if removed > 0 {
            counters.chunks_embedded = counters.chunks_embedded.saturating_sub(removed as usize);
        }
        counters.bytes_processed += batch_bytes;
        fire_progress(on_progress, &counters, total_bytes);
    }

    Ok(counters)
}

fn fire_progress(
    on_progress: Option<&Arc<dyn Fn(EmbedProgress) + Send + Sync>>,
    counters: &Counters,
    total_bytes: usize,
) {
    if let Some(cb) = on_progress {
        cb(EmbedProgress {
            chunks_embedded: counters.chunks_embedded,
            total_chunks: counters.total_chunks,
            bytes_processed: counters.bytes_processed,
            total_bytes,
            errors: counters.errors,
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
}
