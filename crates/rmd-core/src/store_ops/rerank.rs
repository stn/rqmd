//! `rerank`: per-chunk cached LLM reranking with legacy-key fallback.
//!
//! Port of `tobi/qmd/src/store.ts` lines 3534–3577.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::store::cache::{get_cached_result, set_cached_result};
use crate::store::Store;

use crate::llm::traits::Llm;
use crate::llm::types::{RerankDocument, RerankOptions};

use super::cache_keys::{legacy_rerank_cache_key, rerank_cache_key};
use super::Result;

/// One candidate to rerank: a chunk text bound to its source file. Files
/// repeat naturally — different chunks of the same document score
/// independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankCandidate {
    pub file: String,
    pub text: String,
}

/// Final per-candidate score. Same shape as TS `{ file, score }`.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankScore {
    pub file: String,
    pub score: f32,
}

/// Rerank `documents` against `query` using the configured rerank model.
/// Per-chunk caching keyed on the **chunk text** (not file) so identical
/// chunks across files hit the same cache row. Legacy `file`-included
/// keys are checked for read-side migration.
///
/// `intent` is prepended to the rerank query (`format!("{intent}\n\n{query}")`)
/// — the LLM sees this combined string and the cache key uses it too, so
/// changing intent invalidates the cache.
///
/// Results sort by descending score. Order matches TS exactly: documents
/// with no cache hit and no LLM result return 0.0.
pub async fn rerank(
    store: &Store,
    llm: Arc<dyn Llm>,
    query: &str,
    documents: &[RerankCandidate],
    model: &str,
    intent: Option<&str>,
) -> Result<Vec<RerankScore>> {
    let rerank_query = compose_rerank_query(query, intent);

    let mut cached: HashMap<String, f32> = HashMap::new();
    // Dedupe uncached by chunk text — identical chunks across files only need
    // to be reranked once. TS uses Map<chunk, RerankDocument>.
    let mut uncached_by_chunk: HashMap<String, RerankDocument> = HashMap::new();

    store.with_connection(|conn| {
        for doc in documents {
            let key = rerank_cache_key(&rerank_query, model, &doc.text);
            let legacy_key = legacy_rerank_cache_key(query, &doc.file, model, &doc.text);

            let hit = get_cached_result(conn, &key)?.or(get_cached_result(conn, &legacy_key)?);
            if let Some(raw) = hit {
                if let Ok(score) = raw.parse::<f32>() {
                    cached.insert(doc.text.clone(), score);
                }
            } else {
                uncached_by_chunk
                    .entry(doc.text.clone())
                    .or_insert_with(|| RerankDocument {
                        file: doc.file.clone(),
                        text: doc.text.clone(),
                        title: None,
                    });
            }
        }
        Result::Ok(())
    })?;

    if !uncached_by_chunk.is_empty() {
        let uncached_docs: Vec<RerankDocument> = uncached_by_chunk.values().cloned().collect();
        let opts = RerankOptions {
            model: Some(model.to_string()),
        };
        let rerank_result = llm.rerank(&rerank_query, &uncached_docs, opts).await?;

        // Map result.file → chunk text so we cache by chunk.
        let text_by_file: HashMap<&str, &str> = uncached_docs
            .iter()
            .map(|d| (d.file.as_str(), d.text.as_str()))
            .collect();

        store.with_connection(|conn| {
            for result in &rerank_result.results {
                let chunk = text_by_file.get(result.file.as_str()).copied().unwrap_or("");
                let key = rerank_cache_key(&rerank_query, model, chunk);
                let _ = set_cached_result(conn, &key, &result.score.to_string());
                cached.insert(chunk.to_string(), result.score);
            }
            Result::Ok(())
        })?;
    }

    let mut out: Vec<RerankScore> = documents
        .iter()
        .map(|doc| RerankScore {
            file: doc.file.clone(),
            score: cached.get(&doc.text).copied().unwrap_or(0.0),
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(out)
}

/// Build the rerank query string used both as the LLM input and the
/// cache-key component.
pub(crate) fn compose_rerank_query(query: &str, intent: Option<&str>) -> String {
    match intent {
        Some(i) if !i.is_empty() => format!("{i}\n\n{query}"),
        _ => query.to_string(),
    }
}
