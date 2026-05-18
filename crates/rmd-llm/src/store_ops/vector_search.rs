//! `vector_search_query`: vec/hyde-only search with LLM expansion.
//!
//! Port of `tobi/qmd/src/store.ts` lines 4582–4629. Deliberate departure
//! from TS: we batch-embed every query (original + vec/hyde) in a single
//! `embed_batch` call instead of sequential `embed()`s, then do the
//! `search_vec_with_embedding` lookups sequentially (single SQLite
//! `Connection`).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rmd_core::store::embeddings::search_vec_with_embedding;
use rmd_core::store::Store;

use crate::config::resolve_embed_model;
use crate::format::format_query_for_embedding;
use crate::traits::Llm;
use crate::types::EmbedOptions;

use super::expand::{expand_query, ExpandedQuery, ExpandedQueryType};
use super::hybrid::{has_vector_index, SearchHooks};
use super::Result;

#[derive(Debug, Default, Clone)]
pub struct VectorSearchOptions {
    pub collection: Option<String>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
    pub intent: Option<String>,
    /// Only `on_expand` from [`SearchHooks`] is fired (matching TS
    /// `Pick<SearchHooks, 'onExpand'>`).
    pub hooks: SearchHooks,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearchResult {
    pub file: String,
    pub display_path: String,
    pub title: String,
    pub body: String,
    pub score: f64,
    pub context: Option<String>,
    pub docid: String,
}

pub async fn vector_search_query(
    store: &Store,
    llm: Arc<dyn Llm>,
    query: &str,
    options: VectorSearchOptions,
) -> Result<Vec<VectorSearchResult>> {
    let limit = options.limit.unwrap_or(10);
    let min_score = options.min_score.unwrap_or(0.3);
    let collection = options.collection.as_deref();
    let intent = options.intent.as_deref();

    if !has_vector_index(store)? {
        return Ok(Vec::new());
    }

    let embed_model = resolve_embed_model(None);

    // Step 1: expand and filter to vec/hyde only.
    let expand_start = Instant::now();
    let all_expanded = expand_query(store, llm.clone(), query, &embed_model, intent).await?;
    let vec_expanded: Vec<&ExpandedQuery> = all_expanded
        .iter()
        .filter(|q| !matches!(q.type_, ExpandedQueryType::Lex))
        .collect();
    if let Some(h) = &options.hooks.on_expand {
        let snapshot: Vec<ExpandedQuery> = vec_expanded.iter().map(|q| (*q).clone()).collect();
        h(query, &snapshot, expand_start.elapsed().as_millis());
    }

    // Step 2: batch embed (original + vec/hyde), then sequential vec lookup.
    let mut texts: Vec<String> = Vec::with_capacity(1 + vec_expanded.len());
    texts.push(query.to_string());
    for q in &vec_expanded {
        texts.push(q.query.clone());
    }
    let formatted: Vec<String> = texts
        .iter()
        .map(|t| format_query_for_embedding(t, &embed_model))
        .collect();

    let embeddings = llm
        .embed_batch(
            &formatted,
            EmbedOptions {
                model: Some(embed_model.clone()),
                is_query: true,
                title: None,
            },
        )
        .await?;

    let mut accum: HashMap<String, VectorSearchResult> = HashMap::new();
    for (i, _) in texts.iter().enumerate() {
        let Some(Some(emb)) = embeddings.get(i) else {
            continue;
        };
        let results =
            store.with_connection(|c| search_vec_with_embedding(c, &emb.embedding, limit, collection))?;
        for r in results {
            let body = r.doc.body.clone().unwrap_or_default();
            let entry = accum
                .entry(r.doc.filepath.clone())
                .or_insert_with(|| VectorSearchResult {
                    file: r.doc.filepath.clone(),
                    display_path: r.doc.display_path.clone(),
                    title: r.doc.title.clone(),
                    body: body.clone(),
                    score: r.score,
                    context: r.doc.context.clone(),
                    docid: r.doc.docid.clone(),
                });
            if r.score > entry.score {
                entry.score = r.score;
            }
        }
    }

    let mut out: Vec<VectorSearchResult> = accum.into_values().collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(out.into_iter().filter(|r| r.score >= min_score).take(limit).collect())
}
