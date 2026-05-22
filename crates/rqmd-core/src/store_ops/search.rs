//! `search_vec`: embed-the-query orchestrator + thin wrapper over
//! [`crate::store::embeddings::search_vec_with_embedding`].
//!
//! Port of `tobi/qmd/src/store.ts` lines 3252–3349 (the LLM half;
//! the pure-SQL half is in rqmd-core).

use std::sync::Arc;

use crate::store::Store;
use crate::store::embeddings::search_vec_with_embedding;
use crate::store::search::SearchResult;

use crate::llm::format::{format_doc_for_embedding, format_query_for_embedding};
use crate::llm::traits::Llm;
use crate::llm::types::EmbedOptions;

use super::Result;

/// Embed `text` for `model` and return the raw `Vec<f32>`, or `None` on
/// a soft failure. Mirrors TS `getEmbedding` (`store.ts:3342`).
pub(crate) async fn get_embedding(
    llm: Arc<dyn Llm>,
    text: &str,
    model: &str,
    is_query: bool,
) -> Result<Option<Vec<f32>>> {
    let formatted = if is_query {
        format_query_for_embedding(text, model)
    } else {
        format_doc_for_embedding(text, None, model)
    };
    let opts = EmbedOptions {
        model: Some(model.to_string()),
        is_query,
        title: None,
    };
    Ok(llm.embed(&formatted, opts).await?.map(|r| r.embedding))
}

/// Vector search. Either embeds `query` via `llm` or uses
/// `precomputed_embedding` (saves a redundant embed when caller already
/// has it — used heavily by `hybrid_query`). Returns up to `limit`
/// results, deduped by filepath and sorted by descending cosine
/// similarity. Returns `[]` if `vectors_vec` does not exist yet.
pub async fn search_vec(
    store: &Store,
    llm: Arc<dyn Llm>,
    query: &str,
    model: &str,
    limit: usize,
    collection: Option<&str>,
    precomputed_embedding: Option<&[f32]>,
) -> Result<Vec<SearchResult>> {
    let embedding = match precomputed_embedding {
        Some(e) => e.to_vec(),
        None => match get_embedding(llm, query, model, true).await? {
            Some(v) => v,
            None => return Ok(Vec::new()),
        },
    };

    Ok(store
        .with_connection(|conn| search_vec_with_embedding(conn, &embedding, limit, collection))?)
}
