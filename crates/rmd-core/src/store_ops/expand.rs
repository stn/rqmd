//! `expand_query`: LLM-driven query expansion with `llm_cache` round-trip.
//!
//! Port of `tobi/qmd/src/store.ts` lines 3495–3528.

use std::sync::Arc;

use crate::store::cache::{get_cached_result, set_cached_result};
use crate::store::Store;
use serde::{Deserialize, Serialize};

use crate::llm::traits::Llm;
use crate::llm::types::{ExpandQueryOptions, QueryType, Queryable};

use super::cache_keys::expand_query_cache_key;
use super::{Error, Result};

/// Routing variant for one expansion of a search query. Distinguishes the
/// three search-strategy targets emitted by the `expandQuery` model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpandedQueryType {
    Lex,
    Vec,
    Hyde,
}

impl From<QueryType> for ExpandedQueryType {
    fn from(q: QueryType) -> Self {
        match q {
            QueryType::Lex => Self::Lex,
            QueryType::Vec => Self::Vec,
            QueryType::Hyde => Self::Hyde,
        }
    }
}

/// One expansion routing entry. Mirrors TS `ExpandedQuery` (`store.ts:328`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpandedQuery {
    #[serde(rename = "type")]
    pub type_: ExpandedQueryType,
    pub query: String,
    /// Optional source line for CLI error reporting. Always `None` for
    /// LLM-produced entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
}

impl From<Queryable> for ExpandedQuery {
    fn from(q: Queryable) -> Self {
        Self {
            type_: q.type_.into(),
            query: q.text,
            line: None,
        }
    }
}

/// Legacy cache shape: `{ "type": "lex", "text": "..." }` (pre rename to
/// `query`). Deserialized into `ExpandedQuery` for forward migration on
/// cache hits. Never written.
#[derive(Debug, Deserialize)]
struct LegacyExpandedQuery {
    #[serde(rename = "type")]
    type_: ExpandedQueryType,
    text: String,
}

/// Expand a search query into lex/vec/hyde variants. Caches the result
/// in the `llm_cache` table keyed on `(query, model, intent?)`.
///
/// `model` is the **expansion model URI** — passed through to the LLM but
/// ignored by `LlamaCpp` (it uses its configured generate model). It is
/// preserved in the cache key so that a model swap invalidates the cache.
pub async fn expand_query(
    store: &Store,
    llm: Arc<dyn Llm>,
    query: &str,
    model: &str,
    intent: Option<&str>,
) -> Result<Vec<ExpandedQuery>> {
    let cache_key = expand_query_cache_key(query, model, intent);

    // Cache lookup. Two formats accepted; one written.
    let cached = store.with_connection(|c| get_cached_result(c, &cache_key))?;
    if let Some(raw) = cached
        && let Some(parsed) = parse_cached(&raw)
    {
        return Ok(parsed);
    }

    let opts = ExpandQueryOptions {
        intent: intent.map(|s| s.to_string()),
        ..Default::default()
    };
    let queryables = llm.expand_query(query, opts).await?;

    let expanded: Vec<ExpandedQuery> = queryables
        .into_iter()
        .filter(|q| q.text != query)
        .map(ExpandedQuery::from)
        .collect();

    if !expanded.is_empty() {
        let serialized = serde_json::to_string(&expanded)?;
        store.with_connection(|c| set_cached_result(c, &cache_key, &serialized))?;
    }

    Ok(expanded)
}

/// Accept current `[{type, query}]` shape or legacy `[{type, text}]`.
/// Returns `None` (re-expand) if the payload is unrecognised.
fn parse_cached(raw: &str) -> Option<Vec<ExpandedQuery>> {
    if let Ok(parsed) = serde_json::from_str::<Vec<ExpandedQuery>>(raw) {
        return Some(parsed);
    }
    if let Ok(legacy) = serde_json::from_str::<Vec<LegacyExpandedQuery>>(raw) {
        return Some(
            legacy
                .into_iter()
                .map(|l| ExpandedQuery {
                    type_: l.type_,
                    query: l.text,
                    line: None,
                })
                .collect(),
        );
    }
    None
}

// `Error` is wired to the orchestrator-level enum; this assert keeps the
// dead-code linter quiet about the unused conversion.
const _: fn() = || {
    fn assert_from<E: Into<Error>>() {}
    assert_from::<crate::store::Error>();
    assert_from::<crate::llm::Error>();
};
