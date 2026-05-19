//! Store-orchestrating operations: the LLM-using half of `tobi/qmd`'s
//! `src/store.ts`, expressed as free functions that take `&Store` plus
//! either an `Arc<dyn Llm>` or `Arc<LlamaCpp>`.
//!
//! Layering: `rmd-core` owns the pure-SQL embedding-side operations
//! ([`crate::store::embeddings`]) and status types
//! ([`crate::store::status`]). This module composes those with the
//! [`crate::Llm`] trait to provide end-to-end orchestrators:
//!
//! | TS function | Rust function |
//! | ---- | ---- |
//! | `expandQuery` | [`expand_query`] |
//! | `rerank` | [`rerank`] |
//! | `chunkDocumentByTokens` | [`chunk_document_by_tokens`] |
//! | `searchVec` | [`search_vec`] |
//! | `generateEmbeddings` | [`generate_embeddings`] |
//! | `hybridQuery` | [`hybrid_query`] |
//! | `vectorSearchQuery` | [`vector_search_query`] |
//! | `structuredSearch` | [`structured_search`] |
//!
//! All orchestrators return [`Result`], an alias over the typed
//! [`Error`] enum so downstream CLI/MCP code can pattern-match on
//! `VecUnavailable` / `SessionExpired` / `InvalidSearch` independently
//! of `rmd_core::collections` / `rmd_core::llm` errors.

mod cache_keys;
pub mod chunk_tokens;
pub mod embed;
pub mod expand;
pub mod hybrid;
pub mod rerank;
pub mod search;
pub mod structured;
pub mod vector_search;

pub use crate::store::embeddings::{EmbeddingDoc, HashForEmbedding, PendingEmbeddingDoc};
pub use crate::store::status::{IndexHealthInfo, IndexStatus};

pub use chunk_tokens::{chunk_document_by_tokens, TokenChunk};
pub use embed::{generate_embeddings, EmbedOptions, EmbedProgress, EmbedResult};
pub use expand::{expand_query, ExpandedQuery, ExpandedQueryType};
pub use hybrid::{hybrid_query, HybridQueryOptions, HybridQueryResult, SearchHooks};
pub use rerank::{rerank, RerankCandidate, RerankScore};
pub use search::search_vec;
pub use structured::{structured_search, StructuredSearchOptions};
pub use vector_search::{vector_search_query, VectorSearchOptions, VectorSearchResult};

/// Errors produced by orchestration functions. Wraps both `collections`
/// and `llm` errors so callers don't have to convert manually, and adds
/// orchestrator-specific variants (vec table missing, session timeout
/// observed mid-run, malformed structured search input).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] crate::store::Error),

    #[error(transparent)]
    Llm(#[from] crate::llm::Error),

    #[error("vector index unavailable: {0}")]
    VecUnavailable(String),

    #[error("embedding failed: {0}")]
    EmbedFailed(String),

    #[error("session expired during {op}")]
    SessionExpired { op: &'static str },

    #[error("invalid structured search: {0}")]
    InvalidSearch(String),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
