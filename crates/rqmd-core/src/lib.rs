//! `rqmd-core` — search engine core for the `rqmd` workspace.
//!
//! Maps to `src/store.ts`, `src/db.ts`, `src/collections.ts`, `src/ast.ts`,
//! and `src/llm.ts` in the original `tobi/qmd` TypeScript implementation.
//!
//! Module layout:
//!
//! * [`collections`] — workspace YAML + inline configuration (Rust equivalent
//!   of `collections.ts`). Provides `Config`, `ConfigData`, `Collection`,
//!   `ContextEntry`, etc.; the crate-root `Error`/`Result` aliases continue
//!   to point at this module.
//! * [`db`] — SQLite connection management (`db.ts`). Downstream callers
//!   access SQLite types as `rqmd_core::db::Connection`,
//!   `rqmd_core::db::open_database`, etc.; no items from `db` are hoisted to
//!   the crate root.
//! * [`store`] — the non-LLM half of `store.ts`: schema, document CRUD,
//!   FTS5 search, virtual paths, chunking, RRF fusion, snippet extraction,
//!   reindexing. Synchronous API.
//! * [`llm`] — local GGUF model integration (Rust equivalent of `llm.ts`).
//!   Provides the [`llm::traits::Llm`] async trait, [`llm::llama_cpp::LlamaCpp`]
//!   concrete implementation, worker pools, scoped sessions, HF download.
//!   Settings resolution lives in [`llm::config`].
//! * [`store_ops`] — the LLM-using half of `store.ts`: hybrid / vector /
//!   structured search, query expansion, reranking, embedding generation,
//!   token-aware chunking. Free functions that combine `store` with `llm`.
//! * [`rqmd_store`] — [`RqmdStore`], the top-level handle that bundles the
//!   three modules above into a single object. Port of
//!   `tobi/qmd/src/index.ts` `createStore()` / `QMDStore`.
//!
//! **Runtime requirement**: the [`llm`] and [`store_ops`] modules expose
//! `async` APIs that internally use `tokio::sync::oneshot` to bridge between
//! the dedicated FFI worker threads and the async caller. Any crate that
//! depends on `rqmd-core` and calls these modules MUST run inside a tokio
//! runtime (`rt-multi-thread` recommended).
//!
//! # RQMD quickstart
//!
//! ```no_run
//! use rqmd_core::{RqmdStore, RqmdStoreOptions, SearchOptions};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let store = RqmdStore::open(RqmdStoreOptions {
//!     db_path: "./index.sqlite".into(),
//!     config_path: Some("./rqmd.yml".into()),
//!     ..Default::default()
//! })?;
//!
//! let hits = store.search(SearchOptions {
//!     query: Some("authentication flow".into()),
//!     ..Default::default()
//! }).await?;
//!
//! store.close().await;
//! # Ok(())
//! # }
//! ```
//!
//! Rust has no async `Drop`. Always call [`RqmdStore::close`] (or
//! [`RqmdStore::shutdown`] when the store is shared) before drop, otherwise
//! the [`LlamaCpp`] worker threads leak.

pub mod bench;
pub mod collections;
pub mod db;
pub mod llm;
pub mod paths;
pub mod rqmd_store;
pub mod store;
pub mod store_ops;

pub use collections::{
    Collection, CollectionSettings, Config, ConfigData, ContextEntry, ContextMap, Error,
    IncludeByDefaultField, ModelsConfig, NamedCollectionRef, Result, UpdateField,
    find_local_config_path, is_valid_collection_name, local_db_path,
};
// Note: the crate-root `Error`/`Result` continue to be the
// `collections::*` ones (matching the existing public API). The
// `store::Error`/`Result`, `llm::Error`/`Result`, and `store_ops::Error`/
// `Result` are accessed via their respective module paths (or
// `StoreOpsError`/`StoreOpsResult` aliases for the latter).

pub use store::Store;
pub use store::ast::{
    AstStatus, LangStatus, SupportedLanguage, detect_language, get_ast_break_points, get_ast_status,
};
pub use store::chunking::{BreakKind, BreakPoint, Chunk, ChunkStrategy, CodeFenceRegion};
pub use store::reindex::{ReindexProgress, ReindexResult};
pub use store::rrf::{
    HybridQueryExplain, QueryType, RRFContributionTrace, RRFExplain, RRFScoreTrace, RankedListMeta,
};
pub use store::search::{
    CollectionInfo, DocumentNotFound, DocumentResult, MultiGetResult, RankedResult, SearchResult,
    SearchSource,
};
pub use store::virtual_path::VirtualPath;

// LLM module convenience re-exports. `llm::Error`/`Result` and
// `llm::types::QueryType` are intentionally NOT re-exported at root to avoid
// colliding with `collections::Error`/`Result` and `store::rrf::QueryType`.
pub use llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
pub use llm::session::{LlmSession, LlmSessionOptions, with_llm_session};
pub use llm::singleton::{default_llama_cpp, dispose_default_llama_cpp, set_default_llama_cpp};
pub use llm::traits::{LlamaToken, Llm};
pub use llm::types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    ModelResolutionConfig, PullOptions, PullResult, Queryable, RerankDocument,
    RerankDocumentResult, RerankOptions, RerankResult, TokenLogProb,
};

// Store-ops module convenience re-exports. `EmbedOptions` from `store_ops`
// is aliased to `StoreOpsEmbedOptions` to avoid colliding with the
// `llm::types::EmbedOptions` already exported above. Likewise `Error` /
// `Result`.
pub use store_ops::{
    EmbedOptions as StoreOpsEmbedOptions, EmbedProgress, EmbedResult, Error as StoreOpsError,
    ExpandedQuery, ExpandedQueryType, HashForEmbedding, HybridQueryOptions, HybridQueryResult,
    IndexHealthInfo, IndexStatus, RerankCandidate, RerankScore, Result as StoreOpsResult,
    SearchHooks, StructuredSearchOptions, TokenChunk, VectorSearchOptions, VectorSearchResult,
    chunk_document_by_tokens, expand_query, generate_embeddings, hybrid_query, rerank, search_vec,
    structured_search, vector_search_query,
};

// `RqmdStore`: combines [`Store`], [`LlamaCpp`], and [`Config`] into the
// single object that `tobi/qmd`'s TypeScript `createStore()` returns.
// Errors are exposed as `RqmdStoreError` to avoid colliding with the
// `collections::Error` re-exported as crate-root `Error`.
pub use rqmd_store::{
    AddCollectionOptions, Error as RqmdStoreError, MultiGetBundle, Result as RqmdStoreResult,
    RqmdStore, RqmdStoreOptions, SearchOptions, UpdateOptions, UpdateProgress, UpdateResult,
};

// Bench module re-exports (pure scoring + fixture/result types). The runner
// itself is CLI-only and lives in `rqmd-cli`.
pub use bench::{
    BackendResult, BenchmarkFixture, BenchmarkQuery, BenchmarkResult, QueryResult, ScoreMetrics,
    SummaryStats, normalize_path, paths_match, score_results,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
