//! `rmd-core` — search engine core for the `rmd` workspace.
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
//!   access SQLite types as `rmd_core::db::Connection`,
//!   `rmd_core::db::open_database`, etc.; no items from `db` are hoisted to
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
//!
//! **Runtime requirement**: the [`llm`] and [`store_ops`] modules expose
//! `async` APIs that internally use `tokio::sync::oneshot` to bridge between
//! the dedicated FFI worker threads and the async caller. Any crate that
//! depends on `rmd-core` and calls these modules MUST run inside a tokio
//! runtime (`rt-multi-thread` recommended).

pub mod collections;
pub mod db;
pub mod llm;
pub mod paths;
pub mod store;
pub mod store_ops;

pub use collections::{
    find_local_config_path, is_valid_collection_name, local_db_path, Collection,
    CollectionSettings, Config, ConfigData, ContextEntry, ContextMap, Error,
    IncludeByDefaultField, ModelsConfig, NamedCollectionRef, Result, UpdateField,
};
// Note: the crate-root `Error`/`Result` continue to be the
// `collections::*` ones (matching the existing public API). The
// `store::Error`/`Result`, `llm::Error`/`Result`, and `store_ops::Error`/
// `Result` are accessed via their respective module paths (or
// `StoreOpsError`/`StoreOpsResult` aliases for the latter).

pub use store::Store;
pub use store::ast::{
    AstStatus, LangStatus, SupportedLanguage, detect_language, get_ast_break_points,
    get_ast_status,
};
pub use store::chunking::{BreakKind, BreakPoint, Chunk, ChunkStrategy, CodeFenceRegion};
pub use store::reindex::{ReindexProgress, ReindexResult};
pub use store::rrf::{
    HybridQueryExplain, QueryType, RankedListMeta, RRFContributionTrace, RRFExplain, RRFScoreTrace,
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
    chunk_document_by_tokens, expand_query, generate_embeddings, hybrid_query, rerank,
    search_vec, structured_search, vector_search_query,
    EmbedOptions as StoreOpsEmbedOptions, EmbedProgress, EmbedResult,
    Error as StoreOpsError, ExpandedQuery, ExpandedQueryType,
    HashForEmbedding, HybridQueryOptions, HybridQueryResult, IndexHealthInfo, IndexStatus,
    RerankCandidate, RerankScore, Result as StoreOpsResult, SearchHooks, StructuredSearchOptions,
    TokenChunk, VectorSearchOptions, VectorSearchResult,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
