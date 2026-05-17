//! `rmd-core` — search engine core for the `rmd` workspace.
//!
//! Maps to `src/store.ts`, `src/db.ts`, `src/collections.ts`, `src/ast.ts`
//! in the original `tobi/qmd` TypeScript implementation. `db.ts` is covered
//! by [`db`]; downstream callers access SQLite types as
//! `rmd_core::db::Connection`, `rmd_core::db::open_database`, and so on
//! (no items from `db` are hoisted to the crate root).

pub mod collections;
pub mod db;
pub mod paths;
pub mod store;

pub use collections::{
    find_local_config_path, is_valid_collection_name, local_db_path, Collection,
    CollectionSettings, Config, ConfigData, ContextEntry, ContextMap, Error,
    IncludeByDefaultField, ModelsConfig, NamedCollectionRef, Result, UpdateField,
};
// Note: the crate-root `Error`/`Result` continue to be the
// `collections::*` ones (matching the existing public API). The
// `store::Error`/`Result` are accessed via `rmd_core::store::{Error, Result}`
// — the two cannot share the crate root.

pub use store::Store;
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

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
