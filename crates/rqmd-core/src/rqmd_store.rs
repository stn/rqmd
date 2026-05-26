//! [`RqmdStore`] — the top-level handle external Rust callers use.
//!
//! Port of `tobi/qmd/src/index.ts` — `createStore()` / `QMDStore` interface
//! (lines 217–314 for the interface, 341+ for the constructor and method
//! implementations). The TS facade hides the SQLite handle, `LlamaCpp`
//! lifecycle, and YAML write-through behind a single object; this Rust
//! struct does the same.
//!
//! # Quick start
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
//! # Lifecycle
//!
//! Rust has no async `Drop`, so callers MUST invoke [`RqmdStore::close`] (or
//! [`RqmdStore::shutdown`] for `Arc<Mutex<RqmdStore>>` patterns) before the
//! store is dropped. Otherwise the underlying [`LlamaCpp`] worker threads
//! leak until process exit.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::collections::{Collection, Config, ConfigData};
use crate::llm::config::{ResolvedModels, resolve_embed_model, resolve_models};
use crate::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use crate::llm::traits::Llm;
use crate::llm::types::ModelResolutionConfig;
use crate::store::DEFAULT_MULTI_GET_MAX_BYTES;
use crate::store::Store;
use crate::store::cache::clear_cache;
use crate::store::chunking::ChunkStrategy;
use crate::store::context::{CollectionListing, list_collections as store_list_collections};
use crate::store::lookup::{
    FindDocumentOptions, FindDocumentOutcome, FindDocumentsOptions, FindDocumentsResult,
    find_document, find_documents, get_document_body,
};
use crate::store::reindex::{ReindexProgress, reindex_collection};
use crate::store::search::{SearchResult, search_fts};
use crate::store::status::{IndexHealthInfo, IndexStatus};
use crate::store::store_config::{
    StoreContextEntry, delete_store_collection, get_store_collection, get_store_collections,
    get_store_contexts, get_store_global_context, remove_store_context, rename_store_collection,
    set_store_global_context, sync_config_to_db, update_store_context, upsert_store_collection,
};
use crate::store_ops::{
    EmbedOptions, EmbedResult, ExpandedQuery, HybridQueryOptions, HybridQueryResult, SearchHooks,
    StructuredSearchOptions, VectorSearchOptions, VectorSearchResult,
    expand_query as ops_expand_query, generate_embeddings, hybrid_query, search_vec,
    structured_search, vector_search_query,
};

// ============================================================================
// Errors
// ============================================================================

/// Errors produced by [`RqmdStore`] methods.
///
/// Each underlying layer keeps its own error type — this enum is a convenience
/// `#[from]` union so callers don't have to manually convert.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] crate::store::Error),

    #[error(transparent)]
    Collections(#[from] crate::collections::Error),

    #[error(transparent)]
    Llm(#[from] crate::llm::Error),

    #[error(transparent)]
    StoreOps(#[from] crate::store_ops::Error),

    #[error("invalid options: {0}")]
    InvalidOptions(String),

    #[error("search requires either 'query' or 'queries'")]
    MissingSearchQuery,
}

pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Configuration
// ============================================================================

/// Construction options for [`RqmdStore::open`].
///
/// `db_path` is required. Exactly one of `config_path` / `config` may be set;
/// both unset = DB-only mode (no YAML write-through, mutations still target
/// SQLite).
#[derive(Debug, Default, Clone)]
pub struct RqmdStoreOptions {
    pub db_path: PathBuf,
    pub config_path: Option<PathBuf>,
    pub config: Option<ConfigData>,
}

// ============================================================================
// DTOs
// ============================================================================

/// Re-export of the multi-get result, identical in shape to the TS
/// `{ docs, errors }` tuple from `findDocuments`.
pub type MultiGetBundle = FindDocumentsResult;

/// Options for the unified [`RqmdStore::search`] entry point.
///
/// Mirrors TS `SearchOptions` (`index.ts:146-169`). At least one of `query`
/// or `queries` must be set; if both are set, `queries` wins (matches TS
/// behaviour at `index.ts:387-423`).
#[derive(Debug, Default)]
pub struct SearchOptions {
    pub query: Option<String>,
    pub queries: Option<Vec<ExpandedQuery>>,
    pub intent: Option<String>,
    /// `None` defaults to `true` (rerank on).
    pub rerank: Option<bool>,
    pub collection: Option<String>,
    pub collections: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub candidate_limit: Option<usize>,
    pub min_score: Option<f64>,
    pub explain: bool,
    pub chunk_strategy: Option<ChunkStrategy>,
}

/// Options for [`RqmdStore::update`].
#[derive(Default, Clone)]
pub struct UpdateOptions {
    /// Restrict reindex to a subset; `None` = all collections in the DB.
    pub collections: Option<Vec<String>>,
    /// Per-file progress callback.
    pub on_progress: Option<Arc<dyn Fn(UpdateProgress) + Send + Sync>>,
}

impl std::fmt::Debug for UpdateOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpdateOptions")
            .field("collections", &self.collections)
            .field("on_progress", &self.on_progress.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

/// Progress snapshot delivered via [`UpdateOptions::on_progress`].
#[derive(Debug, Clone)]
pub struct UpdateProgress {
    pub collection: String,
    pub file: String,
    pub current: usize,
    pub total: usize,
}

/// Aggregate result from [`RqmdStore::update`].
#[derive(Debug, Default, Clone, Copy)]
pub struct UpdateResult {
    pub collections: usize,
    pub indexed: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub needs_embedding: i64,
}

/// Options for [`RqmdStore::add_collection`].
#[derive(Debug, Default, Clone)]
pub struct AddCollectionOptions {
    pub path: String,
    pub pattern: Option<String>,
    /// Additional gitignore-style patterns. Persisted in SQLite only —
    /// `Config::add_collection` does not accept `ignore` (matches TS
    /// `collectionsAddCollection`, `index.ts:439`).
    pub ignore: Option<Vec<String>>,
}

// ============================================================================
// RqmdStore
// ============================================================================

/// Composes [`Store`] (SQLite) + lazy [`LlamaCpp`] + optional [`Config`]
/// (YAML or inline) into the single handle external callers use.
///
/// See module docs for usage and the lifecycle warning.
pub struct RqmdStore {
    inner: Store,
    llm: OnceLock<Arc<LlamaCpp>>,
    llm_config: LlamaCppConfig,
    config: Option<Config>,
}

impl RqmdStore {
    /// Open the store. Synchronous — `Store::open` (SQLite open + schema
    /// migration) and `Config::from_file` (YAML parse) are both blocking
    /// but lightweight. If you call this from a tokio runtime, wrap it in
    /// [`tokio::task::spawn_blocking`].
    pub fn open(opts: RqmdStoreOptions) -> Result<Self> {
        if opts.config_path.is_some() && opts.config.is_some() {
            return Err(Error::InvalidOptions(
                "provide either config_path or config, not both".into(),
            ));
        }
        let mut inner = Store::open(&opts.db_path)?;
        let config = if let Some(p) = opts.config_path {
            Some(Config::from_file(p)?)
        } else {
            opts.config.map(Config::inline)
        };

        // Sync the in-memory config into SQLite so collections / contexts /
        // global_context defined in YAML or inline config are visible to
        // `list_collections`, `update`, `add_context`, etc. immediately after
        // open. Mirrors qmd `createStore` (`index.ts:362,367`) and the CLI
        // (`rqmd state.rs`). DB-only mode (no config) skips this and keeps
        // whatever is already in `store_collections`.
        if let Some(cfg) = config.as_ref() {
            inner.with_connection_mut(|c| sync_config_to_db(c, cfg))?;
        }

        let model_cfg = config_models(config.as_ref());
        let llm_config = LlamaCppConfig {
            embed_model: model_cfg.embed.clone(),
            generate_model: model_cfg.generate.clone(),
            rerank_model: model_cfg.rerank.clone(),
            ..LlamaCppConfig::from_env()
        };

        Ok(Self {
            inner,
            llm: OnceLock::new(),
            llm_config,
            config,
        })
    }

    /// Tear down LLM workers without consuming `self`. Idempotent — safe
    /// to call concurrently or repeatedly. Suitable for shared ownership
    /// patterns (`Arc<Mutex<RqmdStore>>`).
    pub async fn shutdown(&self) {
        if let Some(llm) = self.llm.get() {
            llm.dispose().await;
        }
    }

    /// Tear down LLM workers and drop the store.
    pub async fn close(self) {
        self.shutdown().await;
    }

    /// SQLite path the store was opened from.
    pub fn db_path(&self) -> &Path {
        &self.inner.db_path
    }

    /// Escape hatch into the low-level [`Store`].
    ///
    /// **Warning**: writing through `Store::with_connection_mut` to
    /// `store_collections` / `store_config` will bypass the YAML
    /// write-through that [`RqmdStore`] mutators provide. Use only for
    /// bulk operations where you're willing to call
    /// [`crate::store::store_config::sync_config_to_db`] yourself.
    #[doc(hidden)]
    pub fn internal(&self) -> &Store {
        &self.inner
    }

    /// Fully-resolved model URIs (env > YAML > crate default).
    /// LLM is *not* instantiated by this call.
    pub fn resolved_models(&self) -> ResolvedModels {
        let model_cfg = config_models(self.config.as_ref());
        resolve_models(Some(&model_cfg))
    }

    /// Lazily construct (or fetch) the LLM. First call instantiates
    /// [`LlamaCpp`] from the stored config; subsequent calls hand back a
    /// cloned `Arc`.
    fn llm(&self) -> Arc<LlamaCpp> {
        self.llm
            .get_or_init(|| Arc::new(LlamaCpp::new(self.llm_config.clone())))
            .clone()
    }

    /// `true` if the LLM has been instantiated (e.g. after `search` /
    /// `embed`). For diagnostics and tests asserting the lazy-init
    /// contract.
    #[doc(hidden)]
    pub fn llm_initialized(&self) -> bool {
        self.llm.get().is_some()
    }

    // ── Search ──────────────────────────────────────────────────────────

    /// Hybrid / structured search. `queries` takes priority over `query`
    /// when both are set (matches TS `index.ts:387-423`).
    pub async fn search(&self, opts: SearchOptions) -> Result<Vec<HybridQueryResult>> {
        if opts.query.is_none() && opts.queries.is_none() {
            return Err(Error::MissingSearchQuery);
        }
        // Concat single + plural collection inputs, mirroring TS
        // `index.ts:391-395`.
        let collections: Vec<String> = opts
            .collection
            .into_iter()
            .chain(opts.collections.unwrap_or_default())
            .collect();
        let skip_rerank = opts.rerank == Some(false);
        let llm: Arc<dyn Llm> = self.llm();

        if let Some(queries) = opts.queries {
            let res = structured_search(
                &self.inner,
                llm,
                &queries,
                StructuredSearchOptions {
                    collections: (!collections.is_empty()).then_some(collections),
                    limit: opts.limit,
                    min_score: opts.min_score,
                    explain: opts.explain,
                    intent: opts.intent,
                    candidate_limit: opts.candidate_limit,
                    skip_rerank,
                    chunk_strategy: opts.chunk_strategy,
                    hooks: SearchHooks::default(),
                },
            )
            .await?;
            return Ok(res);
        }

        // TS-parity: `hybrid` accepts only a single collection. TS
        // `index.ts:413-414` takes `collections[0]` and silently drops the rest.
        let res = hybrid_query(
            &self.inner,
            llm,
            &opts.query.expect("checked above"),
            HybridQueryOptions {
                collection: collections.into_iter().next(),
                limit: opts.limit,
                min_score: opts.min_score,
                explain: opts.explain,
                intent: opts.intent,
                candidate_limit: opts.candidate_limit,
                skip_rerank,
                chunk_strategy: opts.chunk_strategy,
                hooks: SearchHooks::default(),
            },
        )
        .await?;
        Ok(res)
    }

    /// BM25 lexical search. No LLM, no expansion. Maps to TS
    /// `internal.searchFTS(q, limit, collection)` (`index.ts:424`).
    pub fn search_lex(
        &self,
        query: &str,
        limit: Option<usize>,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .inner
            .with_connection(|c| search_fts(c, query, limit, collection))?)
    }

    /// Single-vector similarity search (embed-and-search, no expansion,
    /// no rerank). Maps to TS `internal.searchVec(...)`
    /// (`index.ts:425`).
    pub async fn search_vector(
        &self,
        query: &str,
        limit: Option<usize>,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let llm: Arc<dyn Llm> = self.llm();
        let embed_model = resolve_embed_model(None);
        let limit = limit.unwrap_or(10);
        let res = search_vec(
            &self.inner,
            llm,
            query,
            &embed_model,
            limit,
            collection,
            None,
        )
        .await?;
        Ok(res)
    }

    /// LLM-driven multi-pronged search using [`vector_search_query`].
    /// Distinct from [`Self::search_vector`] (single embed) — this one
    /// runs query expansion first.
    pub async fn vector_search(
        &self,
        query: &str,
        opts: VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let llm: Arc<dyn Llm> = self.llm();
        let res = vector_search_query(&self.inner, llm, query, opts).await?;
        Ok(res)
    }

    /// Expand a query into typed sub-searches (lex/vec/hyde). For callers
    /// that want to drive [`Self::search`] with pre-expanded queries.
    pub async fn expand_query(
        &self,
        query: &str,
        intent: Option<&str>,
    ) -> Result<Vec<ExpandedQuery>> {
        let llm: Arc<dyn Llm> = self.llm();
        let embed_model = resolve_embed_model(None);
        let res = ops_expand_query(&self.inner, llm, query, &embed_model, intent).await?;
        Ok(res)
    }

    // ── Retrieval ───────────────────────────────────────────────────────

    /// Look up a single document by path, virtual path (`qmd://...`), or
    /// docid. Returns [`FindDocumentOutcome::NotFound`] with similar
    /// filenames on miss.
    pub fn get(&self, path_or_docid: &str, include_body: bool) -> Result<FindDocumentOutcome> {
        Ok(self.inner.with_connection(|c| {
            find_document(c, path_or_docid, FindDocumentOptions { include_body })
        })?)
    }

    /// Resolve `path_or_docid` to a virtual path and read its body, with
    /// optional `from_line` / `max_lines` slicing.
    pub fn get_document_body(
        &self,
        path_or_docid: &str,
        from_line: Option<usize>,
        max_lines: Option<usize>,
    ) -> Result<Option<String>> {
        let outcome = self.get(path_or_docid, false)?;
        let virtual_path = match outcome {
            FindDocumentOutcome::Found(d) => d.filepath,
            FindDocumentOutcome::NotFound(_) => return Ok(None),
        };
        Ok(self
            .inner
            .with_connection(|c| get_document_body(c, &virtual_path, from_line, max_lines))?)
    }

    /// Look up multiple documents by glob pattern or comma-separated list.
    pub fn multi_get(
        &self,
        pattern: &str,
        include_body: bool,
        max_bytes: Option<usize>,
    ) -> Result<MultiGetBundle> {
        let options = FindDocumentsOptions {
            include_body,
            max_bytes: max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES),
        };
        Ok(self
            .inner
            .with_connection(|c| find_documents(c, pattern, options))?)
    }

    // ── Collection CRUD ─────────────────────────────────────────────────

    /// Add or replace a collection. Writes to SQLite first (matching TS
    /// `index.ts:436-441`), then YAML if a [`Config`] is attached.
    pub fn add_collection(&mut self, name: &str, opts: AddCollectionOptions) -> Result<()> {
        let pattern = opts
            .pattern
            .clone()
            .unwrap_or_else(|| crate::store::DEFAULT_GLOB.to_string());
        let coll = Collection {
            path: opts.path.clone(),
            pattern: pattern.clone(),
            ignore: opts.ignore.clone(),
            context: None,
            update: None,
            include_by_default: None,
        };
        self.inner
            .with_connection_mut(|conn| upsert_store_collection(conn, name, &coll))?;
        if let Some(cfg) = &mut self.config {
            cfg.add_collection(name, opts.path, opts.pattern.as_deref())?;
        }
        Ok(())
    }

    /// Remove a collection from SQLite (and YAML if attached). Returns
    /// `false` when the collection did not exist in SQLite.
    pub fn remove_collection(&mut self, name: &str) -> Result<bool> {
        let removed = self
            .inner
            .with_connection_mut(|conn| delete_store_collection(conn, name))?;
        if let Some(cfg) = &mut self.config {
            cfg.remove_collection(name)?;
        }
        Ok(removed)
    }

    /// Rename a collection. Returns `false` when `old` did not exist, and
    /// errors with [`crate::collections::Error::DuplicateCollection`] when the
    /// target name is already taken (matches qmd `renameCollection`, which
    /// throws "already exists").
    pub fn rename_collection(&mut self, old: &str, new: &str) -> Result<bool> {
        // Guard the target up-front so callers get a typed, clear error instead
        // of a raw SQLite PRIMARY KEY constraint failure from the UPDATE.
        let target_exists = self
            .inner
            .with_connection(|c| get_store_collection(c, new).map(|o| o.is_some()))?;
        if target_exists {
            return Err(Error::Collections(
                crate::collections::Error::DuplicateCollection(new.to_string()),
            ));
        }
        let renamed = self
            .inner
            .with_connection_mut(|conn| rename_store_collection(conn, old, new))?;
        if let Some(cfg) = &mut self.config {
            cfg.rename_collection(old, new)?;
        }
        Ok(renamed)
    }

    /// List collections with document counts (read from SQLite — DB is the
    /// source of truth).
    pub fn list_collections(&self) -> Result<Vec<CollectionListing>> {
        Ok(self.inner.with_connection(store_list_collections)?)
    }

    /// Names of collections included by default in unqualified searches.
    pub fn default_collection_names(&self) -> Result<Vec<String>> {
        Ok(self
            .list_collections()?
            .into_iter()
            .filter(|c| c.include_by_default)
            .map(|c| c.name)
            .collect())
    }

    // ── Context CRUD ────────────────────────────────────────────────────

    /// Add (or replace) the context entry at `(collection, prefix)`.
    /// Returns `false` when `collection` does not exist in SQLite.
    pub fn add_context(&mut self, collection: &str, prefix: &str, text: &str) -> Result<bool> {
        let exists = self
            .inner
            .with_connection(|c| get_store_collection(c, collection).map(|o| o.is_some()))?;
        if !exists {
            return Ok(false);
        }
        self.inner
            .with_connection_mut(|conn| update_store_context(conn, collection, prefix, text))?;
        if let Some(cfg) = &mut self.config {
            // YAML may diverge from DB; ignore the bool return.
            let _ = cfg.add_context(collection, prefix, text)?;
        }
        Ok(true)
    }

    /// Remove the context entry at `(collection, prefix)`. Returns `false`
    /// when no entry exists.
    pub fn remove_context(&mut self, collection: &str, prefix: &str) -> Result<bool> {
        let removed = self
            .inner
            .with_connection_mut(|conn| remove_store_context(conn, collection, prefix))?;
        if let Some(cfg) = &mut self.config {
            let _ = cfg.remove_context(collection, prefix)?;
        }
        Ok(removed)
    }

    /// Set (or clear) the global context applied to every search.
    pub fn set_global_context(&mut self, text: Option<String>) -> Result<()> {
        self.inner
            .with_connection_mut(|conn| set_store_global_context(conn, text.as_deref()))?;
        if let Some(cfg) = &mut self.config {
            cfg.set_global_context(text)?;
        }
        Ok(())
    }

    /// Read the currently-active global context.
    pub fn get_global_context(&self) -> Result<Option<String>> {
        Ok(self.inner.with_connection(get_store_global_context)?)
    }

    /// Flattened view of every `(collection, path, context)` tuple,
    /// including the global entry (returned with `collection = "*"`,
    /// `path = "/"`).
    pub fn list_contexts(&self) -> Result<Vec<StoreContextEntry>> {
        Ok(self.inner.with_connection(get_store_contexts)?)
    }

    // ── Indexing ────────────────────────────────────────────────────────

    /// Re-index every (or a subset of) collection by walking the filesystem.
    /// Mirrors TS `update()` at `index.ts:466-510`.
    pub async fn update(&mut self, opts: UpdateOptions) -> Result<UpdateResult> {
        // DB is the source of truth for the collection list (TS pattern).
        let all = self.inner.with_connection(get_store_collections)?;
        let filter = opts.collections;
        let selected: Vec<_> = all
            .into_iter()
            .filter(|c| {
                filter
                    .as_ref()
                    .map(|f| f.iter().any(|n| n == &c.name))
                    .unwrap_or(true)
            })
            .collect();

        // Clear LLM cache once per update pass (matches TS qmd.ts line 636).
        self.inner.with_connection(clear_cache)?;

        let mut result = UpdateResult {
            collections: selected.len(),
            ..UpdateResult::default()
        };

        for coll in &selected {
            let path = PathBuf::from(&coll.path);
            let ignore = coll.ignore.clone().unwrap_or_default();
            let name = coll.name.clone();
            let pattern = coll.pattern.clone();
            let progress = opts.on_progress.clone();

            let r = self.inner.with_connection_mut(|conn| {
                reindex_collection(
                    conn,
                    &path,
                    &pattern,
                    &name,
                    &ignore,
                    |info: &ReindexProgress| {
                        if let Some(p) = progress.as_ref() {
                            p(UpdateProgress {
                                collection: name.clone(),
                                file: info.file.clone(),
                                current: info.current,
                                total: info.total,
                            });
                        }
                    },
                )
            })?;
            result.indexed += r.indexed;
            result.updated += r.updated;
            result.unchanged += r.unchanged;
            result.removed += r.removed;
        }

        let embed_model = self.resolved_models().embed;
        let fingerprint = crate::llm::embedding_fingerprint(&embed_model);
        result.needs_embedding = self.inner.with_connection(|c| {
            crate::store::embeddings::get_hashes_needing_embedding(
                c,
                None,
                &embed_model,
                &fingerprint,
            )
        })?;
        Ok(result)
    }

    /// Generate vector embeddings for documents that need them. Delegates
    /// to [`generate_embeddings`].
    pub async fn embed(&mut self, opts: EmbedOptions) -> Result<EmbedResult> {
        let llm = self.llm();
        let res = generate_embeddings(&mut self.inner, llm, opts).await?;
        Ok(res)
    }

    // ── Health ──────────────────────────────────────────────────────────

    /// Index status: total documents, embeddings needed, etc.
    pub fn status(&self) -> Result<IndexStatus> {
        let model = self.resolved_models().embed;
        let fingerprint = crate::llm::embedding_fingerprint(&model);
        Ok(self.inner.get_status(&model, &fingerprint)?)
    }

    /// Index health summary (stale-embedding age, etc.).
    pub fn index_health(&self) -> Result<IndexHealthInfo> {
        let model = self.resolved_models().embed;
        let fingerprint = crate::llm::embedding_fingerprint(&model);
        Ok(self.inner.get_index_health(&model, &fingerprint)?)
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn config_models(cfg: Option<&Config>) -> ModelResolutionConfig {
    cfg.map(|c| {
        let m = c.data().models.as_ref();
        let expand = m.and_then(|m| m.expand.as_ref());
        let sampling = expand.and_then(|e| e.sampling.as_ref());
        ModelResolutionConfig {
            embed: m.and_then(|m| m.embed.clone()),
            generate: m.and_then(|m| m.generate.clone()),
            rerank: m.and_then(|m| m.rerank.clone()),
            expand_user_message_prefix: expand.and_then(|e| e.user_message_prefix.clone()),
            expand_system_message: expand.and_then(|e| e.system_message.clone()),
            expand_fallback_hyde_template: expand.and_then(|e| e.fallback_hyde_template.clone()),
            expand_temp: sampling.and_then(|s| s.temp),
            expand_top_k: sampling.and_then(|s| s.top_k),
            expand_top_p: sampling.and_then(|s| s.top_p),
        }
    })
    .unwrap_or_default()
}
