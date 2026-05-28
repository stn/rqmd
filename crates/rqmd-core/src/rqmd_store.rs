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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::collections::{Collection, Config, ConfigData, ModelsConfig};
use crate::env_keys;
use crate::llm::config::{ResolvedModels, resolve_embed_model, resolve_models};
use crate::llm::device::{LlamaBackendDeviceType, probe_devices};
use crate::llm::format::format_doc_for_embedding;
use crate::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use crate::llm::pull::inspect_cached_model;
use crate::llm::session::{LlmSession, LlmSessionOptions};
use crate::llm::traits::Llm;
use crate::llm::types::{EmbedOptions as LlmEmbedOptions, ModelResolutionConfig};
use crate::store::DEFAULT_MULTI_GET_MAX_BYTES;
use crate::store::Store;
use crate::store::cache::clear_cache;
use crate::store::chunking::ChunkStrategy;
use crate::store::context::{CollectionListing, list_collections as store_list_collections};
use crate::store::doctor::{self as docsql, FingerprintGroup};
use crate::store::documents::extract_title;
use crate::store::embeddings::{
    cosine_distance, get_hashes_needing_embedding, get_stored_embedding, nearest_vector,
    vec_table_exists,
};
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
use crate::store_ops::chunk_tokens::chunk_document_by_tokens;
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
// Doctor DTOs
// ============================================================================

/// Cosine-distance threshold for "reproduces the stored vector" (qmd parity).
pub const VECTOR_MATCH_THRESHOLD: f64 = 0.0001;
/// Default wall-clock budget for LLM-backed doctor checks (qmd parity).
pub const DEFAULT_DOCTOR_LLM_TIMEOUT: Duration = Duration::from_secs(600);

/// Structured result of [`RqmdStore::doctor_report`]. Mirrors the data
/// `rqmd doctor` displays; the CLI is just a formatter on top of this.
#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub db_path: PathBuf,
    pub sqlite_version: String,
    pub vec_version: Option<String>,
    pub collection_count: usize,
    pub configured_models: Option<ModelsConfig>,
    pub resolved_models: ResolvedModels,
    pub model_cache: Vec<CachedModelEntry>,
    pub env_overrides: Vec<EnvOverride>,
    /// `device_mode()` snapshot at report time (e.g. `"auto"`, `"metal"`,
    /// `"CPU forced (RQMD_FORCE_CPU)"`).
    pub device_mode: String,
    /// `true` when `RQMD_DOCTOR_DEVICE_PROBE` skipped the probe.
    pub device_probe_skipped: bool,
    /// `Err` from the underlying llama backend probe, if any. `None` when
    /// the probe succeeded or was skipped.
    pub device_probe_error: Option<String>,
    /// Empty when `device_probe_skipped` is `true` or the probe errored.
    pub devices: Vec<DeviceInfo>,
    pub cpu_cores: usize,
    pub needs_embedding: i64,
    pub fingerprint_groups: Vec<FingerprintGroup>,
    /// Legacy (empty-fingerprint) rows observed for the active embed model.
    /// `doctor_report` itself does **not** write; call
    /// [`RqmdStore::adopt_legacy_embeddings`] to actually adopt.
    pub legacy_pending: Option<LegacyPending>,
    /// Always present — see [`VectorSampleStatus`] for the skip variants.
    pub vector_sample: VectorSampleCheck,
    /// Active embed model URI (cached here so CLI / SDK consumers don't have
    /// to re-resolve to format messages).
    pub active_embed_model: String,
    /// Current embedding fingerprint for `active_embed_model`.
    pub active_embed_fingerprint: String,
}

/// One entry in the resolved model cache view.
#[derive(Debug, Clone)]
pub struct CachedModelEntry {
    pub model_uri: String,
    pub used_for_embed: bool,
    pub used_for_generate: bool,
    pub used_for_rerank: bool,
    /// `Some(path)` when a valid cached GGUF exists.
    pub path: Option<PathBuf>,
    /// Diagnostics for cached-but-invalid files (`<path>: <detail>` strings).
    pub invalid: Vec<String>,
}

/// One environment variable rqmd actually reads, with a human-readable
/// `consequence` explaining why it matters.
#[derive(Debug, Clone)]
pub struct EnvOverride {
    pub name: String,
    pub value: String,
    pub consequence: String,
}

/// One probed llama backend device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: LlamaBackendDeviceType,
    pub backend: String,
    pub name: String,
    pub description: String,
    pub memory_total: usize,
    pub memory_free: usize,
}

/// Observation that legacy (empty-fingerprint) rows exist for the active
/// embed model. The report stage never writes — calling
/// [`RqmdStore::adopt_legacy_embeddings`] is what actually stamps them.
#[derive(Debug, Clone)]
pub struct LegacyPending {
    pub model: String,
    pub legacy_distinct_hashes: i64,
    /// One representative `(hash, seq)` from the legacy rows, or `None` when
    /// no active document references any of them.
    pub sample_hash_seq: Option<String>,
    /// Physical preconditions for adoption are met: the embed model is
    /// cached and a `vectors_vec` table exists. Note this is *necessary but
    /// not sufficient*: if no active document references any legacy row,
    /// [`RqmdStore::adopt_legacy_embeddings`] still returns `Ok(None)`.
    pub adoption_possible: bool,
}

/// Result of [`RqmdStore::adopt_legacy_embeddings`].
#[derive(Debug, Clone)]
pub struct LegacyAdoptionOutcome {
    pub model: String,
    pub fingerprint: String,
    /// `(hash, seq)` of the sample chunk that was re-embedded to verify
    /// equivalence before adopting.
    pub sample_hash_seq: String,
    /// Cosine distance between the re-embedded sample and the stored vector.
    pub sample_distance: f64,
    /// `true` when the sample matched within [`VECTOR_MATCH_THRESHOLD`] and
    /// the UPDATE was issued.
    pub adopted: bool,
    /// Rows updated by the adoption UPDATE (0 when `adopted == false`).
    pub adopted_rows: usize,
    /// Diagnostic shown to the user (e.g. "sample h1_0 matched..."), mirrors
    /// CLI `legacy fingerprint adoption` details.
    pub reason: String,
}

/// Result of the embedding-vector-sample check. Always reported (even when
/// the heavy check could not run) so CLI / SDK consumers can render a
/// qmd-parity diagnostic line.
#[derive(Debug, Clone)]
pub struct VectorSampleCheck {
    pub model: String,
    pub fingerprint: String,
    pub threshold: f64,
    pub status: VectorSampleStatus,
}

/// Why the embedding-vector-sample check resolved as it did. The first four
/// variants are "could not / did not need to sample"; [`Self::Sampled`] is the
/// only one where re-embedding actually happened.
#[derive(Debug, Clone)]
pub enum VectorSampleStatus {
    /// No active documents in the index — nothing to verify (qmd parity: ok).
    NoActiveDocuments,
    /// The `vectors_vec` virtual table does not exist yet — `rqmd embed` has
    /// not run.
    NoVectorTable,
    /// Embed model GGUF is not cached locally — needs `rqmd pull`.
    ModelNotCached,
    /// `vectors_vec` exists but no rows under the current `(model, fingerprint)`.
    NoCurrentChunks,
    /// Sampling actually ran. `failures.is_empty()` ⇔ qmd ✓.
    Sampled {
        sampled: usize,
        passed: usize,
        failures: Vec<VectorSampleFailure>,
    },
}

/// One failing sample inside [`VectorSampleStatus::Sampled::failures`].
#[derive(Debug, Clone)]
pub struct VectorSampleFailure {
    pub hash_seq: String,
    pub reason: String,
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

    // ── Doctor ──────────────────────────────────────────────────────────

    /// Structured `rqmd doctor` report. Read-only — the only write path
    /// (legacy fingerprint adoption) is exposed separately via
    /// [`RqmdStore::adopt_legacy_embeddings`].
    ///
    /// `timeout` caps the LLM session budget for the embedding-vector-sample
    /// check; `None` falls back to [`DEFAULT_DOCTOR_LLM_TIMEOUT`] (qmd parity:
    /// 600 s). Heavy checks (`vector_sample`, `legacy_pending.adoption_possible`)
    /// require the embed model to be cached and a `vectors_vec` table to
    /// exist; otherwise they are reported as `None` / `false` rather than
    /// erroring.
    pub async fn doctor_report(&self, timeout: Option<Duration>) -> Result<DoctorReport> {
        let resolved = self.resolved_models();
        let configured: Option<ModelsConfig> =
            self.config.as_ref().and_then(|c| c.data().models.clone());
        let model = resolved.embed.clone();
        let fingerprint = crate::llm::embedding_fingerprint(&model);

        let sqlite_version = self
            .inner
            .with_connection(|c| {
                c.query_row("SELECT sqlite_version()", [], |r| r.get::<_, String>(0))
            })
            .map_err(crate::store::Error::from)?;
        let vec_version = self
            .inner
            .with_connection(|c| c.query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0)))
            .ok();

        let collection_count = self
            .inner
            .with_connection(store_list_collections)
            .map(|c| c.len())
            .unwrap_or(0);

        let model_cache = collect_model_cache(&resolved);
        let env_overrides = collect_environment_overrides(&resolved, &configured);

        let (device_mode, device_probe_skipped, device_probe_error, devices) =
            collect_device_info();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);

        let needs_embedding = self
            .inner
            .with_connection(|c| get_hashes_needing_embedding(c, None, &model, &fingerprint))?;
        let fingerprint_groups = self
            .inner
            .with_connection(docsql::fingerprint_groups)
            .unwrap_or_default();

        // Cached preconditions shared by `collect_legacy_pending` and the
        // vector-sample gate — computed once so the same SQL / filesystem
        // probe isn't re-run two or three times during one report.
        let vec_table = self
            .inner
            .with_connection(vec_table_exists)
            .unwrap_or(false);
        let model_cached = inspect_cached_model(&model).path.is_some();

        let legacy_pending = self.collect_legacy_pending(&model, vec_table, model_cached)?;

        // Avoid instantiating the LLM unless we will actually call into it:
        // CI / model-less environments hit one of the early skip branches.
        let session_timeout = timeout.unwrap_or(DEFAULT_DOCTOR_LLM_TIMEOUT);
        let vector_sample = if vec_table && model_cached {
            run_vector_sample_check(
                &self.inner,
                self.llm(),
                &model,
                &fingerprint,
                session_timeout,
            )
            .await?
        } else {
            let active = self
                .inner
                .with_connection(docsql::count_active_documents)
                .unwrap_or(0);
            let status = if active == 0 {
                VectorSampleStatus::NoActiveDocuments
            } else if !vec_table {
                VectorSampleStatus::NoVectorTable
            } else {
                VectorSampleStatus::ModelNotCached
            };
            VectorSampleCheck {
                model: model.clone(),
                fingerprint: fingerprint.clone(),
                threshold: VECTOR_MATCH_THRESHOLD,
                status,
            }
        };

        Ok(DoctorReport {
            db_path: self.inner.db_path.clone(),
            sqlite_version,
            vec_version,
            collection_count,
            configured_models: configured,
            resolved_models: resolved,
            model_cache,
            env_overrides,
            device_mode,
            device_probe_skipped,
            device_probe_error,
            devices,
            cpu_cores,
            needs_embedding,
            fingerprint_groups,
            legacy_pending,
            vector_sample,
            active_embed_model: model,
            active_embed_fingerprint: fingerprint,
        })
    }

    /// Adopt legacy (empty-fingerprint) embedding rows for the active embed
    /// model: re-embed one sample chunk, confirm it reproduces the stored
    /// vector within [`VECTOR_MATCH_THRESHOLD`], then stamp the current
    /// fingerprint onto every legacy row for the model.
    ///
    /// Returns `Ok(None)` when there is nothing to adopt, the embed model is
    /// not cached, or the `vectors_vec` table does not exist. `timeout`
    /// caps the LLM session budget; `None` uses [`DEFAULT_DOCTOR_LLM_TIMEOUT`].
    pub async fn adopt_legacy_embeddings(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<LegacyAdoptionOutcome>> {
        let model = self.resolved_models().embed;
        let fingerprint = crate::llm::embedding_fingerprint(&model);

        let legacy = self
            .inner
            .with_connection(|c| docsql::count_legacy_distinct_hashes(c, &model))?;
        if legacy == 0 {
            return Ok(None);
        }
        if !self
            .inner
            .with_connection(vec_table_exists)
            .unwrap_or(false)
        {
            return Ok(None);
        }
        if inspect_cached_model(&model).path.is_none() {
            return Ok(None);
        }
        let Some(sample) = self
            .inner
            .with_connection(|c| docsql::sample_legacy_chunk(c, &model))?
        else {
            return Ok(None);
        };

        let expected = format!("{}_{}", sample.hash, sample.seq);
        let title = extract_title(&sample.body, &sample.path);
        let session = LlmSession::new(
            self.llm(),
            LlmSessionOptions {
                max_duration: Some(timeout.unwrap_or(DEFAULT_DOCTOR_LLM_TIMEOUT)),
                name: Some("doctorLegacyAdoption".into()),
            },
        );

        let embedding = async {
            let chunks = chunk_document_by_tokens(
                session.clone(),
                &sample.body,
                None,
                None,
                None,
                Some(&sample.path),
                ChunkStrategy::Auto,
                Some(session.signal()),
            )
            .await
            .ok()?;
            let chunk = chunks.get(sample.seq as usize)?;
            let formatted = format_doc_for_embedding(&chunk.text, Some(&title), &model);
            let result = session
                .embed(
                    &formatted,
                    LlmEmbedOptions {
                        model: Some(model.clone()),
                        is_query: false,
                        title: None,
                    },
                )
                .await
                .ok()??;
            Some(result.embedding)
        }
        .await;
        session.release();

        let Some(embedding) = embedding else {
            return Ok(Some(LegacyAdoptionOutcome {
                model: model.clone(),
                fingerprint,
                sample_hash_seq: expected,
                sample_distance: f64::INFINITY,
                adopted: false,
                adopted_rows: 0,
                reason: "failed to embed legacy sample".into(),
            }));
        };

        let nearest = self
            .inner
            .with_connection(|c| nearest_vector(c, &embedding))?;
        let Some((hash_seq, distance)) = nearest else {
            return Ok(Some(LegacyAdoptionOutcome {
                model: model.clone(),
                fingerprint,
                sample_hash_seq: expected,
                sample_distance: f64::INFINITY,
                adopted: false,
                adopted_rows: 0,
                reason: "legacy sample vector not found".into(),
            }));
        };
        if hash_seq != expected || distance > VECTOR_MATCH_THRESHOLD {
            return Ok(Some(LegacyAdoptionOutcome {
                model: model.clone(),
                fingerprint,
                sample_hash_seq: expected,
                sample_distance: distance,
                adopted: false,
                adopted_rows: 0,
                reason: format!(
                    "legacy sample differs from current fingerprint (nearest {hash_seq}, distance {distance:.6})"
                ),
            }));
        }

        let adopted_rows = self
            .inner
            .with_connection_mut(|c| docsql::adopt_legacy_fingerprint(c, &model, &fingerprint))?;
        let reason =
            format!("sample {expected} matched current fingerprint at distance {distance:.6}");
        Ok(Some(LegacyAdoptionOutcome {
            model,
            fingerprint,
            sample_hash_seq: expected,
            sample_distance: distance,
            adopted: adopted_rows > 0,
            adopted_rows,
            reason,
        }))
    }

    fn collect_legacy_pending(
        &self,
        model: &str,
        vec_table: bool,
        model_cached: bool,
    ) -> Result<Option<LegacyPending>> {
        let legacy = self
            .inner
            .with_connection(|c| docsql::count_legacy_distinct_hashes(c, model))?;
        if legacy == 0 {
            return Ok(None);
        }
        let sample = self
            .inner
            .with_connection(|c| docsql::sample_legacy_chunk(c, model))?
            .map(|s| format!("{}_{}", s.hash, s.seq));
        Ok(Some(LegacyPending {
            model: model.to_string(),
            legacy_distinct_hashes: legacy,
            sample_hash_seq: sample,
            adoption_possible: vec_table && model_cached,
        }))
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// `device_mode()` snapshot. Mirrors qmd `configuredGpuModeLabel`.
fn device_mode() -> String {
    if is_force_cpu() {
        return "CPU forced (RQMD_FORCE_CPU)".to_string();
    }
    if let Ok(v) = std::env::var(env_keys::LLAMA_GPU) {
        let t = v.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    "auto".to_string()
}

fn is_force_cpu() -> bool {
    match std::env::var(env_keys::FORCE_CPU) {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !t.is_empty()
                && !matches!(
                    t.as_str(),
                    "false" | "off" | "none" | "disable" | "disabled" | "0"
                )
        }
        Err(_) => false,
    }
}

fn env_value_for_display(value: &str) -> String {
    if value.chars().count() > 96 {
        let head: String = value.chars().take(93).collect();
        format!("{head}...")
    } else {
        value.to_string()
    }
}

fn model_config<'a>(
    configured: &'a Option<ModelsConfig>,
    pick: impl Fn(&'a ModelsConfig) -> &'a Option<String>,
) -> Option<&'a str> {
    configured.as_ref().and_then(|m| pick(m).as_deref())
}

fn push_model_override(
    out: &mut Vec<EnvOverride>,
    name: &str,
    key: &str,
    active: &str,
    configured: Option<&str>,
) {
    let Ok(raw) = std::env::var(name) else {
        return;
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return;
    }
    let consequence = match configured {
        Some(c) if c != raw => {
            format!("set but ignored because index models.{key} is configured as {c}")
        }
        _ => format!(
            "sets the active {key} model to {active}; changes embedding/search semantics and may require `rqmd pull` plus `rqmd embed`"
        ),
    };
    out.push(EnvOverride {
        name: name.to_string(),
        value: env_value_for_display(raw),
        consequence,
    });
}

/// Collect every `RQMD_*` / known env var that rqmd reads at runtime, with
/// a human-readable `consequence`. Port of CLI `collect_environment_overrides`.
fn collect_environment_overrides(
    resolved: &ResolvedModels,
    configured: &Option<ModelsConfig>,
) -> Vec<EnvOverride> {
    let mut out: Vec<EnvOverride> = Vec::new();

    macro_rules! add {
        ($name:expr, $consequence:expr) => {
            if let Ok(raw) = std::env::var($name) {
                let raw = raw.trim();
                if !raw.is_empty() {
                    out.push(EnvOverride {
                        name: $name.to_string(),
                        value: env_value_for_display(raw),
                        consequence: ($consequence).to_string(),
                    });
                }
            }
        };
    }

    add!(
        "RQMD_INDEX_PATH",
        "overrides the SQLite index path; rqmd reads/writes a different database"
    );
    add!(
        "RQMD_CONFIG_DIR",
        "overrides the rqmd config directory and takes precedence over XDG_CONFIG_HOME"
    );
    add!(
        "RQMD_CACHE_DIR",
        "overrides the rqmd cache directory (index cache and model cache)"
    );
    add!(
        "XDG_CONFIG_HOME",
        "moves rqmd config to $XDG_CONFIG_HOME/qmd when RQMD_CONFIG_DIR is not set"
    );
    add!(
        "XDG_CACHE_HOME",
        "moves the default index cache and model cache"
    );

    push_model_override(
        &mut out,
        env_keys::EMBED_MODEL,
        "embed",
        &resolved.embed,
        model_config(configured, |m| &m.embed),
    );
    push_model_override(
        &mut out,
        env_keys::GENERATE_MODEL,
        "generate",
        &resolved.generate,
        model_config(configured, |m| &m.generate),
    );
    push_model_override(
        &mut out,
        env_keys::RERANK_MODEL,
        "rerank",
        &resolved.rerank,
        model_config(configured, |m| &m.rerank),
    );

    add!(
        env_keys::FORCE_CPU,
        "forces llama.cpp to bypass GPU backends; embeddings/query will be slower but GPU crashes are avoided"
    );
    add!(
        env_keys::LLAMA_GPU,
        "selects llama.cpp GPU backend (metal/cuda/vulkan) or disables GPU when set to false/off/0"
    );
    add!(
        env_keys::DOCTOR_DEVICE_PROBE,
        "controls rqmd doctor native device probing; 0/off skips GPU probing"
    );
    add!(
        env_keys::EMBED_PARALLELISM,
        "overrides embedding parallel context count; too high can exhaust RAM/VRAM"
    );
    add!(
        env_keys::RERANK_PARALLELISM,
        "overrides reranker parallel context count; too high can exhaust RAM/VRAM"
    );
    add!(
        env_keys::EXPAND_CONTEXT_SIZE,
        "overrides query expansion context size; larger values use more memory"
    );
    add!(
        env_keys::RERANK_CONTEXT_SIZE,
        "overrides reranker context size; larger values use more memory"
    );
    add!(
        env_keys::EMBED_CONTEXT_SIZE,
        "overrides embed context size; larger values use more memory"
    );
    add!(
        "RQMD_EDITOR_URI",
        "overrides clickable editor link template in terminal output"
    );
    add!(
        "RQMD_SKILLS_DIR",
        "overrides where rqmd skills are discovered from"
    );
    add!("NO_COLOR", "disables colored terminal output");
    add!(
        "CI",
        "disables real LLM operations inside rqmd's LlamaCpp wrapper"
    );
    add!(
        "HF_ENDPOINT",
        "changes Hugging Face download endpoint used when pulling models"
    );
    add!("WSL_DISTRO_NAME", "enables WSL path handling heuristics");
    add!("WSL_INTEROP", "enables WSL path handling heuristics");

    out
}

/// Per-URI model cache view: dedup `embed`/`generate`/`rerank` URIs (preserve
/// first-seen order) and inspect each with [`inspect_cached_model`].
fn collect_model_cache(resolved: &ResolvedModels) -> Vec<CachedModelEntry> {
    let roles: [(&str, &str); 3] = [
        ("embed", resolved.embed.as_str()),
        ("generate", resolved.generate.as_str()),
        ("rerank", resolved.rerank.as_str()),
    ];

    let mut order: Vec<String> = Vec::new();
    let mut roles_by: HashMap<String, [bool; 3]> = HashMap::new();
    for (role, uri) in roles {
        let entry = roles_by.entry(uri.to_string()).or_insert_with(|| {
            order.push(uri.to_string());
            [false; 3]
        });
        match role {
            "embed" => entry[0] = true,
            "generate" => entry[1] = true,
            "rerank" => entry[2] = true,
            _ => {}
        }
    }

    order
        .into_iter()
        .map(|uri| {
            let flags = roles_by[&uri];
            let inspection = inspect_cached_model(&uri);
            CachedModelEntry {
                model_uri: uri,
                used_for_embed: flags[0],
                used_for_generate: flags[1],
                used_for_rerank: flags[2],
                path: inspection.path,
                invalid: inspection.invalid,
            }
        })
        .collect()
}

/// Probe llama backend devices, gated by `RQMD_DOCTOR_DEVICE_PROBE`.
/// Returns `(device_mode, skipped, probe_error, devices)`.
fn collect_device_info() -> (String, bool, Option<String>, Vec<DeviceInfo>) {
    let mode = device_mode();
    let skip = matches!(
        std::env::var(env_keys::DOCTOR_DEVICE_PROBE)
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off" | "no" | "skip")
    );
    if skip {
        return (mode, true, None, Vec::new());
    }
    match probe_devices() {
        Ok(devs) => {
            let infos = devs
                .into_iter()
                .map(|d| DeviceInfo {
                    device_type: d.device_type,
                    backend: d.backend,
                    name: d.name,
                    description: d.description,
                    memory_total: d.memory_total,
                    memory_free: d.memory_free,
                })
                .collect();
            (mode, false, None, infos)
        }
        Err(e) => (mode, false, Some(e.to_string()), Vec::new()),
    }
}

/// Sample up to 3 chunks under `(model, fingerprint)`, re-chunk + re-embed
/// each, and compare to the stored vector. Always returns a
/// [`VectorSampleCheck`]; the `status` distinguishes "actually sampled" from
/// the four skip reasons.
async fn run_vector_sample_check(
    store: &Store,
    llm: Arc<LlamaCpp>,
    model: &str,
    fingerprint: &str,
    timeout: Duration,
) -> Result<VectorSampleCheck> {
    let mk = |status: VectorSampleStatus| VectorSampleCheck {
        model: model.to_string(),
        fingerprint: fingerprint.to_string(),
        threshold: VECTOR_MATCH_THRESHOLD,
        status,
    };

    let active = store
        .with_connection(docsql::count_active_documents)
        .unwrap_or(0);
    if active == 0 {
        return Ok(mk(VectorSampleStatus::NoActiveDocuments));
    }
    if !store.with_connection(vec_table_exists).unwrap_or(false) {
        return Ok(mk(VectorSampleStatus::NoVectorTable));
    }
    if inspect_cached_model(model).path.is_none() {
        return Ok(mk(VectorSampleStatus::ModelNotCached));
    }
    let samples = store
        .with_connection(|c| docsql::sample_current_chunks(c, model, fingerprint, 3))
        .unwrap_or_default();
    if samples.is_empty() {
        return Ok(mk(VectorSampleStatus::NoCurrentChunks));
    }

    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            max_duration: Some(timeout),
            name: Some("doctorEmbeddingVectorSample".into()),
        },
    );

    let total = samples.len();
    let mut failures: Vec<VectorSampleFailure> = Vec::new();
    for sample in samples {
        let hash_seq = format!("{}_{}", sample.hash, sample.seq);

        let chunks = match chunk_document_by_tokens(
            session.clone(),
            &sample.body,
            None,
            None,
            None,
            Some(&sample.path),
            ChunkStrategy::Auto,
            Some(session.signal()),
        )
        .await
        {
            Ok(c) => c,
            Err(_) => {
                failures.push(VectorSampleFailure {
                    hash_seq,
                    reason: "chunk no longer exists".into(),
                });
                continue;
            }
        };
        let Some(chunk) = chunks.get(sample.seq as usize) else {
            failures.push(VectorSampleFailure {
                hash_seq,
                reason: "chunk no longer exists".into(),
            });
            continue;
        };
        let title = extract_title(&sample.body, &sample.path);
        let formatted = format_doc_for_embedding(&chunk.text, Some(&title), model);
        let embedding = match session
            .embed(
                &formatted,
                LlmEmbedOptions {
                    model: Some(model.to_string()),
                    is_query: false,
                    title: None,
                },
            )
            .await
        {
            Ok(Some(r)) => r.embedding,
            _ => {
                failures.push(VectorSampleFailure {
                    hash_seq,
                    reason: "embedding failed".into(),
                });
                continue;
            }
        };
        let stored = store
            .with_connection(|c| get_stored_embedding(c, &hash_seq))
            .ok()
            .flatten();
        let Some(stored) = stored else {
            failures.push(VectorSampleFailure {
                hash_seq,
                reason: "stored vector missing".into(),
            });
            continue;
        };
        let distance = cosine_distance(&embedding, &stored);
        if distance > VECTOR_MATCH_THRESHOLD {
            failures.push(VectorSampleFailure {
                hash_seq,
                reason: format!("stored vector distance {distance:.6}"),
            });
        }
    }
    session.release();

    let passed = total - failures.len();
    Ok(mk(VectorSampleStatus::Sampled {
        sampled: total,
        passed,
        failures,
    }))
}

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
