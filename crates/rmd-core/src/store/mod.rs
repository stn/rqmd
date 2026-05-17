//! Search-engine state and operations.
//!
//! Port of the non-LLM half of `tobi/qmd`'s `src/store.ts`. Owns the SQLite
//! schema, document CRUD, full-text search, virtual paths, chunking, RRF
//! fusion, snippet extraction, and reindexing. LLM-using functions
//! (`expandQuery`, `rerank`, `generateEmbeddings`, `searchVec`, `hybridQuery`,
//! `vectorSearch`, `structuredSearch`, `chunkDocumentByTokens`) and the
//! embedding-side SQL ops are deferred to a future pass; the schema still
//! creates the `content_vectors` and `vectors_vec` tables so that pass is
//! purely additive.

use std::path::{Path, PathBuf};

use crate::db::{self, Connection};

pub mod cache;
pub mod chunking;
pub mod context;
pub mod docid;
pub mod documents;
pub mod lookup;
pub mod maintenance;
pub mod path;
pub mod reindex;
pub mod rrf;
pub mod schema;
pub mod search;
pub mod snippet;
pub mod store_config;
pub mod virtual_path;

// ============================================================================
// Constants (TS lines 45–62, 314–318)
// ============================================================================

pub const DEFAULT_GLOB: &str = "**/*.md";
pub const DEFAULT_MULTI_GET_MAX_BYTES: usize = 10 * 1024;
pub const DEFAULT_EMBED_MAX_DOCS_PER_BATCH: usize = 64;
pub const DEFAULT_EMBED_MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

pub const CHUNK_SIZE_TOKENS: usize = 900;
pub const CHUNK_OVERLAP_TOKENS: usize = (CHUNK_SIZE_TOKENS * 15) / 100;
pub const CHUNK_SIZE_CHARS: usize = CHUNK_SIZE_TOKENS * 4;
pub const CHUNK_OVERLAP_CHARS: usize = CHUNK_OVERLAP_TOKENS * 4;
pub const CHUNK_WINDOW_TOKENS: usize = 200;
pub const CHUNK_WINDOW_CHARS: usize = CHUNK_WINDOW_TOKENS * 4;

pub const STRONG_SIGNAL_MIN_SCORE: f64 = 0.85;
pub const STRONG_SIGNAL_MIN_GAP: f64 = 0.15;
pub const RERANK_CANDIDATE_LIMIT: usize = 40;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to open database: {0}")]
    OpenDb(#[from] db::Error),

    #[error("sqlite operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("io error at {path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid virtual path: '{0}'")]
    InvalidVirtualPath(String),

    #[error("invalid glob pattern: '{0}'")]
    InvalidGlob(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error(
        "database path not set: tests must set RMD_INDEX_PATH or use Store::open() with an explicit path"
    )]
    DbPathNotSet,
}

pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Store
// ============================================================================

/// Search-engine state. Holds the SQLite connection and the path it was
/// opened from. The TS `createStore` produced an object literal with a bag
/// of methods bound to `db`; in Rust this is a struct with methods that
/// delegate to free functions in the submodules.
pub struct Store {
    pub(crate) conn: Connection,
    pub db_path: PathBuf,
    pub vec_available: bool,
}

impl Store {
    /// Open a store at an explicit path. Mirrors TS `createStore(dbPath)`.
    pub fn open<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut conn = db::open_database(&db_path)?;
        let vec_available = db::probe_sqlite_vec(&conn).is_ok();
        if !vec_available {
            eprintln!("sqlite-vec extension unavailable; vector search disabled");
        }
        schema::initialize(&mut conn)?;
        Ok(Self {
            conn,
            db_path,
            vec_available,
        })
    }

    /// Open a store at the default location. Mirrors TS `createStore()`.
    ///
    /// Respects `RMD_INDEX_PATH` and the production-mode guard — see
    /// [`path::default_db_path`].
    pub fn open_default() -> Result<Self> {
        Self::open_with_index_name("index")
    }

    /// Open the store for a named index (`<index>.sqlite` under the cache dir).
    pub fn open_with_index_name(index_name: &str) -> Result<Self> {
        let p = path::default_db_path(Some(index_name))?;
        Self::open(p)
    }

    /// Explicit close. `Connection` is dropped on its own at end-of-scope;
    /// keeping this method matches the TS `close()` shape.
    pub fn close(self) {
        drop(self.conn);
    }

    /// Borrow the underlying connection. Discouraged in favour of typed
    /// methods on `Store`; useful while `rmd-llm` does not yet have its own
    /// extension points on `Store`.
    pub fn with_connection<R>(&self, f: impl FnOnce(&Connection) -> R) -> R {
        f(&self.conn)
    }

    /// Mutable variant for transactions / DDL.
    pub fn with_connection_mut<R>(&mut self, f: impl FnOnce(&mut Connection) -> R) -> R {
        f(&mut self.conn)
    }
}
