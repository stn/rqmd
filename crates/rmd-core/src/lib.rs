//! `rmd-core` — search engine core for the `rmd` workspace.
//!
//! Maps to `src/store.ts`, `src/db.ts`, `src/collections.ts`, `src/ast.ts`
//! in the original `tobi/qmd` TypeScript implementation.

pub mod collections;
pub mod paths;

pub use collections::{
    find_local_config_path, is_valid_collection_name, local_db_path, Collection,
    CollectionSettings, Config, ConfigData, ContextEntry, ContextMap, Error, IncludeByDefaultField,
    ModelsConfig, NamedCollectionRef, Result, UpdateField,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
