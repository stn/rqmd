//! CLI lifecycle: lazy [`Store`] + [`Config`] managed by an [`IndexState`].
//!
//! Maps to qmd's module-level `let store / getStore()` helpers and the
//! `--index` / local-config resolution in `src/cli/qmd.ts` (lines 119–188).

use std::path::PathBuf;

use anyhow::{Context, Result};
use rmd_core::collections::{find_local_config_path, local_db_path, Config};
use rmd_core::store::path::{default_db_path, pwd};
use rmd_core::store::store_config::sync_config_to_db;
use rmd_core::Store;

/// Holds the CLI's index selection plus the lazily-opened [`Store`] and
/// [`Config`]. One per `rmd` invocation.
pub struct IndexState {
    index_name: String,
    db_path_override: Option<PathBuf>,
    config_path_override: Option<PathBuf>,
    store: Option<Store>,
    config: Option<Config>,
}

impl IndexState {
    /// Build state from the parsed CLI globals.
    pub fn new(index_name: Option<&str>) -> Self {
        let (index_name, db_path_override, config_path_override) = match index_name {
            Some(name) => (name.to_string(), None, None),
            None => match find_local_config_path(&pwd()) {
                Some(cfg) => ("index".to_string(), Some(local_db_path(&cfg)), Some(cfg)),
                None => ("index".to_string(), None, None),
            },
        };
        Self {
            index_name,
            db_path_override,
            config_path_override,
            store: None,
            config: None,
        }
    }

    /// Path to the SQLite index that will be opened on first use.
    pub fn db_path(&self) -> Result<PathBuf> {
        if let Some(p) = &self.db_path_override {
            return Ok(p.clone());
        }
        default_db_path(Some(&self.index_name))
            .context("resolving default index path (set RMD_INDEX_PATH or use --index)")
    }

    /// Open the [`Store`] (lazy). Subsequent calls return the same handle.
    pub fn store_mut(&mut self) -> Result<&mut Store> {
        if self.store.is_none() {
            let path = self.db_path()?;
            let store = Store::open(&path)
                .with_context(|| format!("opening index at {}", path.display()))?;
            self.store = Some(store);
        }
        Ok(self.store.as_mut().expect("just inserted"))
    }

    /// Load the YAML [`Config`] (lazy).
    pub fn config_mut(&mut self) -> Result<&mut Config> {
        if self.config.is_none() {
            let cfg = match &self.config_path_override {
                Some(p) => Config::from_file(p.clone())
                    .with_context(|| format!("loading config at {}", p.display()))?,
                None => Config::from_index_name(self.index_name.clone())
                    .context("loading config from default location")?,
            };
            self.config = Some(cfg);
        }
        Ok(self.config.as_mut().expect("just inserted"))
    }

    /// Re-load the YAML and re-sync it into the SQLite `store_collections` /
    /// `store_config` tables. Call after any CLI mutation that touches
    /// collections or contexts in the config.
    pub fn resync_config(&mut self) -> Result<()> {
        // Drop and reload the in-memory config from disk so we see the just-saved file.
        self.config = None;
        let config_snapshot = self.config_mut()?.clone();
        let store = self.store_mut()?;
        store.with_connection_mut(|conn| sync_config_to_db(conn, &config_snapshot))?;
        Ok(())
    }
}
