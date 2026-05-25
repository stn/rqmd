//! CLI lifecycle: lazy [`Store`] + [`Config`] + [`LlamaCpp`] managed by an [`IndexState`].
//!
//! Maps to qmd's module-level `let store / getStore()` helpers and the
//! `--index` / local-config resolution in `src/cli/qmd.ts` (lines 119–188).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rqmd_core::collections::{Config, find_local_config_path, local_db_path, sanitize_index_name};
use rqmd_core::llm::config::{ResolvedModels, resolve_models};
use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::types::ModelResolutionConfig;
use rqmd_core::paths::config_file_path;
use rqmd_core::store::path::{default_db_path, pwd};
use rqmd_core::store::store_config::sync_config_to_db;
use rqmd_core::{Llm, RqmdStoreOptions, Store};

/// Holds the CLI's index selection plus the lazily-opened [`Store`],
/// [`Config`], and [`LlamaCpp`] handle. One per `rqmd` invocation.
pub struct IndexState {
    index_name: String,
    db_path_override: Option<PathBuf>,
    config_path_override: Option<PathBuf>,
    store: Option<Store>,
    config: Option<Config>,
    llama: Option<Arc<LlamaCpp>>,
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
            llama: None,
        }
    }

    /// The active index name (default `"index"`). Used to annotate output
    /// links with `?index=` for non-default indexes (qmd `getActiveIndexName`).
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Switch the active index at runtime, dropping the lazily-opened store and
    /// config so the next access reopens against the new index. Mirrors qmd's
    /// `setIndexName` + `setConfigIndexName` (`qmd.ts:176-188`), used by `get`
    /// to honour a `?index=` carried in a `qmd://` link.
    ///
    /// NOTE: `db_path_override = None` deliberately diverges from qmd's
    /// `setIndexName` (which sets the override to the *resolved* path). rqmd
    /// resolves lazily in [`Self::db_path`] via `default_db_path(Some(name))`,
    /// so `None` correctly opens `<cache>/<name>.sqlite`. It also clears any
    /// stale local-config (`.rqmd`) override picked up by [`Self::new`] so the
    /// link's index — not the cwd — wins.
    pub fn set_index_name(&mut self, name: &str) {
        self.index_name = sanitize_index_name(name);
        self.db_path_override = None;
        self.config_path_override = None;
        self.store = None;
        self.config = None;
    }

    /// Path to the SQLite index that will be opened on first use.
    pub fn db_path(&self) -> Result<PathBuf> {
        if let Some(p) = &self.db_path_override {
            return Ok(p.clone());
        }
        default_db_path(Some(&self.index_name))
            .context("resolving default index path (set RQMD_INDEX_PATH or use --index)")
    }

    /// Build [`RqmdStoreOptions`] for the active index. Used by `rqmd mcp`, which
    /// drives the higher-level `RqmdStore` facade (the analog of qmd's
    /// `createStore`) rather than the lazy `Store`/`Config`/`LlamaCpp` trio.
    ///
    /// `config_path` is set only when the file exists, mirroring qmd's
    /// `existsSync(getConfigPath())` guard in `server.ts:558`. When absent, the
    /// store runs DB-only: collections already synced into SQLite by earlier
    /// `collection add` / `update` runs remain searchable.
    pub fn rqmd_store_options(&self) -> Result<RqmdStoreOptions> {
        let db_path = self.db_path()?;
        let candidate = match &self.config_path_override {
            Some(p) => p.clone(),
            None => config_file_path(&self.index_name),
        };
        let config_path = candidate.is_file().then_some(candidate);
        Ok(RqmdStoreOptions {
            db_path,
            config_path,
            config: None,
        })
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

    /// Immutable accessor; assumes [`Self::config_mut`] has already been called
    /// to populate the lazy cache. LLM commands trigger that on entry.
    fn config(&self) -> Result<&Config> {
        self.config
            .as_ref()
            .ok_or_else(|| anyhow!("config not loaded; call config_mut() first"))
    }

    /// Translate the YAML `models:` section into a [`ModelResolutionConfig`].
    /// Missing fields stay `None`; resolution against env vars and crate
    /// defaults happens inside [`LlamaCpp::new`].
    fn models_config(&self) -> Result<ModelResolutionConfig> {
        let data = self.config()?.data();
        Ok(ModelResolutionConfig {
            embed: data.models.as_ref().and_then(|m| m.embed.clone()),
            generate: data.models.as_ref().and_then(|m| m.generate.clone()),
            rerank: data.models.as_ref().and_then(|m| m.rerank.clone()),
        })
    }

    /// Lazily construct (and cache) an [`LlamaCpp`] handle. Subsequent calls
    /// return the same `Arc`, so multiple LLM commands invoked in the same
    /// process share one worker pool / model cache.
    ///
    /// Model URI resolution (env > YAML > crate default) is delegated to
    /// `LlamaCpp::new`; `ci_mode` / `parallelism` etc. come from
    /// `LlamaCppConfig::from_env`.
    ///
    /// Note: only the write path (`embed`) currently threads the YAML
    /// `models:` section through; the read path (`hybrid_query` /
    /// `vector_search_query`) calls `resolve_*_model(None)` internally and
    /// sees only env vars and defaults. See the plan's Known Limitations.
    pub fn llama_cpp(&mut self) -> Result<Arc<LlamaCpp>> {
        if self.llama.is_none() {
            // Force the lazy YAML load before we read it.
            self.config_mut()?;
            let model_cfg = self.models_config()?;
            let llama = LlamaCppConfig {
                embed_model: model_cfg.embed,
                generate_model: model_cfg.generate,
                rerank_model: model_cfg.rerank,
                ..LlamaCppConfig::from_env()
            };
            self.llama = Some(Arc::new(LlamaCpp::new(llama)));
        }
        Ok(self.llama.as_ref().expect("just inserted").clone())
    }

    /// Fully-resolved model URIs (env > YAML > crate default). The single
    /// source of truth shared by `pull` (which feeds them to `pull_models`)
    /// and any future caller that needs the URIs without instantiating an
    /// `LlamaCpp`.
    pub fn resolved_model_uris(&mut self) -> Result<ResolvedModels> {
        self.config_mut()?;
        let cfg = self.models_config()?;
        Ok(resolve_models(Some(&cfg)))
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

    /// Tear down LLM workers if instantiated. Mostly cosmetic for a one-shot
    /// CLI (the OS reclaims everything on exit), but matches the
    /// [`LlamaCpp`] / SDK contract that callers MUST dispose before drop —
    /// Rust has no async `Drop`. Called once from `main.rs::run` at end of
    /// dispatch (Ok and Err paths both).
    ///
    /// Note: `dispose` is a method on the [`Llm`] trait, not an inherent
    /// method on [`LlamaCpp`], so the trait must be in scope (see the
    /// `use rqmd_core::Llm` at the top of this file).
    pub async fn close(self) {
        if let Some(llama) = self.llama {
            llama.dispose().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn close_is_no_op_when_llama_not_constructed() {
        // Lazy state with no LLM yet — close() must short-circuit cleanly.
        let state = IndexState::new(None);
        assert!(state.llama.is_none());
        state.close().await;
    }

    #[tokio::test]
    async fn close_disposes_llama_cpp() {
        // Bypass YAML/db setup: inject a ci-mode LlamaCpp directly into
        // `state.llama`. We keep an `Arc::clone` to observe `is_disposed()`
        // after `close()` consumes `state`.
        let llama = Arc::new(LlamaCpp::new(LlamaCppConfig {
            ci_mode: true,
            ..LlamaCppConfig::default()
        }));
        let observer = Arc::clone(&llama);

        let mut state = IndexState::new(None);
        state.llama = Some(llama);

        assert!(!observer.is_disposed());
        state.close().await;
        assert!(observer.is_disposed());
    }

    #[test]
    fn set_index_name_switches_and_resets_lazy_handles() {
        let mut state = IndexState::new(Some("index"));
        assert_eq!(state.index_name(), "index");
        // Pretend a local-config override was picked up.
        state.db_path_override = Some(PathBuf::from("/tmp/local.sqlite"));
        state.config_path_override = Some(PathBuf::from("/tmp/local.yml"));

        state.set_index_name("release-notes");

        assert_eq!(state.index_name(), "release-notes");
        // Overrides cleared so the new index resolves from its name.
        assert!(state.db_path_override.is_none());
        assert!(state.config_path_override.is_none());
        // Lazy handles dropped so the next access reopens the new index.
        assert!(state.store.is_none());
        assert!(state.config.is_none());
    }

    #[test]
    fn set_index_name_sanitizes_path_like_names() {
        let mut state = IndexState::new(None);
        state.set_index_name("a/b");
        assert!(!state.index_name().contains('/'));
    }
}
