//! CLI lifecycle: lazy [`Store`] + [`Config`] managed by an [`IndexState`].
//!
//! Maps to qmd's module-level `let store / getStore()` helpers and the
//! `--index` / local-config resolution in `src/cli/qmd.ts` (lines 119–188).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use rmd_core::collections::{find_local_config_path, local_db_path, Config};
use rmd_core::store::path::{default_db_path, pwd};
use rmd_core::store::store_config::sync_config_to_db;
use rmd_core::Store;
use rmd_llm::config::{resolve_embed_model, resolve_generate_model, resolve_rerank_model};
use rmd_llm::{LlamaCpp, LlamaCppConfig, ModelResolutionConfig};

/// Holds the CLI's index selection plus the lazily-opened [`Store`],
/// [`Config`], and [`LlamaCpp`] handle. One per `rmd` invocation.
pub struct IndexState {
    index_name: String,
    db_path_override: Option<PathBuf>,
    config_path_override: Option<PathBuf>,
    store: Option<Store>,
    config: Option<Config>,
    llama: Option<Arc<LlamaCpp>>,
}

/// The three model URIs the CLI cares about, fully resolved against env vars
/// and YAML `models:` overrides.
#[derive(Debug, Clone)]
pub struct ResolvedModelUris {
    pub embed: String,
    pub generate: String,
    pub rerank: String,
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

    /// Immutable accessor; assumes [`Self::config_mut`] has already been called
    /// to populate the lazy cache. LLM commands trigger that on entry.
    pub fn config(&self) -> Result<&Config> {
        self.config
            .as_ref()
            .ok_or_else(|| anyhow!("config not loaded; call config_mut() first"))
    }

    /// Translate the YAML `models:` section into a [`ModelResolutionConfig`].
    /// Missing fields stay `None`; resolution against env vars and crate
    /// defaults happens inside [`LlamaCpp::new`].
    pub fn models_config(&self) -> Result<ModelResolutionConfig> {
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
    pub fn resolved_model_uris(&mut self) -> Result<ResolvedModelUris> {
        self.config_mut()?;
        let cfg = self.models_config()?;
        Ok(ResolvedModelUris {
            embed: resolve_embed_model(Some(&cfg)),
            generate: resolve_generate_model(Some(&cfg)),
            rerank: resolve_rerank_model(Some(&cfg)),
        })
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
