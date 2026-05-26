//! `rqmd init` — create a project-local index (`.rqmd/index.yml` + `index.sqlite`).
//!
//! Port of qmd's `initLocalIndex` (`tobi/qmd/src/cli/qmd.ts`). Refuses to run in
//! `$HOME` (the global index is created on demand), creates a `.rqmd/` directory
//! in the current project, seeds the `models:` section with env/default-resolved
//! URIs, and syncs the (empty) config into a fresh SQLite store. Afterwards the
//! existing upward `.rqmd` discovery in [`crate::state::IndexState::new`] picks
//! the local index up for all other commands run inside the tree.
//!
//! `--index` is ignored: `init` is always local, matching qmd.

use anyhow::{Context, Result, bail};

use rqmd_core::Store;
use rqmd_core::collections::{Config, ModelsConfig};
use rqmd_core::llm::config::resolve_models;
use rqmd_core::llm::types::ModelResolutionConfig;
use rqmd_core::paths::rqmd_homedir;
use rqmd_core::store::path::{pwd, real_path};
use rqmd_core::store::store_config::sync_config_to_db;

use crate::color::Palette;

pub fn run(p: &Palette) -> Result<()> {
    let cwd = pwd();

    // Refuse to initialize in $HOME — the global index lives there and is
    // created on demand. Compare canonicalized paths so symlinked / drive-cased
    // HOMEs still match (qmd `sameDirectory`).
    if real_path(&cwd) == real_path(&rqmd_homedir()) {
        bail!(
            "Refusing to initialize a local index in $HOME. The global index is \
             automatically created; run `rqmd collection add <path>` for the global \
             index, or run `rqmd init` inside a project folder."
        );
    }

    let rqmd_dir = cwd.join(".rqmd");
    std::fs::create_dir_all(&rqmd_dir)
        .with_context(|| format!("creating {}", rqmd_dir.display()))?;

    // Prefer an existing `.yaml`; otherwise create `.yml` (qmd parity).
    let yaml = rqmd_dir.join("index.yaml");
    let config_path = if yaml.is_file() {
        yaml
    } else {
        rqmd_dir.join("index.yml")
    };
    let db_path = rqmd_dir.join("index.sqlite");

    // Always (re)write the models section, resolving each slot as
    // existing YAML > env > crate default. A fresh config has no models, so this
    // is env > default; re-running on an existing config keeps explicit values
    // and only fills empty slots (qmd `resolveModels()`).
    let mut cfg = Config::from_file(&config_path)
        .with_context(|| format!("loading config at {}", config_path.display()))?;
    let existing = ModelResolutionConfig {
        embed: cfg.data().models.as_ref().and_then(|m| m.embed.clone()),
        generate: cfg.data().models.as_ref().and_then(|m| m.generate.clone()),
        rerank: cfg.data().models.as_ref().and_then(|m| m.rerank.clone()),
        // `resolve_models` only inspects the three URI fields, so leave the
        // expand-prompt knobs at their defaults here — they're not URIs.
        ..Default::default()
    };
    let resolved = resolve_models(Some(&existing));
    // Preserve any user-supplied `expand:` block verbatim on rewrite. `init`
    // only canonicalises model URIs; it doesn't generate `expand` defaults.
    let preserved_expand = cfg.data().models.as_ref().and_then(|m| m.expand.clone());
    cfg.set_models(ModelsConfig {
        embed: Some(resolved.embed),
        rerank: Some(resolved.rerank),
        generate: Some(resolved.generate),
        expand: preserved_expand,
    })
    .with_context(|| format!("writing config at {}", config_path.display()))?;

    // Create the store (file + schema) and push the config into it.
    let mut store =
        Store::open(&db_path).with_context(|| format!("opening index at {}", db_path.display()))?;
    store
        .with_connection_mut(|conn| sync_config_to_db(conn, &cfg))
        .context("syncing config into the new local index")?;

    println!("Created local index at {}", rqmd_dir.display());
    println!("{}ready to go with new local index{}", p.green(), p.reset());
    Ok(())
}
