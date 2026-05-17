//! `rmd collection ...` — manage indexed folders.
//!
//! Maps to qmd's `collection` subcommands in `src/cli/qmd.ts`
//! (lines 1521–1642, 3397–3535).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use rmd_core::collections::{CollectionSettings, IncludeByDefaultField, UpdateField};
use rmd_core::store::context as store_context;
use rmd_core::store::path::{pwd, real_path};
use rmd_core::store::reindex;

use crate::cli::{
    CollectionAddArgs, CollectionCmd, CollectionNameArg, CollectionRemoveArgs,
    CollectionRenameArgs, CollectionShowArgs, CollectionUpdateCmdArgs,
};
use crate::color::Palette;
use crate::format_helpers::format_time_ago;
use crate::state::IndexState;

pub fn run(cmd: CollectionCmd, state: &mut IndexState, p: &Palette) -> Result<()> {
    match cmd {
        CollectionCmd::List => list(state, p),
        CollectionCmd::Add(a) => add(a, state, p),
        CollectionCmd::Remove(a) => remove(a, state, p),
        CollectionCmd::Rename(a) => rename(a, state, p),
        CollectionCmd::Show(a) => show(a, state),
        CollectionCmd::UpdateCmd(a) => update_cmd(a, state, p),
        CollectionCmd::Include(a) => set_include(a, true, state, p),
        CollectionCmd::Exclude(a) => set_include(a, false, state, p),
    }
}

fn list(state: &mut IndexState, p: &Palette) -> Result<()> {
    // Pull pattern/ignore display from YAML; counts/timestamps from the DB.
    let yaml_excluded: Vec<(String, bool, Option<Vec<String>>)> = state
        .config_mut()?
        .list_collections()
        .iter()
        .map(|c| {
            (
                c.name.to_string(),
                c.collection.is_included_by_default(),
                c.collection.ignore.clone(),
            )
        })
        .collect();

    let store = state.store_mut()?;
    let collections = store.with_connection(|conn| store_context::list_collections(conn))?;

    if collections.is_empty() {
        println!("No collections found. Run 'rmd collection add .' to create one.");
        return Ok(());
    }

    println!(
        "{}Collections ({}):{}\n",
        p.bold(),
        collections.len(),
        p.reset()
    );

    for coll in &collections {
        let yaml = yaml_excluded.iter().find(|(n, _, _)| n == &coll.name);
        let excluded = yaml.map(|(_, inc, _)| !*inc).unwrap_or(false);
        let exclude_tag = if excluded {
            format!(" {}[excluded]{}", p.yellow(), p.reset())
        } else {
            String::new()
        };
        let updated = coll
            .last_modified
            .as_deref()
            .map(format_time_ago)
            .unwrap_or_else(|| "never".to_string());

        println!(
            "{}{}{} {}(qmd://{}/){}{}",
            p.cyan(),
            coll.name,
            p.reset(),
            p.dim(),
            coll.name,
            p.reset(),
            exclude_tag
        );
        println!("  {}Pattern:{}  {}", p.dim(), p.reset(), coll.glob_pattern);
        if let Some((_, _, Some(ignore))) = yaml
            && !ignore.is_empty()
        {
            println!("  {}Ignore:{}   {}", p.dim(), p.reset(), ignore.join(", "));
        }
        println!("  {}Files:{}    {}", p.dim(), p.reset(), coll.active_count);
        println!("  {}Updated:{}  {updated}", p.dim(), p.reset());
        println!();
    }
    Ok(())
}

fn add(a: CollectionAddArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let pwd_arg = a.path.as_deref().unwrap_or(".");
    let resolved: PathBuf = if pwd_arg == "." {
        pwd()
    } else {
        real_path(Path::new(pwd_arg))
    };
    let pattern = a.mask.as_deref().unwrap_or("**/*.md").to_string();

    let name = match a.name {
        Some(n) => n,
        None => resolved
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "root".to_string()),
    };

    let resolved_str = resolved.to_string_lossy().replace('\\', "/");

    // Pre-flight checks against existing YAML.
    {
        let cfg = state.config_mut()?;
        if cfg.get_collection(&name).is_some() {
            return Err(anyhow!(
                "{}Collection '{name}' already exists.{}\nUse a different name with --name <name>",
                p.yellow(),
                p.reset()
            ));
        }
        let dup = cfg
            .list_collections()
            .iter()
            .find(|c| c.collection.path == resolved_str && c.collection.pattern == pattern)
            .map(|c| c.name.to_string());
        if let Some(existing) = dup {
            return Err(anyhow!(
                "A collection already exists for this path and pattern: {existing} ({pattern})\n\
                 Use 'rmd update' to re-index it, or remove it first with 'rmd collection remove {existing}'"
            ));
        }
        cfg.add_collection(&name, resolved_str.clone(), Some(&pattern))
            .with_context(|| format!("adding collection '{name}'"))?;
    }
    state.resync_config()?;

    println!("Creating collection '{name}'...");

    let ignore = state
        .config_mut()?
        .get_collection(&name)
        .and_then(|c| c.collection.ignore.clone())
        .unwrap_or_default();

    let store = state.store_mut()?;
    let result = store.with_connection_mut(|conn| {
        reindex::reindex_collection(conn, &resolved, &pattern, &name, &ignore, |info| {
            eprint!("\rIndexing: {}/{}        ", info.current, info.total);
        })
    })?;
    eprintln!();
    println!(
        "Indexed: {} new, {} updated, {} unchanged, {} removed",
        result.indexed, result.updated, result.unchanged, result.removed
    );
    if result.orphaned_cleaned > 0 {
        println!(
            "Cleaned up {} orphaned content hash(es)",
            result.orphaned_cleaned
        );
    }
    println!(
        "{}✓{} Collection '{name}' created successfully",
        p.green(),
        p.reset()
    );
    Ok(())
}

fn remove(a: CollectionRemoveArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    {
        let cfg = state.config_mut()?;
        if cfg.get_collection(&a.name).is_none() {
            return Err(anyhow!(
                "{}Collection not found: {}{}\nRun 'rmd collection list' to see available collections.",
                p.yellow(),
                a.name,
                p.reset()
            ));
        }
    }

    let store = state.store_mut()?;
    let removed =
        store.with_connection_mut(|conn| store_context::remove_collection(conn, &a.name))?;

    let cfg = state.config_mut()?;
    cfg.remove_collection(&a.name)?;
    state.resync_config()?;

    println!("{}✓{} Removed collection '{}'", p.green(), p.reset(), a.name);
    println!("  Deleted {} documents", removed.deleted_docs);
    if removed.cleaned_hashes > 0 {
        println!(
            "  Cleaned up {} orphaned content hashes",
            removed.cleaned_hashes
        );
    }
    Ok(())
}

fn rename(a: CollectionRenameArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    {
        let cfg = state.config_mut()?;
        if cfg.get_collection(&a.old).is_none() {
            return Err(anyhow!(
                "{}Collection not found: {}{}\nRun 'rmd collection list' to see available collections.",
                p.yellow(),
                a.old,
                p.reset()
            ));
        }
        if cfg.get_collection(&a.new).is_some() {
            return Err(anyhow!(
                "{}Collection name already exists: {}{}\nChoose a different name or remove the existing collection first.",
                p.yellow(),
                a.new,
                p.reset()
            ));
        }
    }

    let store = state.store_mut()?;
    store.with_connection_mut(|conn| store_context::rename_collection(conn, &a.old, &a.new))?;

    let cfg = state.config_mut()?;
    cfg.rename_collection(&a.old, &a.new)?;
    state.resync_config()?;

    println!(
        "{}✓{} Renamed collection '{}' to '{}'",
        p.green(),
        p.reset(),
        a.old,
        a.new
    );
    println!(
        "  Virtual paths updated: {}qmd://{}/{} → {}qmd://{}/{}",
        p.cyan(),
        a.old,
        p.reset(),
        p.cyan(),
        a.new,
        p.reset()
    );
    Ok(())
}

fn show(a: CollectionShowArgs, state: &mut IndexState) -> Result<()> {
    let cfg = state.config_mut()?;
    let Some(coll) = cfg.get_collection(&a.name) else {
        return Err(anyhow!("Collection not found: {}", a.name));
    };
    let c = coll.collection;
    println!("Collection: {}", a.name);
    println!("  Path:     {}", c.path);
    println!("  Pattern:  {}", c.pattern);
    println!(
        "  Include:  {}",
        if c.is_included_by_default() {
            "yes (default)"
        } else {
            "no"
        }
    );
    if let Some(u) = &c.update {
        println!("  Update:   {u}");
    }
    if let Some(ctx) = &c.context {
        println!("  Contexts: {}", ctx.len());
    }
    Ok(())
}

fn update_cmd(a: CollectionUpdateCmdArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let joined = a.command.join(" ");
    let trimmed = joined.trim().to_string();
    let setting = if trimmed.is_empty() {
        UpdateField::Clear
    } else {
        UpdateField::Set(trimmed.clone())
    };
    let cfg = state.config_mut()?;
    let applied = cfg.update_collection_settings(
        &a.name,
        CollectionSettings {
            update: setting,
            include_by_default: IncludeByDefaultField::Keep,
        },
    )?;
    if !applied {
        return Err(anyhow!("Collection not found: {}", a.name));
    }
    state.resync_config()?;
    if trimmed.is_empty() {
        println!(
            "{}✓{} Cleared update command for '{}'",
            p.green(),
            p.reset(),
            a.name
        );
    } else {
        println!(
            "{}✓{} Set update command for '{}': {trimmed}",
            p.green(),
            p.reset(),
            a.name
        );
        println!(
            "{}Note:{} rmd does not currently execute `update:` scripts.",
            p.dim(),
            p.reset()
        );
    }
    Ok(())
}

fn set_include(
    a: CollectionNameArg,
    include: bool,
    state: &mut IndexState,
    p: &Palette,
) -> Result<()> {
    let setting = if include {
        IncludeByDefaultField::ResetToDefault
    } else {
        IncludeByDefaultField::SetFalse
    };
    let cfg = state.config_mut()?;
    let applied = cfg.update_collection_settings(
        &a.name,
        CollectionSettings {
            update: UpdateField::Keep,
            include_by_default: setting,
        },
    )?;
    if !applied {
        return Err(anyhow!("Collection not found: {}", a.name));
    }
    state.resync_config()?;
    let verb = if include {
        "included in"
    } else {
        "excluded from"
    };
    println!(
        "{}✓{} Collection '{}' {verb} default queries",
        p.green(),
        p.reset(),
        a.name
    );
    Ok(())
}
