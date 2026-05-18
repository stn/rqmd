//! `rmd update` — re-index every configured collection.
//!
//! Maps to qmd's `qmd update` in `src/cli/qmd.ts` (lines 630–721, 3542–3544),
//! minus the per-collection `update:` shell-out (skipped per plan).

use std::path::PathBuf;

use anyhow::Result;
use rmd_core::store::cache::clear_cache;
use rmd_core::store::context::list_collections;
use rmd_core::store::reindex::reindex_collection;

use crate::color::Palette;
use crate::state::IndexState;

pub fn run(state: &mut IndexState, p: &Palette) -> Result<()> {
    // Snapshot collection list as owned values so the connection borrow used
    // by `reindex_collection` doesn't overlap with `state.config_mut()`.
    let snapshot: Vec<(String, PathBuf, String, Vec<String>)> = state
        .config_mut()?
        .list_collections()
        .iter()
        .map(|c| {
            (
                c.name.to_string(),
                PathBuf::from(&c.collection.path),
                c.collection.pattern.clone(),
                c.collection.ignore.clone().unwrap_or_default(),
            )
        })
        .collect();

    let store = state.store_mut()?;
    let db_empty = store.with_connection(list_collections)?.is_empty();

    if snapshot.is_empty() {
        if db_empty {
            println!(
                "{}No collections found. Run 'rmd collection add .' to index markdown files.{}",
                p.dim(),
                p.reset()
            );
            return Ok(());
        }
        // Config empty but DB has rows — let the store catch up via the
        // normal resync path.
        state.resync_config()?;
    }

    // Clear stale LLM cache once per update pass (mirrors qmd line 636).
    state.store_mut()?.with_connection(clear_cache)?;

    println!(
        "{}Updating {} collection(s)...{}\n",
        p.bold(),
        snapshot.len(),
        p.reset()
    );

    for (i, (name, path, pattern, ignore)) in snapshot.iter().enumerate() {
        println!(
            "{}[{}/{}]{} {}{name}{} {}({pattern}){}",
            p.cyan(),
            i + 1,
            snapshot.len(),
            p.reset(),
            p.bold(),
            p.reset(),
            p.dim(),
            p.reset()
        );
        println!("Collection: {} ({pattern})", path.display());

        let store = state.store_mut()?;
        let result = store.with_connection_mut(|conn| {
            reindex_collection(conn, path, pattern, name, ignore, |info| {
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
        println!();
    }

    println!("{}✓ All collections updated.{}", p.green(), p.reset());
    Ok(())
}
