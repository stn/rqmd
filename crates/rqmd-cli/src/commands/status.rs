//! `rqmd status` — index + collection health summary.
//!
//! Maps to qmd's `qmd status` in `src/cli/qmd.ts` (lines 393–628). The native
//! device probe behind `QMD_STATUS_DEVICE_PROBE=1` is best-effort (see
//! [`device_mode`]).

use anyhow::Result;
use rqmd_core::store::context::list_collections;

use crate::color::Palette;
use crate::format_helpers::{format_bytes, format_time_ago};
use crate::state::IndexState;

pub fn run(state: &mut IndexState, p: &Palette) -> Result<()> {
    let db_path = state.db_path()?;

    // Snapshot YAML side first (immutable borrow + clones).
    let yaml_contexts: Vec<(String, String, String)> = state
        .config_mut()?
        .list_all_contexts()
        .into_iter()
        .map(|e| {
            (
                e.collection.to_string(),
                e.path.to_string(),
                e.context.to_string(),
            )
        })
        .collect();
    let yaml_no_update: Vec<String> = state
        .config_mut()?
        .list_collections()
        .iter()
        .filter(|c| c.collection.update.is_none())
        .map(|c| c.name.to_string())
        .collect();

    let store = state.store_mut()?;
    let collections = store.with_connection(list_collections)?;

    let (total_docs, vector_count, most_recent): (i64, i64, Option<String>) = store
        .with_connection(|conn| {
            let total: i64 =
                conn.query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                    r.get(0)
                })?;
            let vecs: i64 = conn
                .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
                .unwrap_or(0);
            let last: Option<String> = conn
                .query_row(
                    "SELECT MAX(modified_at) FROM documents WHERE active = 1",
                    [],
                    |r| r.get::<_, Option<String>>(0),
                )
                .unwrap_or(None);
            Ok::<_, rqmd_core::store::Error>((total, vecs, last))
        })?;

    let index_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    println!("{}rqmd Status{}\n", p.bold(), p.reset());
    println!("Index: {}", db_path.display());
    println!("Size:  {}", format_bytes(index_size));
    println!();

    println!("{}Documents{}", p.bold(), p.reset());
    println!("  Total:    {total_docs} files indexed");
    println!("  Vectors:  {vector_count} embedded");
    if let Some(last) = &most_recent {
        println!("  Updated:  {}", format_time_ago(last));
    }

    // Group contexts by collection for display.
    let mut by_collection: std::collections::BTreeMap<String, Vec<(String, String)>> =
        std::collections::BTreeMap::new();
    for (collection, path, context) in &yaml_contexts {
        by_collection
            .entry(collection.clone())
            .or_default()
            .push((path.clone(), context.clone()));
    }

    if collections.is_empty() {
        println!(
            "\n{}No collections. Run 'rqmd collection add .' to index markdown files.{}",
            p.dim(),
            p.reset()
        );
    } else {
        println!("\n{}Collections{}", p.bold(), p.reset());
        for coll in &collections {
            let last_mod = coll
                .last_modified
                .as_deref()
                .map(format_time_ago)
                .unwrap_or_else(|| "never".to_string());
            let contexts = by_collection.get(&coll.name);

            println!(
                "  {}{}{} {}(qmd://{}/){}",
                p.cyan(),
                coll.name,
                p.reset(),
                p.dim(),
                coll.name,
                p.reset()
            );
            println!(
                "    {}Pattern:{}  {}",
                p.dim(),
                p.reset(),
                coll.glob_pattern
            );
            println!(
                "    {}Files:{}    {} (updated {last_mod})",
                p.dim(),
                p.reset(),
                coll.active_count
            );
            if let Some(ctxs) = contexts
                && !ctxs.is_empty()
            {
                println!("    {}Contexts:{} {}", p.dim(), p.reset(), ctxs.len());
                for (path, context) in ctxs {
                    let path_display = if path.is_empty() || path == "/" {
                        "/".to_string()
                    } else {
                        format!("/{path}")
                    };
                    let preview = if context.len() > 60 {
                        format!("{}...", &context[..57])
                    } else {
                        context.clone()
                    };
                    println!("      {}{path_display}:{} {preview}", p.dim(), p.reset());
                }
            }
        }
    }

    let ast_status = rqmd_core::get_ast_status();
    println!("\n{}AST Grammars{}", p.bold(), p.reset());
    for lang in &ast_status.languages {
        let mark = if lang.available { "✓" } else { "✗" };
        println!("  {mark} {}", lang.language.as_str());
        if let Some(err) = &lang.error {
            println!("    {}{}{}", p.dim(), err, p.reset());
        }
    }

    // Device / Mode section (qmd.ts:551-590). The native probe behind
    // `QMD_STATUS_DEVICE_PROBE=1` is opt-in because, on machines with a broken
    // GPU loader, probing can abort the process; the default path only reports
    // the configured mode.
    println!("\n{}Device{}", p.bold(), p.reset());
    println!("  Mode:     {}", device_mode());
    if std::env::var("QMD_STATUS_DEVICE_PROBE").as_deref() == Ok("1") {
        println!("  Status:   probing native llama backend...");
        // Real GPU/VRAM enumeration is not yet wired through the llama.cpp
        // bindings; report gracefully rather than aborting.
        println!(
            "  Status:   {}device probe not yet implemented{}",
            p.dim(),
            p.reset()
        );
    } else {
        println!(
            "  Status:   {}not probed{} (set QMD_STATUS_DEVICE_PROBE=1 to test GPU/CPU backend)",
            p.dim(),
            p.reset()
        );
    }

    // MCP daemon status (qmd.ts:423-437): report whether a background HTTP
    // daemon is running (via its PID file under the cache dir).
    println!("\n{}MCP{}", p.bold(), p.reset());
    match crate::commands::mcp::running_daemon_pid() {
        Some(pid) => println!("  Daemon:   running (PID {pid})"),
        None => println!("  Daemon:   {}not running{}", p.dim(), p.reset()),
    }

    // Tips section.
    let collections_without_context: Vec<String> = collections
        .iter()
        .filter(|c| by_collection.get(&c.name).is_none_or(|v| v.is_empty()))
        .map(|c| c.name.clone())
        .collect();
    let mut tips: Vec<String> = Vec::new();
    if !collections_without_context.is_empty() {
        let head = collections_without_context
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = if collections_without_context.len() > 3 {
            format!(" +{} more", collections_without_context.len() - 3)
        } else {
            String::new()
        };
        tips.push(format!(
            "Add context to collections for better search results: {head}{more}"
        ));
        tips.push(format!(
            "  {}rqmd context add qmd://<name>/ \"What this collection contains\"{}",
            p.dim(),
            p.reset()
        ));
    }
    if !yaml_no_update.is_empty() && collections.len() > 1 {
        let head = yaml_no_update
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = if yaml_no_update.len() > 3 {
            format!(" +{} more", yaml_no_update.len() - 3)
        } else {
            String::new()
        };
        tips.push(format!(
            "Set update commands to track them with git: {head}{more}"
        ));
        tips.push(format!(
            "  {}rqmd collection update-cmd <name> 'git pull'{}",
            p.dim(),
            p.reset()
        ));
    }
    if !tips.is_empty() {
        println!("\n{}Tips{}", p.bold(), p.reset());
        for tip in tips {
            println!("  {tip}");
        }
    }

    Ok(())
}

/// Configured device mode for the status `Device` section. Mirrors qmd's
/// `configuredGpuMode` (`qmd.ts:553-557`): `CPU forced` when `QMD_FORCE_CPU` is
/// set to a truthy value (rqmd's global `--no-gpu` sets `QMD_FORCE_CPU=1`), else
/// the explicit `QMD_LLAMA_GPU` value, else `auto`.
fn device_mode() -> String {
    if let Ok(v) = std::env::var("QMD_FORCE_CPU") {
        let t = v.trim().to_ascii_lowercase();
        let falsey = matches!(
            t.as_str(),
            "false" | "off" | "none" | "disable" | "disabled" | "0"
        );
        if !t.is_empty() && !falsey {
            return "CPU forced (QMD_FORCE_CPU)".to_string();
        }
    }
    if let Ok(v) = std::env::var("QMD_LLAMA_GPU") {
        let t = v.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    "auto".to_string()
}
