//! `rqmd ls` — list collections, or files within a collection.
//!
//! Maps to qmd's `qmd ls` in `src/cli/qmd.ts` (lines 1340–1498).

use anyhow::{anyhow, Result};
use rqmd_core::db::params;
use rqmd_core::store::context::list_collections;
use rqmd_core::store::virtual_path::{is_virtual_path, parse_virtual_path};

use crate::cli::LsArgs;
use crate::color::Palette;
use crate::format_helpers::{format_bytes, format_ls_time};
use crate::state::IndexState;

pub fn run(a: LsArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let store = state.store_mut()?;
    let collections = store.with_connection(list_collections)?;

    let Some(arg) = a.path else {
        // No argument: list all collections.
        if collections.is_empty() {
            println!("No collections found. Run 'rqmd collection add .' to index files.");
            return Ok(());
        }
        println!("{}Collections:{}\n", p.bold(), p.reset());
        for coll in &collections {
            println!(
                "  {}qmd://{}{}{}/{}  {}({} files){}",
                p.dim(),
                p.reset(),
                p.cyan(),
                coll.name,
                p.reset(),
                p.dim(),
                coll.active_count,
                p.reset()
            );
        }
        return Ok(());
    };

    // Parse the argument into (collection_name, prefix).
    let (collection_name, path_prefix) = resolve_collection_arg(&arg, &collections)?;

    // Make sure the collection actually exists.
    if !collections.iter().any(|c| c.name == collection_name) {
        return Err(anyhow!(
            "Collection not found: {collection_name}\nRun 'rqmd ls' to see available collections."
        ));
    }

    let store = state.store_mut()?;
    let files: Vec<(String, String, String, i64)> = store.with_connection(|conn| {
        let (sql, like) = match &path_prefix {
            Some(prefix) => (
                "SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) AS size
                 FROM documents d
                 JOIN content ct ON d.hash = ct.hash
                 WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
                 ORDER BY d.path",
                Some(format!("{prefix}%")),
            ),
            None => (
                "SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) AS size
                 FROM documents d
                 JOIN content ct ON d.hash = ct.hash
                 WHERE d.collection = ? AND d.active = 1
                 ORDER BY d.path",
                None,
            ),
        };
        let mut stmt = conn.prepare(sql)?;
        let rows: Vec<(String, String, String, i64)> = if let Some(like) = like {
            stmt.query_map(params![collection_name, like], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect()
        } else {
            stmt.query_map(params![collection_name], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect()
        };
        Ok::<_, rqmd_core::store::Error>(rows)
    })?;

    if files.is_empty() {
        match &path_prefix {
            Some(pfx) => println!("No files found under qmd://{collection_name}/{pfx}"),
            None => println!("No files found in collection: {collection_name}"),
        }
        return Ok(());
    }

    let max_size = files
        .iter()
        .map(|(_, _, _, sz)| format_bytes(*sz as u64).len())
        .max()
        .unwrap_or(0);

    for (path, _title, modified_at, size) in &files {
        let size_str = format_bytes(*size as u64);
        let pad = " ".repeat(max_size.saturating_sub(size_str.len()));
        let time_str = format_ls_time(modified_at);
        println!(
            "{pad}{size_str}  {time_str}  {}qmd://{}/{}{}{}{}",
            p.dim(),
            collection_name,
            p.reset(),
            p.cyan(),
            path,
            p.reset()
        );
    }
    Ok(())
}

/// Parse a `ls` argument into `(collection_name, optional_path_prefix)`.
/// Supports `qmd://name/...`, `//name/...`, `name`, and `name/sub/path`.
fn resolve_collection_arg(
    arg: &str,
    known: &[rqmd_core::store::context::CollectionListing],
) -> Result<(String, Option<String>)> {
    if is_virtual_path(arg) {
        let vp = parse_virtual_path(arg).map_err(|_| anyhow!("Invalid virtual path: {arg}"))?;
        let prefix = if vp.path.is_empty() {
            None
        } else {
            Some(vp.path)
        };
        return Ok((vp.collection, prefix));
    }

    if let Some(rest) = arg.strip_prefix('/') {
        // Longest-prefix match against collection names (qmd preserves this
        // historical alias for absolute-style paths).
        let normalized = rest.trim_end_matches('/');
        if let Some(c) = longest_prefix_match(normalized, known) {
            let rel = normalized[c.len()..].trim_start_matches('/');
            return Ok((
                c.to_string(),
                if rel.is_empty() {
                    None
                } else {
                    Some(rel.to_string())
                },
            ));
        }
        return Ok((normalized.to_string(), None));
    }

    let parts: Vec<&str> = arg.splitn(2, '/').collect();
    let name = parts[0].to_string();
    let prefix = parts.get(1).map(|s| s.to_string());
    Ok((name, prefix))
}

fn longest_prefix_match<'a>(
    s: &str,
    known: &'a [rqmd_core::store::context::CollectionListing],
) -> Option<&'a str> {
    known
        .iter()
        .filter(|c| s == c.name || s.starts_with(&format!("{}/", c.name)))
        .max_by_key(|c| c.name.len())
        .map(|c| c.name.as_str())
}
