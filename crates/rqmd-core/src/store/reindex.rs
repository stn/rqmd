//! Filesystem walk + index update.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 1247–1365 (`reindexCollection`,
//! plus its progress / result types). LLM-using embedding generation
//! (`generateEmbeddings`, lines 1511–1704) is deferred.
//!
//! **rqmd divergence**: unlike qmd (which walks with `fast-glob` and reads no
//! VCS ignore files), rqmd honours a dedicated `.qmdignore` file — per
//! directory and a global `~/.qmdignore` — using gitignore syntax. `.gitignore`
//! / `.ignore` are deliberately NOT consulted, so git/ripgrep config never
//! leaks into the index.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use globset::Glob;
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use rusqlite::Connection;

use super::docid::handelize;
use super::documents::{
    deactivate_document, extract_title, find_or_migrate_legacy_document, hash_content,
    insert_content, insert_document, update_document, update_document_title,
};
use super::maintenance::cleanup_orphaned_content;
use super::path::now_rfc3339;
use super::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexProgress {
    pub file: String,
    pub current: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReindexResult {
    pub indexed: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub orphaned_cleaned: usize,
}

/// Default directory excludes — matches TS line 1277.
const DEFAULT_EXCLUDE_DIRS: &[&str] =
    &["node_modules", ".git", ".cache", "vendor", "dist", "build"];

/// Global `.qmdignore` files applied to every collection, lowest precedence
/// (a per-directory `.qmdignore` overrides them). Currently just
/// `~/.qmdignore`, if it exists. Returned as a list so callers can pass it to
/// [`reindex_collection`]'s `extra_ignore_files`, and so tests can inject their
/// own paths instead of touching the real home directory.
pub fn global_qmdignore_files() -> Vec<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".qmdignore"))
        .filter(|p| p.exists())
        .into_iter()
        .collect()
}

/// Walk `collection_path` matching `glob_pattern`, syncing the result into
/// the index. `on_progress`, if provided, is called once per file.
///
/// `ignore_patterns` are added to the default exclude list and matched as
/// gitignore patterns. `extra_ignore_files` are additional global `.qmdignore`
/// files (e.g. `~/.qmdignore` via [`global_qmdignore_files`]) applied at the
/// lowest precedence — a per-directory `.qmdignore` overrides them.
///
/// `.gitignore` / `.ignore` and VCS-global ignores are NOT consulted; only
/// `.qmdignore` files, the default excludes, and `ignore_patterns` filter the
/// walk.
pub fn reindex_collection(
    conn: &mut Connection,
    collection_path: &Path,
    glob_pattern: &str,
    collection_name: &str,
    ignore_patterns: &[String],
    extra_ignore_files: &[PathBuf],
    mut on_progress: impl FnMut(&ReindexProgress),
) -> Result<ReindexResult> {
    let now = now_rfc3339();

    // Exclude-only override matcher: the default dir excludes + the user's
    // `ignore_patterns`. The include `glob_pattern` is matched separately
    // (below) rather than as an override whitelist: in the `ignore` crate an
    // override match has the highest precedence and short-circuits before any
    // `.qmdignore` is consulted, which would make `.qmdignore` file patterns
    // ineffective for matched files.
    let mut overrides = OverrideBuilder::new(collection_path);
    for d in DEFAULT_EXCLUDE_DIRS {
        let pat = format!("!**/{d}/**");
        overrides
            .add(&pat)
            .map_err(|e| Error::InvalidGlob(format!("{pat}: {e}")))?;
    }
    for p in ignore_patterns {
        let pat = format!("!{p}");
        overrides
            .add(&pat)
            .map_err(|e| Error::InvalidGlob(format!("{pat}: {e}")))?;
    }
    let overrides = overrides
        .build()
        .map_err(|e| Error::InvalidGlob(format!("override build: {e}")))?;

    // Include matcher for `glob_pattern`, applied per file during collection.
    let include = Glob::new(glob_pattern)
        .map_err(|e| Error::InvalidGlob(format!("{glob_pattern}: {e}")))?
        .compile_matcher();

    // Walk the tree, collecting file relative paths first so we can report
    // a `total` to the progress callback. `standard_filters(false)` disables
    // the `ignore` crate's `.gitignore` / `.ignore` / global-git / hidden /
    // parents handling; we re-enable only a dedicated `.qmdignore` (per
    // directory) plus the explicit global files in `extra_ignore_files`.
    let mut wb = WalkBuilder::new(collection_path);
    wb.standard_filters(false)
        .add_custom_ignore_filename(".qmdignore")
        .follow_links(false)
        .overrides(overrides);
    for f in extra_ignore_files {
        // Lowest precedence; a per-directory `.qmdignore` overrides these.
        // `add_ignore` returns `Option<Error>` for partial failures (e.g. a
        // bad glob in the file); ignore it so one bad line can't abort indexing.
        let _ = wb.add_ignore(f);
    }
    let walker = wb.build();

    let mut files: Vec<(std::path::PathBuf, String)> = Vec::new();
    for dent in walker.flatten() {
        if !dent.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let abs = dent.path();
        let rel = match abs.strip_prefix(collection_path) {
            Ok(r) => r.to_path_buf(),
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        // Skip dotfiles / dotted directories anywhere in the path.
        if rel_str.split('/').any(|seg| seg.starts_with('.')) {
            continue;
        }
        // Apply the include pattern (replaces the former override whitelist).
        if !include.is_match(&rel_str) {
            continue;
        }
        files.push((abs.to_path_buf(), rel_str));
    }

    let total = files.len();
    let mut indexed = 0_usize;
    let mut updated = 0_usize;
    let mut unchanged = 0_usize;
    let mut processed = 0_usize;
    let mut seen: HashSet<String> = HashSet::new();

    for (abs, rel_str) in files {
        let path_key = match handelize(&rel_str) {
            Ok(k) => k,
            Err(_) => {
                processed += 1;
                on_progress(&ReindexProgress {
                    file: rel_str.clone(),
                    current: processed,
                    total,
                });
                continue;
            }
        };
        seen.insert(path_key.clone());

        let content = match std::fs::read_to_string(&abs) {
            Ok(c) => c,
            Err(_) => {
                processed += 1;
                on_progress(&ReindexProgress {
                    file: rel_str,
                    current: processed,
                    total,
                });
                continue;
            }
        };

        if content.trim().is_empty() {
            processed += 1;
            continue;
        }

        let hash = hash_content(&content);
        let title = extract_title(&content, &rel_str);
        let mtime = file_mtime_rfc3339(&abs).unwrap_or_else(|| now.clone());
        let btime = file_btime_rfc3339(&abs).unwrap_or_else(|| now.clone());

        let existing = find_or_migrate_legacy_document(conn, collection_name, &path_key)?;
        if let Some(doc) = existing {
            if doc.hash == hash {
                if doc.title != title {
                    update_document_title(conn, doc.id, &title, &now)?;
                    updated += 1;
                } else {
                    unchanged += 1;
                }
            } else {
                insert_content(conn, &hash, &content, &now)?;
                update_document(conn, doc.id, &title, &hash, &mtime)?;
                updated += 1;
            }
        } else {
            insert_content(conn, &hash, &content, &now)?;
            insert_document(
                conn,
                collection_name,
                &path_key,
                &title,
                &hash,
                &btime,
                &mtime,
            )?;
            indexed += 1;
        }

        processed += 1;
        on_progress(&ReindexProgress {
            file: rel_str,
            current: processed,
            total,
        });
    }

    // Deactivate documents that no longer exist on disk.
    let active = super::documents::get_active_document_paths(conn, collection_name)?;
    let mut removed = 0_usize;
    for path in active {
        if !seen.contains(&path) {
            deactivate_document(conn, collection_name, &path)?;
            removed += 1;
        }
    }

    let orphaned_cleaned = cleanup_orphaned_content(conn)?;

    Ok(ReindexResult {
        indexed,
        updated,
        unchanged,
        removed,
        orphaned_cleaned,
    })
}

fn file_mtime_rfc3339(p: &Path) -> Option<String> {
    let meta = std::fs::metadata(p).ok()?;
    let mt = meta.modified().ok()?;
    let dur = mt.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(super::path::format_rfc3339_utc(
        dur.as_secs(),
        dur.subsec_millis(),
    ))
}

fn file_btime_rfc3339(p: &Path) -> Option<String> {
    let meta = std::fs::metadata(p).ok()?;
    let bt = meta.created().ok()?;
    let dur = bt.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(super::path::format_rfc3339_utc(
        dur.as_secs(),
        dur.subsec_millis(),
    ))
}
