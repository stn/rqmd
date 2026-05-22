//! Context lookup and collection-level operations.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 2630–3010. Context is a string
//! attached to a `(collection, path-prefix)` pair (plus an optional global
//! context). The resolver joins every matching prefix — shortest first —
//! with double newlines.

use rusqlite::{Connection, OptionalExtension, params};

use super::Result;
use super::store_config::{
    delete_store_collection, get_store_collection, get_store_collections, get_store_contexts,
    get_store_global_context, remove_store_context, rename_store_collection,
    set_store_global_context, update_store_context,
};
use super::virtual_path::parse_virtual_path;

/// Get the context for a virtual `(collection, relative path)`.
pub fn get_context_for_path(
    conn: &Connection,
    collection: &str,
    path: &str,
) -> Result<Option<String>> {
    let Some(coll) = get_store_collection(conn, collection)? else {
        return Ok(None);
    };

    let mut parts: Vec<String> = Vec::new();

    if let Some(global) = get_store_global_context(conn)? {
        parts.push(global);
    }

    if let Some(map) = coll.context {
        let np = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };

        let mut matching: Vec<(String, String)> = Vec::new();
        for (prefix, ctx) in &map {
            let pp = if prefix.starts_with('/') {
                prefix.clone()
            } else {
                format!("/{prefix}")
            };
            if np.starts_with(&pp) {
                matching.push((pp, ctx.clone()));
            }
        }
        matching.sort_by_key(|(p, _)| p.len());
        for (_, ctx) in matching {
            parts.push(ctx);
        }
    }

    Ok(if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    })
}

/// Get the context for a `qmd://` or absolute filesystem path.
pub fn get_context_for_file(conn: &Connection, filepath: &str) -> Result<Option<String>> {
    if filepath.is_empty() {
        return Ok(None);
    }

    let collections = get_store_collections(conn)?;

    let (collection_name, relative_path): (String, String) = if filepath.starts_with("qmd://") {
        match parse_virtual_path(filepath) {
            Ok(vp) => (vp.collection, vp.path),
            Err(_) => return Ok(None),
        }
    } else {
        let mut found: Option<(String, String)> = None;
        for coll in &collections {
            if coll.path.is_empty() {
                continue;
            }
            let prefix = format!("{}/", coll.path);
            if filepath == coll.path {
                found = Some((coll.name.clone(), String::new()));
                break;
            }
            if let Some(rel) = filepath.strip_prefix(&prefix) {
                found = Some((coll.name.clone(), rel.to_string()));
                break;
            }
        }
        let Some(pair) = found else {
            return Ok(None);
        };
        pair
    };

    // Verify the document exists.
    let exists: Option<String> = conn
        .query_row(
            "SELECT path FROM documents WHERE collection = ? AND path = ? AND active = 1 LIMIT 1",
            params![collection_name, relative_path],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if exists.is_none() {
        return Ok(None);
    }

    get_context_for_path(conn, &collection_name, &relative_path)
}

// ============================================================================
// Collection ops backed by store_collections
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionNamePathPattern {
    pub name: String,
    pub pwd: String,
    pub glob_pattern: String,
}

pub fn get_collection_by_name(
    conn: &Connection,
    name: &str,
) -> Result<Option<CollectionNamePathPattern>> {
    let coll = get_store_collection(conn, name)?;
    Ok(coll.map(|c| CollectionNamePathPattern {
        name: c.name,
        pwd: c.path,
        glob_pattern: c.pattern,
    }))
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionListing {
    pub name: String,
    pub pwd: String,
    pub glob_pattern: String,
    pub doc_count: i64,
    pub active_count: i64,
    pub last_modified: Option<String>,
    pub include_by_default: bool,
}

pub fn list_collections(conn: &Connection) -> Result<Vec<CollectionListing>> {
    let mut out = Vec::new();
    for coll in get_store_collections(conn)? {
        let (doc_count, active_count, last_modified): (i64, i64, Option<String>) = conn
            .query_row(
                r#"SELECT
                       COUNT(d.id),
                       SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END),
                       MAX(d.modified_at)
                   FROM documents d WHERE d.collection = ?"#,
                params![coll.name],
                |row| {
                    Ok((
                        row.get::<_, Option<i64>>(0)?.unwrap_or(0),
                        row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .unwrap_or((0, 0, None));

        out.push(CollectionListing {
            name: coll.name,
            pwd: coll.path,
            glob_pattern: coll.pattern,
            doc_count,
            active_count,
            last_modified,
            include_by_default: coll.include_by_default,
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemovedCounts {
    pub deleted_docs: usize,
    pub cleaned_hashes: usize,
}

pub fn remove_collection(conn: &Connection, collection_name: &str) -> Result<RemovedCounts> {
    let deleted_docs = conn.execute(
        "DELETE FROM documents WHERE collection = ?",
        params![collection_name],
    )?;
    let cleaned_hashes = conn.execute(
        "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
        [],
    )?;
    delete_store_collection(conn, collection_name)?;
    Ok(RemovedCounts {
        deleted_docs,
        cleaned_hashes,
    })
}

pub fn rename_collection(conn: &Connection, old_name: &str, new_name: &str) -> Result<()> {
    conn.execute(
        "UPDATE documents SET collection = ? WHERE collection = ?",
        params![new_name, old_name],
    )?;
    rename_store_collection(conn, old_name, new_name)?;
    Ok(())
}

pub fn insert_context(
    conn: &Connection,
    collection_name: &str,
    path_prefix: &str,
    context: &str,
) -> Result<()> {
    update_store_context(conn, collection_name, path_prefix, context)
}

pub fn delete_context(conn: &Connection, collection: &str, path_prefix: &str) -> Result<usize> {
    Ok(usize::from(remove_store_context(
        conn,
        collection,
        path_prefix,
    )?))
}

pub fn delete_global_contexts(conn: &Connection) -> Result<usize> {
    let mut deleted = 0_usize;
    if get_store_global_context(conn)?.is_some() {
        set_store_global_context(conn, None)?;
        deleted += 1;
    }
    for coll in get_store_collections(conn)? {
        if remove_store_context(conn, &coll.name, "")? {
            deleted += 1;
        }
    }
    Ok(deleted)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathContextEntry {
    pub collection_name: String,
    pub path_prefix: String,
    pub context: String,
}

pub fn list_path_contexts(conn: &Connection) -> Result<Vec<PathContextEntry>> {
    let mut out: Vec<PathContextEntry> = get_store_contexts(conn)?
        .into_iter()
        .map(|e| PathContextEntry {
            collection_name: e.collection,
            path_prefix: e.path,
            context: e.context,
        })
        .collect();

    out.sort_by(|a, b| {
        a.collection_name
            .cmp(&b.collection_name)
            .then_with(|| b.path_prefix.len().cmp(&a.path_prefix.len()))
            .then_with(|| a.path_prefix.cmp(&b.path_prefix))
    });
    Ok(out)
}

pub fn get_all_collections(conn: &Connection) -> Result<Vec<String>> {
    Ok(get_store_collections(conn)?
        .into_iter()
        .map(|c| c.name)
        .collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionWithoutContext {
    pub name: String,
    pub pwd: String,
    pub doc_count: i64,
}

pub fn get_collections_without_context(conn: &Connection) -> Result<Vec<CollectionWithoutContext>> {
    let mut out = Vec::new();
    for coll in get_store_collections(conn)? {
        let empty = match &coll.context {
            None => true,
            Some(m) => m.is_empty(),
        };
        if !empty {
            continue;
        }
        let doc_count: i64 = conn
            .query_row(
                "SELECT COUNT(d.id) FROM documents d WHERE d.collection = ? AND d.active = 1",
                params![coll.name],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0);
        out.push(CollectionWithoutContext {
            name: coll.name,
            pwd: coll.path,
            doc_count,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn get_top_level_paths_without_context(
    conn: &Connection,
    collection_name: &str,
) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT DISTINCT path FROM documents WHERE collection = ? AND active = 1")?;
    let paths: Vec<String> = stmt
        .query_map(params![collection_name], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut tops: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for p in paths {
        let top = p.split('/').next().unwrap_or("").to_string();
        if !top.is_empty() && top != p {
            tops.insert(top);
        }
    }

    let Some(coll) = get_store_collection(conn, collection_name)? else {
        return Ok(tops.into_iter().collect());
    };
    if let Some(ctx) = coll.context {
        for prefix in ctx.keys() {
            let cleaned = prefix
                .trim_start_matches('/')
                .split('/')
                .next()
                .unwrap_or("");
            tops.remove(cleaned);
        }
    }

    Ok(tops.into_iter().collect())
}
