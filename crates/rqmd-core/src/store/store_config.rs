//! Access to the self-contained `store_collections` / `store_config` tables.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 962–1141. The DB is the source
//! of truth — the YAML config (from [`crate::collections`]) is supplementary
//! metadata. [`sync_config_to_db`] writes a YAML-loaded `Config` into the
//! `store_collections` and `store_config` tables.

use indexmap::IndexMap;
use rusqlite::{Connection, OptionalExtension, params};

use crate::collections::{Collection, Config, ContextMap};

use super::Result;

/// A collection row stored in `store_collections`. Mirrors TS
/// `NamedCollection` (`name` + flattened collection fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedStoreCollection {
    pub name: String,
    pub path: String,
    pub pattern: String,
    pub ignore: Option<Vec<String>>,
    pub include_by_default: bool,
    pub update: Option<String>,
    pub context: Option<ContextMap>,
}

impl NamedStoreCollection {
    fn from_row(name: String, row: StoreCollectionRow) -> Self {
        Self {
            name,
            path: row.path,
            pattern: row.pattern,
            ignore: row
                .ignore_patterns
                .and_then(|s| serde_json::from_str(&s).ok()),
            include_by_default: row.include_by_default,
            update: row.update_command,
            context: row.context.and_then(|s| serde_json::from_str(&s).ok()),
        }
    }
}

#[derive(Debug)]
struct StoreCollectionRow {
    path: String,
    pattern: String,
    ignore_patterns: Option<String>,
    include_by_default: bool,
    update_command: Option<String>,
    context: Option<String>,
}

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, StoreCollectionRow)> {
    let include = row.get::<_, i64>(4)? != 0;
    Ok((
        row.get::<_, String>(0)?,
        StoreCollectionRow {
            path: row.get(1)?,
            pattern: row.get(2)?,
            ignore_patterns: row.get(3)?,
            include_by_default: include,
            update_command: row.get(5)?,
            context: row.get(6)?,
        },
    ))
}

const SELECT_COLS: &str =
    "name, path, pattern, ignore_patterns, include_by_default, update_command, context";

pub fn get_store_collections(conn: &Connection) -> Result<Vec<NamedStoreCollection>> {
    let mut stmt = conn.prepare(&format!("SELECT {SELECT_COLS} FROM store_collections"))?;
    let rows = stmt
        .query_map([], map_row)?
        .filter_map(|r| r.ok())
        .map(|(name, row)| NamedStoreCollection::from_row(name, row))
        .collect();
    Ok(rows)
}

pub fn get_store_collection(conn: &Connection, name: &str) -> Result<Option<NamedStoreCollection>> {
    let row = conn
        .query_row(
            &format!("SELECT {SELECT_COLS} FROM store_collections WHERE name = ?"),
            params![name],
            map_row,
        )
        .optional()?;
    Ok(row.map(|(n, r)| NamedStoreCollection::from_row(n, r)))
}

pub fn get_store_global_context(conn: &Connection) -> Result<Option<String>> {
    let value: Option<String> = conn
        .query_row(
            "SELECT value FROM store_config WHERE key = 'global_context'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(value.filter(|s| !s.is_empty()))
}

pub fn set_store_global_context(conn: &Connection, value: Option<&str>) -> Result<()> {
    match value {
        Some(v) => {
            conn.execute(
                "INSERT OR REPLACE INTO store_config(key, value) VALUES ('global_context', ?)",
                params![v],
            )?;
        }
        None => {
            conn.execute("DELETE FROM store_config WHERE key = 'global_context'", [])?;
        }
    }
    Ok(())
}

/// Flattened view of every (collection, path, context) tuple in the DB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreContextEntry {
    pub collection: String,
    pub path: String,
    pub context: String,
}

pub fn get_store_contexts(conn: &Connection) -> Result<Vec<StoreContextEntry>> {
    let mut out = Vec::new();

    if let Some(global) = get_store_global_context(conn)? {
        out.push(StoreContextEntry {
            collection: "*".into(),
            path: "/".into(),
            context: global,
        });
    }

    let mut stmt =
        conn.prepare("SELECT name, context FROM store_collections WHERE context IS NOT NULL")?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (name, ctx_json) in rows {
        if let Ok(map) = serde_json::from_str::<IndexMap<String, String>>(&ctx_json) {
            for (path, context) in map {
                out.push(StoreContextEntry {
                    collection: name.clone(),
                    path,
                    context,
                });
            }
        }
    }

    Ok(out)
}

pub fn upsert_store_collection(
    conn: &Connection,
    name: &str,
    collection: &Collection,
) -> Result<()> {
    let ignore_json = collection
        .ignore
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let context_json = collection
        .context
        .as_ref()
        .map(|c| serde_json::to_string(c).unwrap_or_default());
    let include = if matches!(collection.include_by_default, Some(false)) {
        0_i64
    } else {
        1_i64
    };
    let pattern = if collection.pattern.is_empty() {
        super::DEFAULT_GLOB.to_string()
    } else {
        collection.pattern.clone()
    };

    conn.execute(
        r#"INSERT INTO store_collections
              (name, path, pattern, ignore_patterns, include_by_default, update_command, context)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               path = excluded.path,
               pattern = excluded.pattern,
               ignore_patterns = excluded.ignore_patterns,
               include_by_default = excluded.include_by_default,
               update_command = excluded.update_command,
               context = excluded.context"#,
        params![
            name,
            collection.path,
            pattern,
            ignore_json,
            include,
            collection.update,
            context_json,
        ],
    )?;
    Ok(())
}

pub fn delete_store_collection(conn: &Connection, name: &str) -> Result<bool> {
    let n = conn.execute(
        "DELETE FROM store_collections WHERE name = ?",
        params![name],
    )?;
    Ok(n > 0)
}

pub fn rename_store_collection(conn: &Connection, old: &str, new: &str) -> Result<bool> {
    let n = conn.execute(
        "UPDATE store_collections SET name = ? WHERE name = ?",
        params![new, old],
    )?;
    Ok(n > 0)
}

pub fn update_store_context(
    conn: &Connection,
    collection: &str,
    path_prefix: &str,
    context: &str,
) -> Result<()> {
    let existing = get_store_collection(conn, collection)?;
    let mut map = existing.and_then(|c| c.context).unwrap_or_default();
    map.insert(path_prefix.to_string(), context.to_string());
    let json = serde_json::to_string(&map).unwrap_or_default();
    conn.execute(
        "UPDATE store_collections SET context = ? WHERE name = ?",
        params![json, collection],
    )?;
    Ok(())
}

pub fn remove_store_context(
    conn: &Connection,
    collection: &str,
    path_prefix: &str,
) -> Result<bool> {
    let Some(existing) = get_store_collection(conn, collection)? else {
        return Ok(false);
    };
    let Some(mut map) = existing.context else {
        return Ok(false);
    };
    if map.shift_remove(path_prefix).is_none() {
        return Ok(false);
    }
    let json = if map.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&map).unwrap_or_default())
    };
    conn.execute(
        "UPDATE store_collections SET context = ? WHERE name = ?",
        params![json, collection],
    )?;
    Ok(true)
}

/// Push the in-memory YAML [`Config`] into `store_collections` / `store_config`.
/// Mirrors `syncConfigToDb` (`store.ts:…`). Removes rows for collections that
/// no longer exist in the config.
pub fn sync_config_to_db(conn: &Connection, config: &Config) -> Result<()> {
    // Skip the sync when the config is byte-identical to what was last written.
    // Mirrors qmd `syncConfigToDb` (`store.ts:1101-1133`): the early return
    // protects DB-only mutations across a re-open with an unchanged config and
    // avoids running the delete-not-in-config pass needlessly.
    let config_json = serde_json::to_string(config.data()).unwrap_or_default();
    let hash = crate::store::documents::hash_content(&config_json);
    let existing_hash: Option<String> = conn
        .query_row(
            "SELECT value FROM store_config WHERE key = 'config_hash'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if existing_hash.as_deref() == Some(hash.as_str()) {
        return Ok(());
    }

    set_store_global_context(conn, config.data().global_context.as_deref())?;

    let mut keep: Vec<String> = Vec::new();
    for (name, coll) in &config.data().collections {
        upsert_store_collection(conn, name, coll)?;
        keep.push(name.clone());
    }

    if keep.is_empty() {
        conn.execute("DELETE FROM store_collections", [])?;
    } else {
        let placeholders = keep.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!("DELETE FROM store_collections WHERE name NOT IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            keep.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        stmt.execute(params.as_slice())?;
    }

    conn.execute(
        "INSERT OR REPLACE INTO store_config(key, value) VALUES ('config_hash', ?)",
        params![hash],
    )?;

    Ok(())
}
