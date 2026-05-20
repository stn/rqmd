//! Index status / health summaries.
//!
//! Port of `tobi/qmd`'s `src/store.ts`: `getStatus` (3989–4033) and
//! `getIndexHealth` (2006–2018). Both take an `&str` model — the
//! TS `DEFAULT_EMBED_MODEL` default is resolved at the call site (rqmd-core
//! intentionally has no LLM constants).

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension};

use super::embeddings::get_hashes_needing_embedding;
use super::path::{days_since_rfc3339, now_rfc3339};
use super::search::CollectionInfo;
use super::store_config::get_store_collections;
use super::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct IndexStatus {
    pub total_documents: i64,
    pub needs_embedding: i64,
    pub has_vector_index: bool,
    pub collections: Vec<CollectionInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexHealthInfo {
    pub needs_embedding: i64,
    pub total_docs: i64,
    pub days_stale: Option<i64>,
}

/// Per-index summary: per-collection counts, total active docs, count needing
/// embedding for `model`, and whether the `vectors_vec` table exists.
/// Collections are sorted by `last_updated` descending. Mirrors TS
/// `getStatus` (3989).
pub fn get_status(conn: &Connection, model: &str) -> Result<IndexStatus> {
    // Per-collection counts.
    let mut stmt = conn.prepare(
        "SELECT collection, COUNT(*) AS active_count, MAX(modified_at) AS last_doc_update
         FROM documents
         WHERE active = 1
         GROUP BY collection",
    )?;
    struct Row {
        name: String,
        count: i64,
        last: Option<String>,
    }
    let db_rows: Vec<Row> = stmt
        .query_map([], |row| {
            Ok(Row {
                name: row.get(0)?,
                count: row.get(1)?,
                last: row.get::<_, Option<String>>(2)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);

    // Lookup metadata from store_collections (path/pattern).
    let store_collections = get_store_collections(conn)?;
    let meta: HashMap<String, (String, String)> = store_collections
        .into_iter()
        .map(|c| (c.name, (c.path, c.pattern)))
        .collect();

    let now = now_rfc3339();
    let mut collections: Vec<CollectionInfo> = db_rows
        .into_iter()
        .map(|row| {
            let m = meta.get(&row.name);
            CollectionInfo {
                name: row.name,
                path: m.map(|x| x.0.clone()),
                pattern: m.map(|x| x.1.clone()),
                documents: row.count,
                last_updated: row.last.unwrap_or_else(|| now.clone()),
            }
        })
        .collect();
    // RFC 3339 strings sort correctly lexically.
    collections.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));

    let total_documents: i64 =
        conn.query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })?;
    let needs_embedding = get_hashes_needing_embedding(conn, None, model)?;
    let has_vector_index: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )
        .optional()?;

    Ok(IndexStatus {
        total_documents,
        needs_embedding,
        has_vector_index: has_vector_index.is_some(),
        collections,
    })
}

/// Lightweight health check: how many hashes need embedding, total active
/// doc count, and days since the most recent `modified_at`. Mirrors TS
/// `getIndexHealth` (2006).
pub fn get_index_health(conn: &Connection, model: &str) -> Result<IndexHealthInfo> {
    let needs_embedding = get_hashes_needing_embedding(conn, None, model)?;
    let total_docs: i64 =
        conn.query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })?;
    let most_recent: Option<String> = conn
        .query_row(
            "SELECT MAX(modified_at) FROM documents WHERE active = 1",
            [],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();
    let days_stale = most_recent.as_deref().and_then(days_since_rfc3339);

    Ok(IndexHealthInfo {
        needs_embedding,
        total_docs,
        days_stale,
    })
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections::Collection;
    use crate::store::embeddings::{ensure_vec_table, insert_embedding};
    use crate::store::store_config::upsert_store_collection;
    use crate::store::Store;
    use rusqlite::params;
    use tempfile::NamedTempFile;

    fn open_test_store() -> (NamedTempFile, Store) {
        let tmp = NamedTempFile::new().unwrap();
        let store = Store::open(tmp.path()).expect("open store");
        (tmp, store)
    }

    fn insert_doc(conn: &Connection, collection: &str, path: &str, hash: &str, modified_at: &str) {
        insert_doc_active(conn, collection, path, hash, modified_at, 1);
    }

    fn insert_doc_active(
        conn: &Connection,
        collection: &str,
        path: &str,
        hash: &str,
        modified_at: &str,
        active: i64,
    ) {
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)",
            params![hash, format!("body of {hash}"), modified_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
             VALUES (?, ?, 'title', ?, ?, ?, ?)",
            params![collection, path, hash, modified_at, modified_at, active],
        )
        .unwrap();
    }

    #[test]
    fn get_status_empty_db_has_zero_counts() {
        let (_t, store) = open_test_store();
        let s = get_status(&store.conn, "m").unwrap();
        assert_eq!(s.total_documents, 0);
        assert_eq!(s.needs_embedding, 0);
        assert!(!s.has_vector_index);
        assert!(s.collections.is_empty());
    }

    #[test]
    fn get_status_aggregates_and_sorts_collections() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c2", "a.md", "h1", "2024-01-01T00:00:00.000Z");
        insert_doc(&store.conn, "c2", "b.md", "h2", "2024-03-01T00:00:00.000Z");
        insert_doc(&store.conn, "c1", "a.md", "h3", "2024-02-01T00:00:00.000Z");

        let s = get_status(&store.conn, "m").unwrap();
        assert_eq!(s.total_documents, 3);
        assert_eq!(s.needs_embedding, 3); // none embedded yet
        assert!(!s.has_vector_index);
        // Sorted by last_updated desc: c2 (2024-03) before c1 (2024-02).
        assert_eq!(s.collections.len(), 2);
        assert_eq!(s.collections[0].name, "c2");
        assert_eq!(s.collections[0].documents, 2);
        assert_eq!(s.collections[1].name, "c1");
        assert_eq!(s.collections[1].documents, 1);
    }

    #[test]
    fn get_index_health_reports_no_docs() {
        let (_t, store) = open_test_store();
        let h = get_index_health(&store.conn, "m").unwrap();
        assert_eq!(h.total_docs, 0);
        assert_eq!(h.needs_embedding, 0);
        assert_eq!(h.days_stale, None);
    }

    #[test]
    fn get_index_health_reports_days_stale_from_modified_at() {
        let (_t, store) = open_test_store();
        // A doc with a known old timestamp.
        insert_doc(&store.conn, "c", "a.md", "h1", "2020-01-01T00:00:00.000Z");
        let h = get_index_health(&store.conn, "m").unwrap();
        assert_eq!(h.total_docs, 1);
        assert_eq!(h.needs_embedding, 1);
        // 2020-01-01 → at least 4 years (1460 days) old by 2026.
        assert!(h.days_stale.unwrap() >= 1_460);
    }

    // --- ported from store.test.ts `describe("Index Status")` ---

    #[test]
    fn get_status_counts_only_active_documents() {
        let (_t, store) = open_test_store();
        insert_doc_active(
            &store.conn,
            "c",
            "a.md",
            "h1",
            "2024-01-01T00:00:00.000Z",
            1,
        );
        insert_doc_active(
            &store.conn,
            "c",
            "b.md",
            "h2",
            "2024-01-01T00:00:00.000Z",
            1,
        );
        insert_doc_active(
            &store.conn,
            "c",
            "c.md",
            "h3",
            "2024-01-01T00:00:00.000Z",
            0,
        ); // inactive

        let s = get_status(&store.conn, "m").unwrap();
        assert_eq!(s.total_documents, 2); // only active
    }

    #[test]
    fn get_status_reports_collection_info() {
        let (_t, store) = open_test_store();
        let coll = Collection {
            path: "/test/path".to_string(),
            pattern: "**/*.md".to_string(),
            ignore: None,
            context: None,
            update: None,
            include_by_default: None,
        };
        upsert_store_collection(&store.conn, "myapp", &coll).unwrap();
        insert_doc(
            &store.conn,
            "myapp",
            "doc1.md",
            "h1",
            "2024-01-01T00:00:00.000Z",
        );

        let s = get_status(&store.conn, "m").unwrap();
        let col = s.collections.iter().find(|c| c.name == "myapp").unwrap();
        assert_eq!(col.path.as_deref(), Some("/test/path"));
        assert_eq!(col.pattern.as_deref(), Some("**/*.md"));
        assert_eq!(col.documents, 1);
    }

    #[test]
    fn get_hashes_needing_embedding_dedups_by_hash() {
        let (_t, store) = open_test_store();
        // doc1 + doc3 share hash h1; doc2 has h2 → 2 distinct hashes.
        insert_doc(
            &store.conn,
            "c",
            "doc1.md",
            "h1",
            "2024-01-01T00:00:00.000Z",
        );
        insert_doc(
            &store.conn,
            "c",
            "doc2.md",
            "h2",
            "2024-01-01T00:00:00.000Z",
        );
        insert_doc(
            &store.conn,
            "c",
            "doc3.md",
            "h1",
            "2024-01-01T00:00:00.000Z",
        );

        assert_eq!(
            get_hashes_needing_embedding(&store.conn, None, "m").unwrap(),
            2
        );
    }

    #[test]
    fn embedding_health_scoped_to_active_model() {
        let (_t, store) = open_test_store();
        let active = "hf:active/embed-model.gguf";
        let stale = "hf:stale/embed-model.gguf";
        insert_doc(
            &store.conn,
            "c",
            "doc1.md",
            "hash1",
            "2024-01-01T00:00:00.000Z",
        );

        ensure_vec_table(&store.conn, 3).unwrap();
        // Embed hash1 only under the *stale* model.
        insert_embedding(
            &store.conn,
            "hash1",
            0,
            0,
            &[1.0, 2.0, 3.0],
            stale,
            "2024-01-01T00:00:00.000Z",
            1,
        )
        .unwrap();

        // Active model still sees hash1 as unembedded.
        assert_eq!(
            get_hashes_needing_embedding(&store.conn, None, active).unwrap(),
            1
        );
        assert_eq!(get_status(&store.conn, active).unwrap().needs_embedding, 1);
        assert_eq!(
            get_index_health(&store.conn, active)
                .unwrap()
                .needs_embedding,
            1
        );
        // Stale model: hash1 is complete.
        assert_eq!(
            get_hashes_needing_embedding(&store.conn, None, stale).unwrap(),
            0
        );
    }
}
