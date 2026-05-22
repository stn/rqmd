//! Embedding-side SQL: `content_vectors` + `vectors_vec` CRUD, dimension
//! probing, and the LLM-free half of `searchVec`.
//!
//! Port of the pure-SQL portion of `tobi/qmd`'s `src/store.ts`:
//! `ensureVecTableInternal` (1141–1163), `getPendingEmbeddingDocs` (1436–1456),
//! `getEmbeddingDocsForBatch` (1489–1504), `getHashesNeedingEmbedding`
//! (1980–1998), `searchVec` (3252–3336, SQL half only), `getHashesForEmbedding`
//! (3355–3371), `clearAllEmbeddings` (3389–3436), `insertEmbedding` (3448–3469),
//! `removeIncompleteEmbeddings` (3471–3489), `contentVectorExpectedChunksExpr`
//! (1431–1434).
//!
//! No LLM dependency — `model` is always an opaque `&str` resolved upstream.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use super::context::get_context_for_file;
use super::docid::get_docid;
use super::search::{DocumentResult, SearchResult, SearchSource};
use super::{Error, Result};

// ============================================================================
// Public types
// ============================================================================

/// Distinct content hash that still needs embedding for `model`, plus a
/// representative body and a sample path for display. Mirrors the return shape
/// of `getHashesForEmbedding` (TS 3355).
#[derive(Debug, Clone, PartialEq)]
pub struct HashForEmbedding {
    pub hash: String,
    pub body: String,
    pub path: String,
}

/// Lightweight pending-doc descriptor used by `generate_embeddings` batching.
/// Mirrors TS `PendingEmbeddingDoc` (1396).
#[derive(Debug, Clone, PartialEq)]
pub struct PendingEmbeddingDoc {
    pub hash: String,
    pub path: String,
    pub bytes: usize,
}

/// `PendingEmbeddingDoc` enriched with the full body, returned by
/// `get_embedding_docs_for_batch`. Mirrors TS `EmbeddingDoc` (1402).
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingDoc {
    pub hash: String,
    pub path: String,
    pub bytes: usize,
    pub body: String,
}

// ============================================================================
// Internal helpers
// ============================================================================

static VEC_DIM_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"float\[(\d+)\]").expect("valid regex"));

/// Returns `"MAX(total_chunks)"` if `content_vectors.total_chunks` exists,
/// otherwise `"1"`. Mirrors TS `contentVectorExpectedChunksExpr` (1431).
///
/// The schema we initialise always includes the column, but the helper exists
/// so an upgraded DB (older schemas) still queries correctly.
pub(crate) fn content_vector_expected_chunks_expr(conn: &Connection) -> Result<&'static str> {
    let mut stmt = conn.prepare("PRAGMA table_info(content_vectors)")?;
    let cols: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(if cols.iter().any(|c| c == "total_chunks") {
        "MAX(total_chunks)"
    } else {
        "1"
    })
}

fn vec_table_exists(conn: &Connection) -> Result<bool> {
    let row: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    Ok(row.is_some())
}

// ============================================================================
// Vector table provisioning
// ============================================================================

/// Create the `vectors_vec` virtual table for `dimensions`-d cosine vectors,
/// or validate that an existing table already matches. Mirrors TS
/// `ensureVecTableInternal` (1141).
///
/// Errors:
/// * `Error::InvalidQuery` — existing table has a different dimension count.
/// * `Error::Sqlite` — sqlite-vec extension unavailable (the `CREATE VIRTUAL
///   TABLE` call surfaces it). Callers should check `Store::vec_available`
///   first if they want a friendlier message.
pub fn ensure_vec_table(conn: &Connection, dimensions: usize) -> Result<()> {
    let existing: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )
        .optional()?;

    if let Some(sql) = existing {
        let dim_from_ddl = VEC_DIM_REGEX
            .captures(&sql)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse::<usize>().ok());
        let has_hash_seq = sql.contains("hash_seq");
        let has_cosine = sql.contains("distance_metric=cosine");

        if dim_from_ddl == Some(dimensions) && has_hash_seq && has_cosine {
            return Ok(());
        }
        if let Some(existing_dim) = dim_from_ddl
            && existing_dim != dimensions
        {
            return Err(Error::InvalidQuery(format!(
                "Embedding dimension mismatch: existing vectors are {existing_dim}d \
                 but the current model produces {dimensions}d. Run 'rqmd embed -f' \
                 to re-embed with the new model."
            )));
        }
        conn.execute("DROP TABLE IF EXISTS vectors_vec", [])?;
    }

    let ddl = format!(
        "CREATE VIRTUAL TABLE vectors_vec USING vec0(\
            hash_seq TEXT PRIMARY KEY, \
            embedding float[{dimensions}] distance_metric=cosine\
        )"
    );
    conn.execute(&ddl, [])?;
    Ok(())
}

// ============================================================================
// Hash listing / counts
// ============================================================================

/// Return all unique content hashes that still need embedding for `model`
/// (active documents whose embedding row is missing or partial). Each row
/// carries a representative body and a sample path. Mirrors TS
/// `getHashesForEmbedding` (3355).
pub fn get_hashes_for_embedding(conn: &Connection, model: &str) -> Result<Vec<HashForEmbedding>> {
    let expected_expr = content_vector_expected_chunks_expr(conn)?;
    let sql = format!(
        "SELECT d.hash, c.doc AS body, MIN(d.path) AS path
         FROM documents d
         JOIN content c ON d.hash = c.hash
         LEFT JOIN (
             SELECT hash, model, COUNT(*) AS chunk_count, {expected_expr} AS expected_chunks
             FROM content_vectors
             WHERE model = ?
             GROUP BY hash, model
         ) v ON d.hash = v.hash
         WHERE d.active = 1
           AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks)
         GROUP BY d.hash"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params![model], |row| {
            Ok(HashForEmbedding {
                hash: row.get(0)?,
                body: row.get(1)?,
                path: row.get(2)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Count distinct hashes that still need embedding for `model`, optionally
/// scoped to one collection. Mirrors TS `getHashesNeedingEmbedding` (1980).
pub fn get_hashes_needing_embedding(
    conn: &Connection,
    collection: Option<&str>,
    model: &str,
) -> Result<i64> {
    let expected_expr = content_vector_expected_chunks_expr(conn)?;
    let collection_filter = if collection.is_some() {
        "AND d.collection = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT COUNT(DISTINCT d.hash) AS count
         FROM documents d
         LEFT JOIN (
             SELECT hash, model, COUNT(*) AS chunk_count, {expected_expr} AS expected_chunks
             FROM content_vectors
             WHERE model = ?
             GROUP BY hash, model
         ) v ON d.hash = v.hash
         WHERE d.active = 1
           AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks)
           {collection_filter}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let count: i64 = if let Some(c) = collection {
        stmt.query_row(params![model, c], |r| r.get(0))?
    } else {
        stmt.query_row(params![model], |r| r.get(0))?
    };
    Ok(count)
}

/// Pending-doc list for batched embedding: sorted by `MIN(path)`, including
/// byte length. Mirrors TS `getPendingEmbeddingDocs` (1436).
pub fn get_pending_embedding_docs(
    conn: &Connection,
    collection: Option<&str>,
    model: &str,
) -> Result<Vec<PendingEmbeddingDoc>> {
    let expected_expr = content_vector_expected_chunks_expr(conn)?;
    let collection_filter = if collection.is_some() {
        "AND d.collection = ?"
    } else {
        ""
    };
    let sql = format!(
        "SELECT d.hash, MIN(d.path) AS path, length(CAST(c.doc AS BLOB)) AS bytes
         FROM documents d
         JOIN content c ON d.hash = c.hash
         LEFT JOIN (
             SELECT hash, model, COUNT(*) AS chunk_count, {expected_expr} AS expected_chunks
             FROM content_vectors
             WHERE model = ?
             GROUP BY hash, model
         ) v ON d.hash = v.hash
         WHERE d.active = 1
           AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks)
           {collection_filter}
         GROUP BY d.hash
         ORDER BY MIN(d.path)"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = if let Some(c) = collection {
        stmt.query_map(params![model, c], |row| {
            Ok(PendingEmbeddingDoc {
                hash: row.get(0)?,
                path: row.get(1)?,
                bytes: row.get::<_, i64>(2)?.max(0) as usize,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        stmt.query_map(params![model], |row| {
            Ok(PendingEmbeddingDoc {
                hash: row.get(0)?,
                path: row.get(1)?,
                bytes: row.get::<_, i64>(2)?.max(0) as usize,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };
    Ok(rows)
}

/// Hydrate a batch of pending docs with their bodies in a single `IN (?)`
/// query. Mirrors TS `getEmbeddingDocsForBatch` (1489). Missing rows produce
/// an empty body (lost-row tolerance — `generate_embeddings` skips those).
pub fn get_embedding_docs_for_batch(
    conn: &Connection,
    batch: &[PendingEmbeddingDoc],
) -> Result<Vec<EmbeddingDoc>> {
    if batch.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = std::iter::repeat_n("?", batch.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT hash, doc AS body FROM content WHERE hash IN ({placeholders})");
    let mut stmt = conn.prepare(&sql)?;
    let body_by_hash: HashMap<String, String> = stmt
        .query_map(
            params_from_iter(batch.iter().map(|d| d.hash.as_str())),
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )?
        .filter_map(|r| r.ok())
        .collect();

    Ok(batch
        .iter()
        .map(|d| EmbeddingDoc {
            hash: d.hash.clone(),
            path: d.path.clone(),
            bytes: d.bytes,
            body: body_by_hash.get(&d.hash).cloned().unwrap_or_default(),
        })
        .collect())
}

// ============================================================================
// Mutation
// ============================================================================

/// Insert a single embedding into `content_vectors` and `vectors_vec`.
/// Mirrors TS `insertEmbedding` (3448).
///
/// Ordering is load-bearing: `content_vectors` is written first so a crash
/// between the two statements does NOT leave a `vectors_vec` row that
/// `get_hashes_for_embedding` would re-select (and double-embed). vec0
/// virtual tables silently ignore `INSERT OR REPLACE`, hence the explicit
/// DELETE + INSERT pair for `vectors_vec`.
#[allow(clippy::too_many_arguments)]
pub fn insert_embedding(
    conn: &Connection,
    hash: &str,
    seq: i64,
    pos: i64,
    embedding: &[f32],
    model: &str,
    embedded_at: &str,
    total_chunks: i64,
) -> Result<()> {
    let hash_seq = format!("{hash}_{seq}");

    conn.execute(
        "INSERT OR REPLACE INTO content_vectors \
         (hash, seq, pos, model, total_chunks, embedded_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
        params![hash, seq, pos, model, total_chunks, embedded_at],
    )?;

    conn.execute(
        "DELETE FROM vectors_vec WHERE hash_seq = ?",
        params![hash_seq],
    )?;
    let blob: &[u8] = bytemuck::cast_slice(embedding);
    conn.execute(
        "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
        params![hash_seq, blob],
    )?;
    Ok(())
}

/// Clear embeddings globally (`collection == None`) or for one collection.
/// Mirrors TS `clearAllEmbeddings` (3389).
///
/// Global: drop `content_vectors` rows then drop the `vectors_vec` virtual
/// table (it will be recreated by `ensure_vec_table` on the next embed run).
///
/// Per-collection: only remove embeddings for hashes referenced *exclusively*
/// by active documents in that collection (shared hashes stay so other
/// collections keep working). `vectors_vec` is preserved unless
/// `content_vectors` becomes empty, in which case it is dropped to allow
/// dimension changes on the next embed run.
///
/// Not wrapped in an explicit transaction — TS does not either, and
/// `vectors_vec` (vec0) behaves poorly inside transactions with rollback.
/// A crash mid-clear can leave the two tables temporarily inconsistent;
/// `remove_incomplete_embeddings` and re-running this function recover.
pub fn clear_all_embeddings(conn: &Connection, collection: Option<&str>) -> Result<()> {
    let Some(collection) = collection else {
        conn.execute("DELETE FROM content_vectors", [])?;
        conn.execute("DROP TABLE IF EXISTS vectors_vec", [])?;
        return Ok(());
    };

    const EXCLUSIVE_HASHES_SUBQUERY: &str = "\
        SELECT DISTINCT d.hash \
        FROM documents d \
        WHERE d.collection = ? AND d.active = 1 \
          AND NOT EXISTS ( \
            SELECT 1 FROM documents d2 \
            WHERE d2.hash = d.hash AND d2.active = 1 AND d2.collection != d.collection \
          )";

    if vec_table_exists(conn)? {
        let select_sql = format!(
            "SELECT cv.hash, cv.seq FROM content_vectors cv WHERE cv.hash IN ({EXCLUSIVE_HASHES_SUBQUERY})"
        );
        let mut select_stmt = conn.prepare(&select_sql)?;
        let rows: Vec<(String, i64)> = select_stmt
            .query_map(params![collection], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        drop(select_stmt);

        let mut del_stmt = conn.prepare("DELETE FROM vectors_vec WHERE hash_seq = ?")?;
        for (hash, seq) in &rows {
            del_stmt.execute(params![format!("{hash}_{seq}")])?;
        }
    }

    let delete_sql =
        format!("DELETE FROM content_vectors WHERE hash IN ({EXCLUSIVE_HASHES_SUBQUERY})");
    conn.execute(&delete_sql, params![collection])?;

    let remaining: i64 =
        conn.query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?;
    if remaining == 0 {
        conn.execute("DROP TABLE IF EXISTS vectors_vec", [])?;
    }
    Ok(())
}

/// Remove embeddings for any hash in `expected_chunks_by_hash` whose actual
/// chunk count in `content_vectors` differs from the expected count (i.e. a
/// previous embedding run was interrupted partway). Returns the number of
/// chunk rows removed. Mirrors TS `removeIncompleteEmbeddings` (3471).
pub fn remove_incomplete_embeddings(
    conn: &Connection,
    expected_chunks_by_hash: &HashMap<String, i64>,
    model: &str,
) -> Result<i64> {
    let mut removed = 0i64;
    let mut rows_stmt =
        conn.prepare("SELECT seq FROM content_vectors WHERE hash = ? AND model = ?")?;
    let mut delete_content_stmt =
        conn.prepare("DELETE FROM content_vectors WHERE hash = ? AND model = ?")?;
    let mut delete_vec_stmt = conn.prepare("DELETE FROM vectors_vec WHERE hash_seq = ?")?;

    for (hash, expected) in expected_chunks_by_hash {
        let seqs: Vec<i64> = rows_stmt
            .query_map(params![hash, model], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .collect();
        if seqs.is_empty() || seqs.len() as i64 == *expected {
            continue;
        }
        for seq in &seqs {
            // Tolerate missing vec table (e.g. cleared mid-run).
            let _ = delete_vec_stmt.execute(params![format!("{hash}_{seq}")]);
        }
        delete_content_stmt.execute(params![hash, model])?;
        removed += seqs.len() as i64;
    }
    Ok(removed)
}

// ============================================================================
// Pure-SQL vector search
// ============================================================================

/// kNN over `vectors_vec` for `embedding`, joined with document metadata.
/// Mirrors the SQL half of TS `searchVec` (3252) — embedding the query is
/// the orchestration layer's job.
///
/// Returns up to `limit` results, deduped by filepath (keeping the closest
/// chunk per file), sorted ascending by distance (descending by cosine
/// similarity score). `k` for the vec0 kNN is `(limit * 3).min(1000)` —
/// capped so a large `--limit` does not blow past SQLite's variable limit
/// in the follow-up `IN (?)` query.
///
/// Returns an empty vector if `vectors_vec` does not exist yet.
pub fn search_vec_with_embedding(
    conn: &Connection,
    embedding: &[f32],
    limit: usize,
    collection: Option<&str>,
) -> Result<Vec<SearchResult>> {
    if !vec_table_exists(conn)? {
        return Ok(Vec::new());
    }

    // Step 1: kNN on vec0 alone (NO JOINs — vec0 hangs otherwise; TS 3259-3262).
    let k = limit.saturating_mul(3).clamp(1, 1000) as i64;
    let blob: &[u8] = bytemuck::cast_slice(embedding);
    let mut step1 = conn
        .prepare("SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?")?;
    let vec_rows: Vec<(String, f64)> = step1
        .query_map(params![blob, k], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();
    if vec_rows.is_empty() {
        return Ok(Vec::new());
    }
    let distance_by_hash_seq: HashMap<String, f64> = vec_rows.iter().cloned().collect();
    let hash_seqs: Vec<String> = vec_rows.into_iter().map(|(hs, _)| hs).collect();

    // Step 2: join hashes back with content_vectors + documents + content.
    let placeholders = std::iter::repeat_n("?", hash_seqs.len())
        .collect::<Vec<_>>()
        .join(",");
    let mut sql = format!(
        "SELECT
            cv.hash || '_' || cv.seq AS hash_seq,
            cv.hash,
            cv.pos,
            'qmd://' || d.collection || '/' || d.path AS filepath,
            d.collection || '/' || d.path AS display_path,
            d.title,
            d.collection,
            d.modified_at,
            content.doc AS body
         FROM content_vectors cv
         JOIN documents d ON d.hash = cv.hash AND d.active = 1
         JOIN content ON content.hash = d.hash
         WHERE cv.hash || '_' || cv.seq IN ({placeholders})"
    );
    if collection.is_some() {
        sql.push_str(" AND d.collection = ?");
    }

    let mut step2 = conn.prepare(&sql)?;
    let mut bind_values: Vec<String> = hash_seqs.clone();
    if let Some(c) = collection {
        bind_values.push(c.to_string());
    }

    struct VecRow {
        hash_seq: String,
        hash: String,
        pos: i64,
        filepath: String,
        display_path: String,
        title: String,
        collection_name: String,
        modified_at: String,
        body: String,
    }

    let doc_rows: Vec<VecRow> = step2
        .query_map(params_from_iter(bind_values.iter()), |row| {
            Ok(VecRow {
                hash_seq: row.get(0)?,
                hash: row.get(1)?,
                pos: row.get(2)?,
                filepath: row.get(3)?,
                display_path: row.get(4)?,
                title: row.get(5)?,
                collection_name: row.get(6)?,
                modified_at: row.get(7)?,
                body: row.get(8)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
    drop(step2);

    // Dedupe by filepath, keep the chunk with the smallest distance.
    struct Best {
        row: VecRow,
        distance: f64,
    }
    let mut seen: HashMap<String, Best> = HashMap::new();
    for row in doc_rows {
        let distance = distance_by_hash_seq
            .get(&row.hash_seq)
            .copied()
            .unwrap_or(1.0);
        let key = row.filepath.clone();
        match seen.get_mut(&key) {
            Some(existing) if existing.distance <= distance => {}
            _ => {
                seen.insert(key, Best { row, distance });
            }
        }
    }

    let mut ordered: Vec<Best> = seen.into_values().collect();
    ordered.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ordered.truncate(limit);

    let results = ordered
        .into_iter()
        .map(|b| {
            let body = b.row.body;
            let body_length = body.len();
            let context = get_context_for_file(conn, &b.row.filepath).ok().flatten();
            let docid = get_docid(&b.row.hash);
            SearchResult {
                doc: DocumentResult {
                    filepath: b.row.filepath,
                    display_path: b.row.display_path,
                    title: b.row.title,
                    context,
                    hash: b.row.hash,
                    docid,
                    collection_name: b.row.collection_name,
                    modified_at: b.row.modified_at,
                    body_length,
                    body: Some(body),
                },
                score: 1.0 - b.distance,
                source: SearchSource::Vec,
                chunk_pos: Some(b.row.pos),
            }
        })
        .collect();
    Ok(results)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use tempfile::NamedTempFile;

    fn open_test_store() -> (NamedTempFile, Store) {
        let tmp = NamedTempFile::new().unwrap();
        let store = Store::open(tmp.path()).expect("open store");
        (tmp, store)
    }

    fn insert_doc(conn: &Connection, collection: &str, path: &str, body: &str, hash: &str) {
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?, ?, '2024-01-01T00:00:00.000Z')",
            params![hash, body],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
             VALUES (?, ?, ?, ?, '2024-01-01T00:00:00.000Z', '2024-01-01T00:00:00.000Z', 1)",
            params![collection, path, "title", hash],
        )
        .unwrap();
    }

    #[test]
    fn ensure_vec_table_creates_when_absent() {
        let (_t, store) = open_test_store();
        assert!(!vec_table_exists(&store.conn).unwrap());
        ensure_vec_table(&store.conn, 4).unwrap();
        assert!(vec_table_exists(&store.conn).unwrap());
    }

    #[test]
    fn ensure_vec_table_noop_when_matching() {
        let (_t, store) = open_test_store();
        ensure_vec_table(&store.conn, 4).unwrap();
        // Calling again with same dim should not error and should not drop+recreate.
        ensure_vec_table(&store.conn, 4).unwrap();
    }

    #[test]
    fn ensure_vec_table_errors_on_dim_mismatch() {
        let (_t, store) = open_test_store();
        ensure_vec_table(&store.conn, 4).unwrap();
        let err = ensure_vec_table(&store.conn, 8).unwrap_err();
        match err {
            Error::InvalidQuery(msg) => {
                assert!(msg.contains("mismatch"), "{msg}");
                assert!(msg.contains("4d"), "{msg}");
                assert!(msg.contains("8d"), "{msg}");
            }
            other => panic!("expected InvalidQuery, got {other:?}"),
        }
    }

    #[test]
    fn insert_embedding_round_trip_and_replace() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "body", "hash1");
        ensure_vec_table(&store.conn, 4).unwrap();

        insert_embedding(
            &store.conn,
            "hash1",
            0,
            0,
            &[1.0, 0.0, 0.0, 0.0],
            "m",
            "2024-01-01T00:00:00.000Z",
            1,
        )
        .unwrap();

        let cv_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cv_count, 1);

        // Re-inserting the same (hash, seq) replaces both rows.
        insert_embedding(
            &store.conn,
            "hash1",
            0,
            10,
            &[0.0, 1.0, 0.0, 0.0],
            "m",
            "2024-02-01T00:00:00.000Z",
            1,
        )
        .unwrap();
        let cv_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cv_count, 1);
        let vv_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM vectors_vec", [], |r| r.get(0))
            .unwrap();
        assert_eq!(vv_count, 1);
    }

    #[test]
    fn get_hashes_needing_embedding_distinguishes_full_partial_complete() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "body-a", "hash-a"); // none embedded
        insert_doc(&store.conn, "c", "b.md", "body-b", "hash-b"); // partial
        insert_doc(&store.conn, "c", "c.md", "body-c", "hash-c"); // complete
        ensure_vec_table(&store.conn, 4).unwrap();

        // hash-b expects 2 chunks but only 1 is embedded.
        insert_embedding(&store.conn, "hash-b", 0, 0, &[1.0; 4], "m", "ts", 2).unwrap();
        // hash-c is complete: 1 expected, 1 embedded.
        insert_embedding(&store.conn, "hash-c", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();

        let count = get_hashes_needing_embedding(&store.conn, None, "m").unwrap();
        assert_eq!(count, 2);
        let count_scoped = get_hashes_needing_embedding(&store.conn, Some("c"), "m").unwrap();
        assert_eq!(count_scoped, 2);

        let hashes = get_hashes_for_embedding(&store.conn, "m").unwrap();
        let mut got: Vec<&str> = hashes.iter().map(|h| h.hash.as_str()).collect();
        got.sort();
        assert_eq!(got, vec!["hash-a", "hash-b"]);
    }

    #[test]
    fn get_pending_embedding_docs_sorts_by_path() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "z.md", "ZZZ", "hz");
        insert_doc(&store.conn, "c", "a.md", "AAA", "ha");
        let docs = get_pending_embedding_docs(&store.conn, None, "m").unwrap();
        assert_eq!(
            docs.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
            vec!["a.md", "z.md"]
        );
        assert!(docs[0].bytes > 0);
    }

    #[test]
    fn get_embedding_docs_for_batch_hydrates_bodies() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "BODY-A", "ha");
        insert_doc(&store.conn, "c", "b.md", "BODY-B", "hb");
        let batch = vec![
            PendingEmbeddingDoc {
                hash: "ha".into(),
                path: "a.md".into(),
                bytes: 6,
            },
            PendingEmbeddingDoc {
                hash: "hb".into(),
                path: "b.md".into(),
                bytes: 6,
            },
        ];
        let docs = get_embedding_docs_for_batch(&store.conn, &batch).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].body, "BODY-A");
        assert_eq!(docs[1].body, "BODY-B");
    }

    #[test]
    fn clear_all_embeddings_global_drops_vec_table() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "body", "h1");
        ensure_vec_table(&store.conn, 4).unwrap();
        insert_embedding(&store.conn, "h1", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();

        clear_all_embeddings(&store.conn, None).unwrap();
        let cv_count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cv_count, 0);
        assert!(!vec_table_exists(&store.conn).unwrap());
    }

    #[test]
    fn clear_all_embeddings_scoped_preserves_shared_hashes() {
        let (_t, store) = open_test_store();
        // Same hash referenced from two collections.
        insert_doc(&store.conn, "c1", "a.md", "body", "shared");
        store
            .conn
            .execute(
                "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
                 VALUES ('c2', 'a.md', 'title', 'shared', 'ts', 'ts', 1)",
                [],
            )
            .unwrap();
        // Hash exclusive to c1.
        insert_doc(&store.conn, "c1", "b.md", "body-b", "only_c1");
        ensure_vec_table(&store.conn, 4).unwrap();
        insert_embedding(&store.conn, "shared", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();
        insert_embedding(&store.conn, "only_c1", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();

        clear_all_embeddings(&store.conn, Some("c1")).unwrap();

        let remaining: Vec<String> = store
            .conn
            .prepare("SELECT hash FROM content_vectors ORDER BY hash")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(remaining, vec!["shared".to_string()]);
        // vec table preserved because content_vectors is not empty.
        assert!(vec_table_exists(&store.conn).unwrap());
    }

    #[test]
    fn clear_all_embeddings_scoped_drops_vec_table_when_empty() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c1", "a.md", "body", "only_c1");
        ensure_vec_table(&store.conn, 4).unwrap();
        insert_embedding(&store.conn, "only_c1", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();

        clear_all_embeddings(&store.conn, Some("c1")).unwrap();
        assert!(!vec_table_exists(&store.conn).unwrap());
    }

    #[test]
    fn remove_incomplete_embeddings_removes_partial_only() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "body", "h_partial");
        insert_doc(&store.conn, "c", "b.md", "body-b", "h_full");
        ensure_vec_table(&store.conn, 4).unwrap();
        insert_embedding(&store.conn, "h_partial", 0, 0, &[1.0; 4], "m", "ts", 2).unwrap();
        insert_embedding(&store.conn, "h_full", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();

        let mut expected = HashMap::new();
        expected.insert("h_partial".to_string(), 2);
        expected.insert("h_full".to_string(), 1);
        let removed = remove_incomplete_embeddings(&store.conn, &expected, "m").unwrap();
        assert_eq!(removed, 1);

        let remaining: Vec<String> = store
            .conn
            .prepare("SELECT hash FROM content_vectors ORDER BY hash")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(remaining, vec!["h_full".to_string()]);
    }

    #[test]
    fn search_vec_returns_empty_without_table() {
        let (_t, store) = open_test_store();
        let r = search_vec_with_embedding(&store.conn, &[1.0; 4], 5, None).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn search_vec_finds_closest_and_dedupes_by_filepath() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "body-a", "ha");
        insert_doc(&store.conn, "c", "b.md", "body-b", "hb");
        ensure_vec_table(&store.conn, 4).unwrap();

        // Two chunks for doc a — one very close to the query, one farther.
        insert_embedding(&store.conn, "ha", 0, 0, &[1.0, 0.0, 0.0, 0.0], "m", "ts", 2).unwrap();
        insert_embedding(
            &store.conn,
            "ha",
            1,
            10,
            &[0.0, 1.0, 0.0, 0.0],
            "m",
            "ts",
            2,
        )
        .unwrap();
        // Doc b — moderate distance.
        insert_embedding(&store.conn, "hb", 0, 0, &[0.5, 0.5, 0.0, 0.0], "m", "ts", 1).unwrap();

        let results =
            search_vec_with_embedding(&store.conn, &[1.0, 0.0, 0.0, 0.0], 5, None).unwrap();
        assert_eq!(results.len(), 2);
        // Doc a should appear once (chunk 0, closest), and rank first.
        assert!(results[0].doc.filepath.ends_with("/a.md"));
        assert_eq!(results[0].chunk_pos, Some(0));
        assert!(matches!(results[0].source, SearchSource::Vec));
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn search_vec_respects_collection_filter() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c1", "a.md", "body-a", "ha");
        insert_doc(&store.conn, "c2", "b.md", "body-b", "hb");
        ensure_vec_table(&store.conn, 4).unwrap();
        insert_embedding(&store.conn, "ha", 0, 0, &[1.0, 0.0, 0.0, 0.0], "m", "ts", 1).unwrap();
        insert_embedding(&store.conn, "hb", 0, 0, &[1.0, 0.0, 0.0, 0.0], "m", "ts", 1).unwrap();

        let r1 =
            search_vec_with_embedding(&store.conn, &[1.0, 0.0, 0.0, 0.0], 5, Some("c1")).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].doc.collection_name, "c1");

        let r2 =
            search_vec_with_embedding(&store.conn, &[1.0, 0.0, 0.0, 0.0], 5, Some("c2")).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].doc.collection_name, "c2");
    }
}
