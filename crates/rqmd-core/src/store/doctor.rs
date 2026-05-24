//! LLM-free SQL for the `rqmd doctor` diagnostics command.
//!
//! Port of the inline `db.prepare(...)` queries in `tobi/qmd`'s `showDoctor`
//! and `maybeAdoptLegacyEmbeddingFingerprint`. Kept in [`crate::store`] (no LLM
//! dependency, §3-1 boundary): `model` / `fingerprint` arrive as opaque `&str`.
//! Re-chunking / re-embedding and the comparison logic live in the CLI `doctor`
//! command, which combines these with [`crate::store_ops`] / [`crate::llm`].

use rusqlite::{Connection, OptionalExtension, params};

use super::Result;

/// One `(model, fingerprint)` bucket of `content_vectors`, with distinct-doc
/// and chunk counts. Mirrors qmd's `embedding fingerprints` GROUP BY.
#[derive(Debug, Clone, PartialEq)]
pub struct FingerprintGroup {
    pub model: String,
    /// Empty string for legacy (pre-fingerprint) rows.
    pub fingerprint: String,
    pub docs: i64,
    pub chunks: i64,
}

/// A sampled chunk to re-chunk / re-embed (current-fingerprint or legacy).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSample {
    pub hash: String,
    pub seq: i64,
    /// Full document body (for re-chunking).
    pub body: String,
    /// A representative path (for title extraction + AST-aware chunking).
    pub path: String,
}

/// Count of active documents. qmd: `SELECT COUNT(*) FROM documents WHERE active = 1`.
pub fn count_active_documents(conn: &Connection) -> Result<i64> {
    Ok(
        conn.query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
            r.get(0)
        })?,
    )
}

/// All `(model, embed_fingerprint)` buckets, ordered by chunk count desc.
/// Mirrors qmd's `embedding fingerprints` query.
pub fn fingerprint_groups(conn: &Connection) -> Result<Vec<FingerprintGroup>> {
    let mut stmt = conn.prepare(
        "SELECT model, embed_fingerprint AS fingerprint, \
                COUNT(DISTINCT hash) AS docs, COUNT(*) AS chunks \
         FROM content_vectors \
         GROUP BY model, embed_fingerprint \
         ORDER BY chunks DESC, model, embed_fingerprint",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok(FingerprintGroup {
                model: r.get(0)?,
                fingerprint: r.get(1)?,
                docs: r.get(2)?,
                chunks: r.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Sample up to `limit` random chunks embedded under `(model, fingerprint)`.
/// Mirrors qmd `checkEmbeddingVectorSamples`' sampling query.
pub fn sample_current_chunks(
    conn: &Connection,
    model: &str,
    fingerprint: &str,
    limit: i64,
) -> Result<Vec<VectorSample>> {
    let mut stmt = conn.prepare(
        "SELECT cv.hash, cv.seq, c.doc AS body, MIN(d.path) AS path \
         FROM content_vectors cv \
         JOIN documents d ON d.hash = cv.hash AND d.active = 1 \
         JOIN content c ON c.hash = cv.hash \
         WHERE cv.model = ? AND cv.embed_fingerprint = ? \
         GROUP BY cv.hash, cv.seq, c.doc \
         ORDER BY random() \
         LIMIT ?",
    )?;
    let rows = stmt
        .query_map(params![model, fingerprint, limit], |r| {
            Ok(VectorSample {
                hash: r.get(0)?,
                seq: r.get(1)?,
                body: r.get(2)?,
                path: r.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

/// One legacy (empty-fingerprint) chunk for `model`, lowest `(hash, seq)`.
/// Mirrors qmd `maybeAdoptLegacyEmbeddingFingerprint`'s sample query.
pub fn sample_legacy_chunk(conn: &Connection, model: &str) -> Result<Option<VectorSample>> {
    let row = conn
        .query_row(
            "SELECT cv.hash, cv.seq, c.doc AS body, MIN(d.path) AS path \
             FROM content_vectors cv \
             JOIN documents d ON d.hash = cv.hash AND d.active = 1 \
             JOIN content c ON c.hash = cv.hash \
             WHERE cv.model = ? AND cv.embed_fingerprint = '' \
             GROUP BY cv.hash, cv.seq, c.doc \
             ORDER BY cv.hash, cv.seq \
             LIMIT 1",
            params![model],
            |r| {
                Ok(VectorSample {
                    hash: r.get(0)?,
                    seq: r.get(1)?,
                    body: r.get(2)?,
                    path: r.get(3)?,
                })
            },
        )
        .optional()?;
    Ok(row)
}

/// Number of distinct legacy (empty-fingerprint) hashes for `model`.
pub fn count_legacy_distinct_hashes(conn: &Connection, model: &str) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT COUNT(DISTINCT hash) FROM content_vectors WHERE model = ? AND embed_fingerprint = ''",
        params![model],
        |r| r.get(0),
    )?)
}

/// Stamp `fingerprint` onto all legacy (empty-fingerprint) rows for `model`,
/// returning the number of rows updated. qmd `maybeAdopt...`'s `UPDATE`.
pub fn adopt_legacy_fingerprint(
    conn: &Connection,
    model: &str,
    fingerprint: &str,
) -> Result<usize> {
    Ok(conn.execute(
        "UPDATE content_vectors SET embed_fingerprint = ? WHERE model = ? AND embed_fingerprint = ''",
        params![fingerprint, model],
    )?)
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

    fn insert_doc(
        conn: &Connection,
        collection: &str,
        path: &str,
        body: &str,
        hash: &str,
        active: i64,
    ) {
        conn.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, 'ts')",
            params![hash, body],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) \
             VALUES (?, ?, 't', ?, 'ts', 'ts', ?)",
            params![collection, path, hash, active],
        )
        .unwrap();
    }

    fn insert_cv(conn: &Connection, hash: &str, seq: i64, model: &str, fp: &str, total: i64) {
        conn.execute(
            "INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
             VALUES (?, ?, 0, ?, ?, ?, 'ts')",
            params![hash, seq, model, fp, total],
        )
        .unwrap();
    }

    #[test]
    fn count_active_documents_excludes_inactive() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "A", "ha", 1);
        insert_doc(&store.conn, "c", "b.md", "B", "hb", 1);
        insert_doc(&store.conn, "c", "c.md", "C", "hc", 0);
        assert_eq!(count_active_documents(&store.conn).unwrap(), 2);
    }

    #[test]
    fn fingerprint_groups_counts_and_orders_by_chunks() {
        let (_t, store) = open_test_store();
        // model m, fpA: 1 doc / 2 chunks; model m, fpB: 1 doc / 1 chunk.
        insert_cv(&store.conn, "h1", 0, "m", "fpA", 2);
        insert_cv(&store.conn, "h1", 1, "m", "fpA", 2);
        insert_cv(&store.conn, "h2", 0, "m", "fpB", 1);

        let groups = fingerprint_groups(&store.conn).unwrap();
        assert_eq!(groups.len(), 2);
        // Ordered by chunks DESC → fpA (2 chunks) first.
        assert_eq!(groups[0].fingerprint, "fpA");
        assert_eq!(groups[0].docs, 1);
        assert_eq!(groups[0].chunks, 2);
        assert_eq!(groups[1].fingerprint, "fpB");
        assert_eq!(groups[1].chunks, 1);
    }

    #[test]
    fn sample_current_chunks_joins_active_docs() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "BODY-A", "ha", 1);
        insert_cv(&store.conn, "ha", 0, "m", "fp", 1);
        // Wrong fingerprint → not sampled.
        insert_doc(&store.conn, "c", "b.md", "BODY-B", "hb", 1);
        insert_cv(&store.conn, "hb", 0, "m", "other", 1);

        let samples = sample_current_chunks(&store.conn, "m", "fp", 3).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].hash, "ha");
        assert_eq!(samples[0].seq, 0);
        assert_eq!(samples[0].body, "BODY-A");
        assert_eq!(samples[0].path, "a.md");
    }

    #[test]
    fn legacy_helpers_count_sample_and_adopt() {
        let (_t, store) = open_test_store();
        insert_doc(&store.conn, "c", "a.md", "BODY-A", "ha", 1);
        insert_doc(&store.conn, "c", "b.md", "BODY-B", "hb", 1);
        insert_cv(&store.conn, "ha", 0, "m", "", 1); // legacy
        insert_cv(&store.conn, "hb", 0, "m", "", 1); // legacy

        assert_eq!(count_legacy_distinct_hashes(&store.conn, "m").unwrap(), 2);

        let sample = sample_legacy_chunk(&store.conn, "m")
            .unwrap()
            .expect("a sample");
        // Lowest (hash, seq): "ha" < "hb".
        assert_eq!(sample.hash, "ha");
        assert_eq!(sample.seq, 0);

        let adopted = adopt_legacy_fingerprint(&store.conn, "m", "fpNEW").unwrap();
        assert_eq!(adopted, 2);
        // No legacy rows remain; the group is now under the new fingerprint.
        assert_eq!(count_legacy_distinct_hashes(&store.conn, "m").unwrap(), 0);
        let groups = fingerprint_groups(&store.conn).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].fingerprint, "fpNEW");
        assert_eq!(groups[0].docs, 2);
    }
}
