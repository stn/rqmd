//! Cleanup, deactivation, and `VACUUM`.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 2061–2143. The orphan-vector
//! cleanup is included even though embedding-side ops are deferred: it is
//! pure SQL and is harmless on an empty `content_vectors` table.

use rusqlite::Connection;

use super::Result;

/// Delete documents marked inactive (`active = 0`).
pub fn delete_inactive_documents(conn: &Connection) -> Result<usize> {
    Ok(conn.execute("DELETE FROM documents WHERE active = 0", [])?)
}

/// Delete content rows no longer referenced by any document (including
/// inactive ones — those are tombstones until [`delete_inactive_documents`]
/// runs).
pub fn cleanup_orphaned_content(conn: &Connection) -> Result<usize> {
    Ok(conn.execute(
        "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)",
        [],
    )?)
}

/// Delete embedding rows whose document is no longer active.
///
/// If `sqlite-vec` is not loaded, the `vectors_vec` virtual table cannot be
/// queried and this becomes a no-op (mirrors TS `_sqliteVecAvailable`
/// fallback).
pub fn cleanup_orphaned_vectors(conn: &Connection, vec_available: bool) -> Result<usize> {
    if !vec_available {
        return Ok(0);
    }
    if conn
        .query_row("SELECT 1 FROM vectors_vec LIMIT 0", [], |_| Ok(()))
        .is_err()
    {
        return Ok(0);
    }

    let count: i64 = conn.query_row(
        r#"SELECT COUNT(*) FROM content_vectors cv
           WHERE NOT EXISTS (
               SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
           )"#,
        [],
        |row| row.get(0),
    )?;
    if count == 0 {
        return Ok(0);
    }

    conn.execute(
        r#"DELETE FROM vectors_vec WHERE hash_seq IN (
               SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
               WHERE NOT EXISTS (
                   SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
               )
           )"#,
        [],
    )?;
    conn.execute(
        r#"DELETE FROM content_vectors
           WHERE hash NOT IN (SELECT hash FROM documents WHERE active = 1)"#,
        [],
    )?;

    Ok(count as usize)
}

/// Run SQLite `VACUUM` to reclaim disk space.
pub fn vacuum_database(conn: &Connection) -> Result<()> {
    conn.execute_batch("VACUUM")?;
    Ok(())
}
