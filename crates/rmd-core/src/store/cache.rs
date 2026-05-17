//! `llm_cache` table CRUD.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 2024–2059. No LLM caller
//! exists in this pass; the module is included so the `llm_cache` table
//! has a typed surface ready for the LLM port. The unit test below also
//! exercises the table to prove [`super::schema::initialize`] created it.

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use super::path::now_rfc3339;
use super::Result;

/// Compute the cache key for an `(url, body)` pair. Hex-encoded SHA-256.
/// Mirrors `getCacheKey` (`store.ts:2024–…`).
pub fn get_cache_key(url: &str, body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hasher.update(b"\0");
    hasher.update(body.as_bytes());
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

pub fn get_cached_result(conn: &Connection, key: &str) -> Result<Option<String>> {
    let row: Option<String> = conn
        .query_row(
            "SELECT result FROM llm_cache WHERE hash = ?",
            params![key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(row)
}

pub fn set_cached_result(conn: &Connection, key: &str, result: &str) -> Result<()> {
    let now = now_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)",
        params![key, result, now],
    )?;
    Ok(())
}

pub fn clear_cache(conn: &Connection) -> Result<usize> {
    let n = conn.execute("DELETE FROM llm_cache", [])?;
    Ok(n)
}

/// Same as [`clear_cache`] but named to match TS `deleteLLMCache`
/// (`store.ts:2056–2059`). Kept distinct in case the LLM pass wants to
/// scope deletion further.
pub fn delete_llm_cache(conn: &Connection) -> Result<usize> {
    clear_cache(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use tempfile::NamedTempFile;

    #[test]
    fn cache_round_trip_against_real_store() {
        let tmp = NamedTempFile::new().unwrap();
        let mut store = Store::open(tmp.path()).expect("open store");
        let conn = &mut store.conn;

        let key = get_cache_key("https://api.example/v1/chat", r#"{"q":"hi"}"#);
        assert_eq!(key.len(), 64);

        assert!(get_cached_result(conn, &key).unwrap().is_none());
        set_cached_result(conn, &key, "stored value").unwrap();
        assert_eq!(
            get_cached_result(conn, &key).unwrap().as_deref(),
            Some("stored value")
        );

        assert_eq!(clear_cache(conn).unwrap(), 1);
        assert!(get_cached_result(conn, &key).unwrap().is_none());

        // delete_llm_cache returns 0 when nothing is left.
        assert_eq!(delete_llm_cache(conn).unwrap(), 0);
    }

    #[test]
    fn cache_keys_are_stable() {
        let a = get_cache_key("u", "b");
        let b = get_cache_key("u", "b");
        assert_eq!(a, b);
        assert_ne!(get_cache_key("u", "b"), get_cache_key("u2", "b"));
        assert_ne!(get_cache_key("u", "b"), get_cache_key("u", "b2"));
    }
}
