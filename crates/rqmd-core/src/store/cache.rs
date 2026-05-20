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
/// Mirrors `getCacheKey` (`store.ts:2024–2029`) **byte for byte** — `url`
/// and `body` are concatenated with no separator, matching the TS
/// `hash.update(url); hash.update(JSON.stringify(body))` pattern. Callers
/// in `rqmd_core::store_ops` are responsible for serialising their `body`
/// argument identically to `JSON.stringify` (use `serde_json` with the
/// `preserve_order` feature so object keys preserve insertion order).
///
/// Inter-tool compatibility with `qmd`'s on-disk `llm_cache` table depends
/// on the byte equality of this digest. Pinned by the unit tests below and
/// by the `cache_keys` tests under `rqmd_core::store_ops`.
pub fn get_cache_key(url: &str, body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
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

    /// Pins the wire-level cache key bytes to match the TypeScript
    /// `getCacheKey(url, body)` (`tobi/qmd/src/store.ts:2024-2029`). Hashes
    /// were generated externally:
    ///   `sha256("expandQuery" || JSON.stringify({query:"hi",model:"m"}))`
    /// Required for cache round-trips between qmd (TS) and rqmd (Rust) over
    /// the same `llm_cache` table.
    #[test]
    fn cache_keys_match_ts_known_inputs() {
        assert_eq!(
            get_cache_key("expandQuery", r#"{"query":"hi","model":"m"}"#),
            "76c821b4ed2aba79b3bd1757beefaee75160d41b337811f51ce1a6e575a63bb4"
        );
        // Japanese — exercises UTF-8 bytes in the body.
        assert_eq!(
            get_cache_key("expandQuery", r#"{"query":"こんにちは","model":"m"}"#),
            "20f8913e9743bfb22d0f8e6122ff1ba21182bec1a055773137badde29dc902ef"
        );
        // Rerank with intent-prepended query. TS `JSON.stringify` escapes
        // the actual newlines in `"intent\n\nq"` (where `\n` is U+000A) to
        // the two-character sequence `\n` in the JSON output, so the body
        // bytes contain literal backslash-n pairs.
        assert_eq!(
            get_cache_key(
                "rerank",
                "{\"query\":\"intent\\n\\nq\",\"model\":\"m\",\"chunk\":\"text\"}"
            ),
            "bb87120a9fcce50bf338d64b7316648275c162bf8bc3d89756620ef8bb7a3ce5"
        );
    }
}
