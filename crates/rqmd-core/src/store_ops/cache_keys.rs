//! Cache-key helpers that round-trip with qmd's `JSON.stringify`.
//!
//! Every key feeds [`crate::store::cache::get_cache_key`] (TS-byte-compatible
//! since the NUL separator was removed). The `body` arg is built with
//! `serde_json::Map::new()` + explicit `insert` calls so the `preserve_order`
//! feature keeps the key order matching the TS object literal — this is
//! load-bearing for cache parity.

use serde_json::{Value, json};

use crate::store::cache::get_cache_key;

/// Cache key for `expandQuery(query, model, intent?, prompt_fingerprint?)`.
/// Mirrors TS `getCacheKey("expandQuery", { query, model, ...(intent && { intent }) })`
/// (`store.ts:3497`).
///
/// `prompt_fingerprint = ""` produces a **bit-identical** key to qmd and to
/// pre-feature rqmd — so existing caches stay valid for users who never set
/// `models.expand.*`. A non-empty fingerprint is appended as an extra
/// `promptFingerprint` field at the end of the body object, which is rqmd's
/// only divergence from qmd cache parity.
pub fn expand_query_cache_key(
    query: &str,
    model: &str,
    intent: Option<&str>,
    prompt_fingerprint: &str,
) -> String {
    let mut body = serde_json::Map::new();
    body.insert("query".into(), Value::String(query.into()));
    body.insert("model".into(), Value::String(model.into()));
    if let Some(i) = intent {
        body.insert("intent".into(), Value::String(i.into()));
    }
    // Only include the fingerprint field when non-empty so the default-config
    // cache key stays parity with qmd (and with pre-feature rqmd).
    if !prompt_fingerprint.is_empty() {
        body.insert(
            "promptFingerprint".into(),
            Value::String(prompt_fingerprint.into()),
        );
    }
    let body_str =
        serde_json::to_string(&Value::Object(body)).expect("serializing a string map cannot fail");
    get_cache_key("expandQuery", &body_str)
}

/// Cache key for `rerank`. The first argument is the **already-composed**
/// rerank query (i.e. `format!("{intent}\n\n{query}")` when intent is
/// present, else the raw query) — caller responsibility. Mirrors TS
/// `getCacheKey("rerank", { query: rerankQuery, model, chunk: doc.text })`
/// (`store.ts:3547`).
pub fn rerank_cache_key(rerank_query: &str, model: &str, chunk: &str) -> String {
    let body = json!({ "query": rerank_query, "model": model, "chunk": chunk });
    let body_str = serde_json::to_string(&body).expect("serializing a string map cannot fail");
    get_cache_key("rerank", &body_str)
}

/// Backward-compat key from before the intent-prepending change. Used to
/// migrate old cache entries — see `store.ts:3548`.
pub fn legacy_rerank_cache_key(query: &str, file: &str, model: &str, chunk: &str) -> String {
    let body = json!({
        "query": query,
        "file": file,
        "model": model,
        "chunk": chunk,
    });
    let body_str = serde_json::to_string(&body).expect("serializing a string map cannot fail");
    get_cache_key("rerank", &body_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_query_key_matches_ts_no_intent() {
        // sha256("expandQuery" || JSON.stringify({query:"hi",model:"m"}))
        assert_eq!(
            expand_query_cache_key("hi", "m", None, ""),
            "76c821b4ed2aba79b3bd1757beefaee75160d41b337811f51ce1a6e575a63bb4"
        );
    }

    #[test]
    fn expand_query_key_matches_ts_japanese() {
        // sha256("expandQuery" || JSON.stringify({query:"こんにちは",model:"m"}))
        assert_eq!(
            expand_query_cache_key("こんにちは", "m", None, ""),
            "20f8913e9743bfb22d0f8e6122ff1ba21182bec1a055773137badde29dc902ef"
        );
    }

    #[test]
    fn expand_query_key_with_intent_includes_intent_field() {
        // The TS spread `...(intent && { intent })` produces
        // `{ query, model, intent }` in source order — preserve_order keeps it.
        let with = expand_query_cache_key("hi", "m", Some("search"), "");
        let without = expand_query_cache_key("hi", "m", None, "");
        assert_ne!(with, without);
        // Re-running with the same intent produces the same key.
        assert_eq!(with, expand_query_cache_key("hi", "m", Some("search"), ""));
    }

    #[test]
    fn empty_fingerprint_is_bit_identical_to_no_fingerprint() {
        // The whole point of the empty-fingerprint short-circuit: pre-feature
        // cache keys must stay valid. Concretely, this must equal the known
        // qmd-parity SHA-256 used in `expand_query_key_matches_ts_no_intent`.
        assert_eq!(
            expand_query_cache_key("hi", "m", None, ""),
            "76c821b4ed2aba79b3bd1757beefaee75160d41b337811f51ce1a6e575a63bb4"
        );
    }

    #[test]
    fn nonempty_fingerprint_changes_key() {
        let baseline = expand_query_cache_key("hi", "m", None, "");
        let with_fp = expand_query_cache_key("hi", "m", None, "abc123");
        assert_ne!(baseline, with_fp);
        // Same fingerprint → same key (stable hashing on this layer).
        assert_eq!(with_fp, expand_query_cache_key("hi", "m", None, "abc123"));
        // Different fingerprint → different key.
        assert_ne!(with_fp, expand_query_cache_key("hi", "m", None, "def456"));
    }

    #[test]
    fn rerank_key_matches_ts_known_input() {
        // sha256("rerank" || JSON.stringify({query:"intent\n\nq",model:"m",chunk:"text"}))
        // — JSON.stringify escapes the U+000A newlines in the query string to
        // literal backslash-n pairs.
        assert_eq!(
            rerank_cache_key("intent\n\nq", "m", "text"),
            "bb87120a9fcce50bf338d64b7316648275c162bf8bc3d89756620ef8bb7a3ce5"
        );
    }

    #[test]
    fn legacy_rerank_key_uses_file_and_raw_query() {
        // Pre-intent format: keys include `file`. Used only for read-side
        // migration; we never produce new entries with this shape.
        let key = legacy_rerank_cache_key("q", "f.md", "m", "chunk");
        // Pinned regression — recompute if format ever changes.
        let expected_input = r#"{"query":"q","file":"f.md","model":"m","chunk":"chunk"}"#;
        assert_eq!(key, get_cache_key("rerank", expected_input));
    }
}
