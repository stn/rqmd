//! Integration tests for `expand_query` and `rerank`. Uses MockLlm +
//! `tempfile`-backed `Store::open` (no GGUF).

mod common;

use std::sync::Arc;

use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{QueryType, Queryable};
use rqmd_core::store::Store;
use rqmd_core::store::cache::{get_cache_key, get_cached_result};
use rqmd_core::store_ops::{ExpandedQueryType, RerankCandidate, expand_query, rerank};
use tempfile::NamedTempFile;

use common::mock_llm::MockLlm;

fn open_store() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

#[tokio::test]
async fn expand_query_calls_llm_then_caches() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_expand(
        "q",
        vec![
            Queryable {
                type_: QueryType::Lex,
                text: "literal q".into(),
            },
            Queryable {
                type_: QueryType::Vec,
                text: "semantic q".into(),
            },
        ],
    );

    let llm: Arc<dyn Llm> = mock.clone();
    let r = expand_query(&store, llm.clone(), "q", "m", None)
        .await
        .unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].type_, ExpandedQueryType::Lex);
    assert_eq!(r[0].query, "literal q");

    // Second call should hit cache — no extra LLM call.
    let calls_before = mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed);
    let r2 = expand_query(&store, llm.clone(), "q", "m", None)
        .await
        .unwrap();
    assert_eq!(r2, r);
    let calls_after = mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(calls_before, calls_after, "cache hit should not call LLM");
}

#[tokio::test]
async fn expand_query_empty_intent_shares_cache_with_none() {
    use std::sync::atomic::Ordering::Relaxed;

    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();

    // No intent populates the cache (synthetic [hyde, vec] is non-empty).
    let _ = expand_query(&store, llm.clone(), "q", "m", None)
        .await
        .unwrap();
    assert_eq!(mock.expand_calls.load(Relaxed), 1);

    // Empty intent normalizes to `None` → same cache key → cache hit, no new
    // LLM call. Fails if `expand_query`'s empty-intent normalization is dropped
    // (the `"intent":""` key would miss and re-expand).
    let _ = expand_query(&store, llm.clone(), "q", "m", Some(""))
        .await
        .unwrap();
    assert_eq!(
        mock.expand_calls.load(Relaxed),
        1,
        "empty intent must reuse the no-intent cache entry"
    );

    // Sanity: a real intent is a distinct cache key → a fresh expansion.
    let _ = expand_query(&store, llm.clone(), "q", "m", Some("real domain"))
        .await
        .unwrap();
    assert_eq!(
        mock.expand_calls.load(Relaxed),
        2,
        "a non-empty intent is a distinct cache key"
    );
}

#[tokio::test]
async fn expand_query_filters_duplicates_of_original() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_expand(
        "q",
        vec![
            Queryable {
                type_: QueryType::Lex,
                text: "q".into(), // exact duplicate — should be dropped
            },
            Queryable {
                type_: QueryType::Hyde,
                text: "doc about q".into(),
            },
        ],
    );

    let r = expand_query(&store, mock as Arc<dyn Llm>, "q", "m", None)
        .await
        .unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].type_, ExpandedQueryType::Hyde);
}

#[tokio::test]
async fn expand_query_does_not_cache_empty_result() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_expand("q", vec![]);

    let llm: Arc<dyn Llm> = mock.clone();
    let r = expand_query(&store, llm.clone(), "q", "m", None)
        .await
        .unwrap();
    assert!(r.is_empty());

    // Second call: should call LLM again (no cache write happened).
    let calls_before = mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed);
    let _ = expand_query(&store, llm, "q", "m", None).await.unwrap();
    let calls_after = mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(calls_after, calls_before + 1);
}

#[tokio::test]
async fn expand_query_migrates_legacy_text_format() {
    let (_t, store) = open_store();
    // Inject legacy-format cache entry directly.
    let body = r#"{"query":"q","model":"m"}"#;
    let key = get_cache_key("expandQuery", body);
    store
        .with_connection(|c| {
            rqmd_core::store::cache::set_cached_result(
                c,
                &key,
                r#"[{"type":"lex","text":"legacy entry"}]"#,
            )
        })
        .unwrap();

    let mock = Arc::new(MockLlm::new(4));
    let r = expand_query(&store, mock as Arc<dyn Llm>, "q", "m", None)
        .await
        .unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].type_, ExpandedQueryType::Lex);
    assert_eq!(r[0].query, "legacy entry");
}

#[tokio::test]
async fn rerank_sorts_descending_and_uses_intent_in_cache_key() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_rerank_score("matches well", 0.9);
    mock.set_rerank_score("less relevant", 0.2);

    let docs = vec![
        RerankCandidate {
            file: "a.md".into(),
            text: "less relevant".into(),
        },
        RerankCandidate {
            file: "b.md".into(),
            text: "matches well".into(),
        },
    ];
    let r = rerank(
        &store,
        mock.clone() as Arc<dyn Llm>,
        "q",
        &docs,
        "model",
        Some("intent"),
    )
    .await
    .unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].file, "b.md");
    assert!(r[0].score > r[1].score);

    // The cache key MUST include the intent-prepended query so a no-intent
    // call doesn't hit the same cache entry.
    let calls_before = mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed);
    let _ = rerank(
        &store,
        mock.clone() as Arc<dyn Llm>,
        "q",
        &docs,
        "model",
        Some("intent"),
    )
    .await
    .unwrap();
    let calls_after_same_intent = mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        calls_before, calls_after_same_intent,
        "same intent should hit cache"
    );

    let _ = rerank(
        &store,
        mock.clone() as Arc<dyn Llm>,
        "q",
        &docs,
        "model",
        None,
    )
    .await
    .unwrap();
    let calls_after_no_intent = mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        calls_after_no_intent > calls_after_same_intent,
        "different intent must miss cache"
    );
}

#[tokio::test]
async fn rerank_caches_per_chunk_text_not_file() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_rerank_score("shared chunk", 0.7);

    // Same chunk text across two files.
    let docs = vec![
        RerankCandidate {
            file: "a.md".into(),
            text: "shared chunk".into(),
        },
        RerankCandidate {
            file: "b.md".into(),
            text: "shared chunk".into(),
        },
    ];
    let r = rerank(&store, mock.clone() as Arc<dyn Llm>, "q", &docs, "m", None)
        .await
        .unwrap();
    // Both files get the same score.
    assert!((r[0].score - 0.7).abs() < 1e-5);
    assert!((r[1].score - 0.7).abs() < 1e-5);

    // The mock's rerank was called once on the dedup'd uncached set (1 entry).
    assert_eq!(
        mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Verify the cache stores the score keyed on chunk text — second
    // invocation should be a pure cache hit.
    let calls_before = mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed);
    let _ = rerank(&store, mock.clone() as Arc<dyn Llm>, "q", &docs, "m", None)
        .await
        .unwrap();
    assert_eq!(
        mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed),
        calls_before
    );

    // Sanity: the cached value is recoverable directly via the public key
    // helper.
    let key_body = r#"{"query":"q","model":"m","chunk":"shared chunk"}"#;
    let cached = store
        .with_connection(|c| get_cached_result(c, &get_cache_key("rerank", key_body)))
        .unwrap();
    assert_eq!(cached.as_deref(), Some("0.7"));
}

#[tokio::test]
async fn rerank_falls_back_to_zero_for_unrecognised_documents() {
    let (_t, store) = open_store();
    let mock = Arc::new(MockLlm::new(4));
    // No score configured — keyword overlap path; LLM never returns nothing
    // for our docs, so they should still get a (possibly 0.0) score.

    let docs = vec![RerankCandidate {
        file: "a.md".into(),
        text: "nothing in common".into(),
    }];
    let r = rerank(&store, mock as Arc<dyn Llm>, "q", &docs, "m", None)
        .await
        .unwrap();
    assert_eq!(r.len(), 1);
    // Mock returns keyword_overlap_score which is 0.0 for no overlap.
    assert_eq!(r[0].score, 0.0);
}
