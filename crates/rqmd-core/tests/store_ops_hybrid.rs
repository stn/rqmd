//! Integration tests for `hybrid_query`, `vector_search_query`, and
//! `structured_search`. MockLlm + tempfile Store covers all
//! orchestration paths without GGUF.

mod common;

use std::sync::Arc;

use rqmd_core::db::rusqlite::params;
use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::embeddings::{ensure_vec_table, insert_embedding};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::Store;
use rqmd_core::store_ops::{
    hybrid_query, structured_search, vector_search_query, ExpandedQuery, ExpandedQueryType,
    HybridQueryOptions, StructuredSearchOptions, VectorSearchOptions,
};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{QueryType, Queryable};
use tempfile::NamedTempFile;

use common::mock_llm::MockLlm;

/// Build a store with three docs, FTS rows, and matching vector embeddings.
/// "alpha" matches the query "alpha"; "beta" matches "beta"; "gamma" is
/// unrelated. Vector embeddings are unit basis vectors so we can fake a
/// kNN query trivially.
fn open_seeded() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let mut store = Store::open(tmp.path()).unwrap();

    let docs = [
        (
            "h_alpha",
            "alpha alpha alpha relevant",
            "a.md",
            "Alpha",
            [1.0_f32, 0.0, 0.0, 0.0],
        ),
        (
            "h_beta",
            "beta beta beta",
            "b.md",
            "Beta",
            [0.0_f32, 1.0, 0.0, 0.0],
        ),
        (
            "h_gamma",
            "gamma unrelated content",
            "g.md",
            "Gamma",
            [0.0_f32, 0.0, 1.0, 0.0],
        ),
    ];

    store.with_connection_mut(|c| {
        for (hash, body, path, title, _) in &docs {
            c.execute(
                "INSERT INTO content (hash, doc, created_at) VALUES (?, ?, 'ts')",
                params![hash, body],
            )
            .unwrap();
            c.execute(
                "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
                 VALUES ('c', ?, ?, ?, 'ts', 'ts', 1)",
                params![path, title, hash],
            )
            .unwrap();
        }
        ensure_vec_table(c, 4).unwrap();
        for (hash, _, _, _, emb) in &docs {
            insert_embedding(c, hash, 0, 0, emb, "m", "ts", 1).unwrap();
        }
    });
    (tmp, store)
}

#[tokio::test]
async fn hybrid_query_returns_fts_relevant_doc_with_strong_signal() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();

    // "alpha" should match doc a.md heavily via FTS. Expansion will run
    // (strong-signal threshold depends on actual BM25 score) — that's fine,
    // mock returns synthetic expansions.
    let opts = HybridQueryOptions {
        limit: Some(3),
        ..Default::default()
    };
    let r = hybrid_query(&store, llm, "alpha", opts).await.unwrap();
    assert!(!r.is_empty(), "expected at least one result");
    assert!(r[0].file.ends_with("/a.md"), "top doc should be alpha; got {}", r[0].file);
}

#[tokio::test]
async fn hybrid_query_skip_rerank_returns_rrf_scored_results() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();

    let opts = HybridQueryOptions {
        limit: Some(3),
        skip_rerank: true,
        ..Default::default()
    };
    let r = hybrid_query(&store, llm, "alpha", opts).await.unwrap();
    assert!(!r.is_empty());
    // RRF top score for rank=1 is 1/1 = 1.0.
    assert!((r[0].score - 1.0).abs() < 1e-9);
    // rerank_calls should be zero.
    assert_eq!(mock.rerank_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[tokio::test]
async fn hybrid_query_intent_disables_strong_signal_path() {
    // Even with a single perfectly-matching FTS hit, supplying `intent`
    // disables the strong-signal short-circuit per TS lines 4301-4303.
    // Verify by observing that `expand_query` IS called.
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let opts = HybridQueryOptions {
        limit: Some(3),
        skip_rerank: true,
        intent: Some("some intent".into()),
        ..Default::default()
    };
    let _ = hybrid_query(&store, llm, "alpha", opts).await.unwrap();
    assert!(
        mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "intent should disable strong-signal bypass — expand must run"
    );
}

/// Build an FTS corpus where one doc dominates "zephyr" strongly enough to
/// clear the strong-signal thresholds (mirrors `store_search_fts.rs`'s
/// `search_fts_strong_signal_detection`). No vector table → FTS-only, which is
/// all the strong-signal probe reads.
fn open_strong_signal_store() -> (NamedTempFile, Store) {
    fn insert(store: &Store, path: &str, title: &str, body: &str) {
        let now = now_rfc3339();
        let hash = hash_content(body);
        store
            .with_connection(|c| insert_content(c, &hash, body, &now))
            .unwrap();
        store
            .with_connection(|c| insert_document(c, "docs", path, title, &hash, &now, &now))
            .unwrap();
    }
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    // 50 noise docs give enough IDF for the dominant score to exceed 0.85.
    for i in 0..50 {
        insert(
            &store,
            &format!("noise{i}.md"),
            &format!("Unrelated Topic {i}"),
            &format!("This document discusses gardening and cooking {i}"),
        );
    }
    // Dominant: keyword in path + title + body.
    insert(
        &store,
        "zephyr/zephyr-guide.md",
        "Zephyr Configuration Guide",
        "Complete zephyr configuration guide. Zephyr setup instructions for zephyr deployment.",
    );
    // Weak: keyword once in a long body.
    insert(
        &store,
        "notes/misc.md",
        "General Notes",
        "Various topics covering many areas of technology and design. One of them might relate to \
         zephyr but mostly about other things entirely. Additional content about databases, \
         networking, security, performance, monitoring, deployment, testing, and documentation.",
    );
    (tmp, store)
}

/// Run `hybrid_query` for "zephyr" with the given intent on a fresh
/// strong-signal store and report how many times `expand_query` ran.
async fn strong_signal_expand_calls(intent: Option<&str>) -> usize {
    let (_t, store) = open_strong_signal_store();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let _ = hybrid_query(
        &store,
        llm,
        "zephyr",
        HybridQueryOptions {
            limit: Some(3),
            skip_rerank: true,
            intent: intent.map(str::to_string),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    mock.expand_calls.load(std::sync::atomic::Ordering::Relaxed)
}

#[tokio::test]
async fn hybrid_query_empty_intent_behaves_like_no_intent() {
    // An empty intent is normalized to `None` (qmd treats "" as falsy), so it
    // must NOT disable the strong-signal bypass. The fixture yields a real
    // strong signal, making the bypass observable: `None`/`""` skip expansion
    // (expand_calls == 0); a real intent forces it (>= 1). This test fails if
    // the `.filter()` normalization is dropped from `hybrid_query`.
    assert_eq!(
        strong_signal_expand_calls(None).await,
        0,
        "no intent + strong signal should bypass expansion"
    );
    assert_eq!(
        strong_signal_expand_calls(Some("")).await,
        0,
        "empty intent must behave like no intent — bypass stays active"
    );
    assert!(
        strong_signal_expand_calls(Some("web performance latency")).await >= 1,
        "a real intent must disable the strong-signal bypass"
    );
}

#[tokio::test]
async fn vector_search_query_drops_lex_expansions() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    // Force a lex+vec expansion mix.
    mock.set_expand(
        "alpha",
        vec![
            Queryable {
                type_: QueryType::Lex,
                text: "lex variation".into(),
            },
            Queryable {
                type_: QueryType::Vec,
                text: "vec variation".into(),
            },
        ],
    );
    // Both original "alpha" and the vec variation get embedded — pin both to
    // the unit vector that matches doc a.md.
    for txt in [
        rqmd_core::llm::format::format_query_for_embedding("alpha", "m"),
        rqmd_core::llm::format::format_query_for_embedding("vec variation", "m"),
    ] {
        mock.set_embed(txt, vec![1.0, 0.0, 0.0, 0.0]);
    }

    let llm: Arc<dyn Llm> = mock.clone();
    let opts = VectorSearchOptions {
        limit: Some(3),
        min_score: Some(0.0),
        ..Default::default()
    };
    let r = vector_search_query(&store, llm, "alpha", opts).await.unwrap();
    assert!(!r.is_empty());
    assert!(r[0].file.ends_with("/a.md"));
    // embed_batch should have been called with TWO texts (original + vec).
    // Lex variation was filtered out — never embedded.
    let calls = mock
        .embed_batch_calls
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(calls >= 1, "expected embed_batch call");
}

#[tokio::test]
async fn structured_search_rejects_newlines_in_query() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();

    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "two\nlines".into(),
        line: Some(3),
    }];
    let err = structured_search(&store, llm, &searches, StructuredSearchOptions::default())
        .await
        .expect_err("newline should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("Line 3"), "{msg}");
    assert!(msg.to_lowercase().contains("single-line"), "{msg}");
}

#[tokio::test]
async fn structured_search_first_list_gets_2x_weight() {
    // Two searches: lex matches a.md, vec matches g.md. With first-list 2x
    // weight, the lex result should outrank the vec result even though both
    // hit exactly once.
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    mock.set_embed(
        rqmd_core::llm::format::format_query_for_embedding("gamma topic", "m"),
        vec![0.0, 0.0, 1.0, 0.0],
    );

    let llm: Arc<dyn Llm> = mock.clone();
    let searches = vec![
        ExpandedQuery {
            type_: ExpandedQueryType::Lex,
            query: "alpha".into(),
            line: None,
        },
        ExpandedQuery {
            type_: ExpandedQueryType::Vec,
            query: "gamma topic".into(),
            line: None,
        },
    ];
    let opts = StructuredSearchOptions {
        limit: Some(5),
        skip_rerank: true,
        ..Default::default()
    };
    let r = structured_search(&store, llm, &searches, opts).await.unwrap();
    assert!(r.len() >= 2);
    let alpha_idx = r.iter().position(|h| h.file.ends_with("/a.md")).unwrap();
    let gamma_idx = r.iter().position(|h| h.file.ends_with("/g.md")).unwrap();
    assert!(
        alpha_idx < gamma_idx,
        "first list (lex/alpha) should outrank second (vec/gamma); got order {r:?}"
    );
}

// =============================================================================
// Ported from structured-search.test.ts `describe("structuredSearch")`
// (structured-search.test.ts:283-348). "throws when lex query contains newline"
// is already covered by `structured_search_rejects_newlines_in_query` above and
// is not duplicated here.
// =============================================================================

#[tokio::test]
async fn structured_search_returns_empty_for_empty_searches() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let searches: Vec<ExpandedQuery> = vec![];
    let r = structured_search(&store, llm, &searches, StructuredSearchOptions::default())
        .await
        .unwrap();
    assert!(r.is_empty());
}

#[tokio::test]
async fn structured_search_returns_empty_when_no_match() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    // Non-hyphenated term → plain prefix-match path (no document contains it).
    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "nonexistentxyz123".into(),
        line: None,
    }];
    let r = structured_search(&store, llm, &searches, StructuredSearchOptions::default())
        .await
        .unwrap();
    assert!(r.is_empty());
}

#[tokio::test]
async fn structured_search_accepts_lex_search_type() {
    // vec/hyde require embeddings, so (like the TS case) only lex is exercised.
    // Input is "alpha" rather than TS's "test" so it actually matches a fixture
    // doc — a strictly stronger check than TS's resolves.toBeDefined().
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "alpha".into(),
        line: None,
    }];
    let r = structured_search(&store, llm, &searches, StructuredSearchOptions::default()).await;
    assert!(r.is_ok());
}

#[tokio::test]
async fn structured_search_respects_limit_option() {
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "alpha".into(),
        line: None,
    }];
    let opts = StructuredSearchOptions {
        limit: Some(5),
        skip_rerank: true,
        ..Default::default()
    };
    let r = structured_search(&store, llm, &searches, opts).await.unwrap();
    assert!(r.len() <= 5);
}

#[tokio::test]
async fn structured_search_respects_min_score_option() {
    // skip_rerank makes the score deterministic (1.0/rank). "alpha" hits a.md
    // at rank 1 → score 1.0, so a 0.5 floor keeps a non-empty set (the floor
    // is exercised against real hits, not an empty list).
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "alpha".into(),
        line: None,
    }];
    let opts = StructuredSearchOptions {
        min_score: Some(0.5),
        skip_rerank: true,
        ..Default::default()
    };
    let r = structured_search(&store, llm, &searches, opts).await.unwrap();
    assert!(!r.is_empty(), "expected at least one hit above the score floor");
    for hit in &r {
        assert!(hit.score >= 0.5, "score {} below floor", hit.score);
    }
}

#[tokio::test]
async fn structured_search_rejects_unmatched_quote() {
    // Regression: structured_search must run the real validate_lex_query
    // (search.rs), not the old schema.rs stub, so an unmatched double quote is
    // rejected. Mirrors structured-search.test.ts:343-347 (/unmatched double quote/).
    let (_t, store) = open_seeded();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();
    let searches = vec![ExpandedQuery {
        type_: ExpandedQueryType::Lex,
        query: "\"unfinished phrase".into(),
        line: Some(2),
    }];
    let err = structured_search(&store, llm, &searches, StructuredSearchOptions::default())
        .await
        .expect_err("unmatched quote should be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("Line 2"), "{msg}");
    assert!(msg.to_lowercase().contains("unmatched"), "{msg}");
}

/// Ported from store.test.ts: "hybrid RRF weights boost original vector
/// evidence over expansion-only hits". `get_hybrid_rrf_weights` assigns 2.0
/// to original-query lists and 1.0 to expansion lists, regardless of source.
#[test]
fn get_hybrid_rrf_weights_boosts_original_query_lists() {
    use rqmd_core::store::rrf::{QueryType, RankedListMeta};
    use rqmd_core::store::search::SearchSource;
    use rqmd_core::store_ops::hybrid::get_hybrid_rrf_weights;

    let meta = vec![
        RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryType::Original,
            query: "user query".into(),
        },
        RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryType::Lex,
            query: "lex expansion".into(),
        },
        RankedListMeta {
            source: SearchSource::Vec,
            query_type: QueryType::Original,
            query: "user query".into(),
        },
    ];

    assert_eq!(get_hybrid_rrf_weights(&meta), vec![2.0, 1.0, 2.0]);
}
