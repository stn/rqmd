//! Tests for `search_vec` (LLM orchestrator wrapping
//! `search_vec_with_embedding`).

mod common;

use std::sync::Arc;

use rmd_core::store::embeddings::{ensure_vec_table, insert_embedding};
use rmd_core::store::Store;
use rmd_core::store_ops::search_vec;
use rmd_core::llm::traits::Llm;
use rmd_core::db::rusqlite::params;
use tempfile::NamedTempFile;

use common::mock_llm::MockLlm;

fn open_store_with_docs() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let mut store = Store::open(tmp.path()).unwrap();

    let body = "body of a";
    store
        .with_connection_mut(|c| {
            c.execute(
                "INSERT INTO content (hash, doc, created_at) VALUES ('h1', ?, 'ts')",
                params![body],
            )
            .unwrap();
            c.execute(
                "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
                 VALUES ('c', 'a.md', 'a', 'h1', 'ts', 'ts', 1)",
                [],
            )
            .unwrap();
            ensure_vec_table(c, 4).unwrap();
            insert_embedding(c, "h1", 0, 0, &[1.0, 0.0, 0.0, 0.0], "m", "ts", 1).unwrap();
        });

    (tmp, store)
}

#[tokio::test]
async fn search_vec_returns_empty_when_table_missing() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    let mock = Arc::new(MockLlm::new(4));
    let r = search_vec(&store, mock as Arc<dyn Llm>, "q", "m", 5, None, None)
        .await
        .unwrap();
    assert!(r.is_empty());
}

#[tokio::test]
async fn search_vec_with_precomputed_embedding_skips_llm_embed() {
    let (_t, store) = open_store_with_docs();
    let mock = Arc::new(MockLlm::new(4));
    let llm: Arc<dyn Llm> = mock.clone();

    let calls_before = mock.embed_calls.load(std::sync::atomic::Ordering::Relaxed);
    let r = search_vec(
        &store,
        llm,
        "q",
        "m",
        5,
        None,
        Some(&[1.0, 0.0, 0.0, 0.0]),
    )
    .await
    .unwrap();
    let calls_after = mock.embed_calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(calls_before, calls_after, "precomputed must skip embed");
    assert_eq!(r.len(), 1);
    assert!(r[0].score > 0.9);
}

#[tokio::test]
async fn search_vec_embeds_query_when_no_precomputed() {
    let (_t, store) = open_store_with_docs();
    let mock = Arc::new(MockLlm::new(4));
    // Make the LLM return the exact embedding that matches our inserted doc
    // so the kNN finds it. format_query_for_embedding rewrites the text;
    // we pre-set the embedding for the rewritten form.
    let formatted = rmd_core::llm::format::format_query_for_embedding("q", "m");
    mock.set_embed(formatted, vec![1.0, 0.0, 0.0, 0.0]);

    let llm: Arc<dyn Llm> = mock.clone();
    let r = search_vec(&store, llm, "q", "m", 5, None, None).await.unwrap();
    assert_eq!(r.len(), 1);
    assert!(r[0].score > 0.9);
    assert!(mock.embed_calls.load(std::sync::atomic::Ordering::Relaxed) >= 1);
}
