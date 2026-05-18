//! Tests for `generate_embeddings`. Full happy-path requires a real GGUF
//! (see `tests/integration_generate_embeddings.rs` — gated `#[ignore]`).
//! Here we exercise the LLM-free early-return paths with a CI-mode
//! `LlamaCpp` (every embed call returns `Err(CiDisabled)`).

mod common;

use std::sync::Arc;

use rmd_core::store::embeddings::{ensure_vec_table, insert_embedding};
use rmd_core::store::Store;
use rmd_llm::EmbedStoreOptions;
use rmd_llm::store_ops::generate_embeddings;
use rmd_llm::{LlamaCpp, LlamaCppConfig};
use tempfile::NamedTempFile;

fn ci_mode_llm() -> Arc<LlamaCpp> {
    Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }))
}

fn open_store_empty() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

#[tokio::test]
async fn empty_collection_returns_zero_without_calling_llm() {
    let (_t, mut store) = open_store_empty();
    let llm = ci_mode_llm();
    let r = generate_embeddings(&mut store, llm, EmbedStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(r.docs_processed, 0);
    assert_eq!(r.chunks_embedded, 0);
    assert_eq!(r.errors, 0);
}

#[tokio::test]
async fn force_with_no_docs_returns_zero_and_clears_vectors_table() {
    let (_t, mut store) = open_store_empty();
    // Pre-populate a stale embedding so we can verify `force` clears it.
    store.with_connection_mut(|c| {
        c.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES ('h_stale', 'x', 'ts')",
            [],
        )
        .unwrap();
        // No active document → hash is orphaned, but `force=true` should still
        // attempt the clear.
        ensure_vec_table(c, 4).unwrap();
        insert_embedding(c, "h_stale", 0, 0, &[1.0; 4], "m", "ts", 1).unwrap();
    });

    let llm = ci_mode_llm();
    let opts = EmbedStoreOptions {
        force: true,
        ..Default::default()
    };
    let r = generate_embeddings(&mut store, llm, opts).await.unwrap();
    assert_eq!(r.docs_processed, 0);
    // The global clear drops vectors_vec entirely.
    let table_count: i64 = store
        .with_connection(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        });
    assert_eq!(table_count, 0, "vectors_vec should have been dropped");
}

// NOTE: testing the happy path of `generate_embeddings` (multi-chunk
// docs, sub-batch sequencing, fallback) requires a real GGUF — see the
// `#[ignore]` integration tests for that. CI mode does NOT prevent the
// single-chunk dim probe from running (only `embed_batch` is CI-gated),
// so we cannot simulate the full pipeline without a real model.
