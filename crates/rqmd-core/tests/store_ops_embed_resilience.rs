//! End-to-end tests for the embed-resilience layer (`generate_embeddings`'s
//! per-chunk failure tracking + bounded retries). Ports of qmd's two
//! resilience tests in `test/store.test.ts`:
//!   * "does not mark a partially embedded multi-chunk document complete"
//!   * "clears chunk errors after successful retry"
//!
//! Failures are injected through `MockLlm` (`fail_embed_after` /
//! `fail_embed_batch_from`), mirroring qmd's per-test inline fakes. This is
//! possible because `generate_embeddings` accepts `Arc<dyn Llm>`.

mod common;

use std::sync::Arc;

use common::mock_llm::MockLlm;
use rqmd_core::StoreOpsEmbedOptions;
use rqmd_core::store::Store;
use rqmd_core::store_ops::generate_embeddings;
use tempfile::NamedTempFile;

/// Seed one active document (content + documents rows) so
/// `get_pending_embedding_docs` returns it.
fn seed_doc(store: &mut Store, hash: &str, path: &str, body: &str) {
    store.with_connection_mut(|c| {
        c.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?1, ?2, 'ts')",
            (hash, body),
        )
        .unwrap();
        c.execute(
            "INSERT INTO documents \
             (collection, path, title, hash, created_at, modified_at, active) \
             VALUES ('default', ?1, ?2, ?3, 'ts', 'ts', 1)",
            (path, path, hash),
        )
        .unwrap();
    });
}

fn content_vectors_count(store: &Store) -> i64 {
    store.with_connection(|c| {
        c.query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap()
    })
}

/// A body large enough to yield several token-chunks under the MockLlm
/// tokenizer (chars_per_token = 3, CHUNK_SIZE_TOKENS = 900): ~6.7k chars
/// ⇒ ~2.2k tokens ⇒ multiple chunks, well under one 32-chunk sub-batch.
fn multichunk_body() -> String {
    "lorem ipsum dolor sit amet ".repeat(250)
}

fn opts() -> StoreOpsEmbedOptions {
    StoreOpsEmbedOptions {
        model: Some("test-embed".into()),
        ..Default::default()
    }
}

fn open_store() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

#[tokio::test]
async fn embeds_all_chunks_when_llm_never_fails() {
    let (_t, mut store) = open_store();
    seed_doc(&mut store, "h1", "doc.md", &multichunk_body());
    let mock = Arc::new(MockLlm::new(4));

    let r = generate_embeddings(&mut store, mock.clone(), opts())
        .await
        .unwrap();

    assert_eq!(r.docs_processed, 1);
    assert!(r.chunks_embedded >= 2, "expected a multi-chunk document");
    assert_eq!(r.errors, 0);
    assert!(r.failures.is_empty());
    assert_eq!(content_vectors_count(&store), r.chunks_embedded as i64);
}

/// qmd: "does not mark a partially embedded multi-chunk document complete".
/// Only chunk 0 ever embeds (batch slot 0 + the dim probe); every later chunk
/// fails in the batch and never recovers, so each hits the retry cap and the
/// partially-embedded document is rolled back.
#[tokio::test]
async fn partial_failure_caps_attempts_and_removes_incomplete_doc() {
    let (_t, mut store) = open_store();
    seed_doc(&mut store, "h1", "doc.md", &multichunk_body());

    let mock = Arc::new(MockLlm::new(4));
    // Dim probe (the first `embed` call) succeeds; all later `embed` calls
    // (the individual retries) return None → permanent failure.
    mock.fail_embed_after(1);
    // In the batch, only slot 0 returns a vector; the rest are None.
    mock.fail_embed_batch_from(1);

    let r = generate_embeddings(&mut store, mock.clone(), opts())
        .await
        .unwrap();

    assert!(r.errors > 0, "expected unrecovered failures");
    assert_eq!(r.errors, r.failures.len());
    assert!(
        r.failures.iter().all(|f| f.attempts == 3),
        "each failure should hit MAX_RETRY_ATTEMPTS (3): {:?}",
        r.failures
    );
    assert_eq!(
        r.failures[0].reason, "embedding returned no vector",
        "retry failures record the None-result reason"
    );
    // Partially-embedded document is rolled back: no vectors survive.
    assert_eq!(r.chunks_embedded, 0);
    assert_eq!(content_vectors_count(&store), 0);
}

/// qmd: "clears chunk errors after successful retry". The batch fails for
/// every slot except 0, but the individual-embed fallback always succeeds, so
/// every failed chunk recovers on retry and the final error count is zero.
#[tokio::test]
async fn recovers_failed_chunks_via_individual_retry() {
    let (_t, mut store) = open_store();
    seed_doc(&mut store, "h1", "doc.md", &multichunk_body());

    let mock = Arc::new(MockLlm::new(4));
    // Batch fails for slots >= 1, but `embed` (the retry path) always succeeds.
    mock.fail_embed_batch_from(1);

    let r = generate_embeddings(&mut store, mock.clone(), opts())
        .await
        .unwrap();

    assert!(r.chunks_embedded >= 2, "expected a multi-chunk document");
    assert_eq!(r.errors, 0, "all failures should recover on retry");
    assert!(r.failures.is_empty());
    assert_eq!(content_vectors_count(&store), r.chunks_embedded as i64);
}
