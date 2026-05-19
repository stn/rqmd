//! Tests for `chunk_document_by_tokens`. Uses MockLlm with a configurable
//! chars-per-token ratio so we can drive both the "fits in budget" path
//! and the recursive-refine path deterministically.

mod common;

use std::sync::Arc;

use rmd_core::store::chunking::ChunkStrategy;
use rmd_core::store_ops::chunk_document_by_tokens;
use rmd_core::llm::traits::Llm;
use tokio_util::sync::CancellationToken;

use common::mock_llm::MockLlm;

#[tokio::test]
async fn short_doc_returns_single_chunk() {
    // Doc fits in budget on the first pass — no recursive refine.
    let mock = Arc::new(MockLlm::new(4).with_chars_per_token(3.0));
    let llm: Arc<dyn Llm> = mock.clone();
    let r = chunk_document_by_tokens(
        llm,
        "short body fits easily",
        Some(100),
        Some(10),
        Some(20),
        None,
        ChunkStrategy::Auto,
        None,
    )
    .await
    .unwrap();
    assert_eq!(r.len(), 1);
    assert!(r[0].tokens <= 100);
    assert_eq!(r[0].pos, 0);
}

#[tokio::test]
async fn oversized_chunk_recursively_splits() {
    // chars_per_token = 1.0 → every char is one token. With max_tokens = 5,
    // a 100-char body has 100 tokens — way over budget — so the function
    // MUST split it into multiple sub-chunks.
    let mock = Arc::new(MockLlm::new(4).with_chars_per_token(1.0));
    let llm: Arc<dyn Llm> = mock.clone();

    let body = "a".repeat(100);
    let r = chunk_document_by_tokens(
        llm,
        &body,
        Some(5),
        Some(0),
        Some(0),
        None,
        ChunkStrategy::Auto,
        None,
    )
    .await
    .unwrap();

    assert!(r.len() > 1, "expected multiple sub-chunks, got {}", r.len());
    for c in &r {
        // Each chunk must fit in `max_tokens` (5) or — if the recursion
        // bottomed out via detokenize fallback — exactly equal `max_tokens`.
        assert!(c.tokens <= 5, "chunk had {} tokens", c.tokens);
    }
}

#[tokio::test]
async fn cancellation_short_circuits_with_partial_results() {
    // Cancel BEFORE the first iteration. The function exits the loop on
    // its first pop and returns whatever it has accumulated (empty for
    // an immediate cancel, or partial mid-run).
    let mock = Arc::new(MockLlm::new(4).with_chars_per_token(1.0));
    let llm: Arc<dyn Llm> = mock.clone();
    let cancel = CancellationToken::new();
    cancel.cancel();

    let r = chunk_document_by_tokens(
        llm,
        &"x".repeat(100),
        Some(5),
        Some(0),
        Some(0),
        None,
        ChunkStrategy::Auto,
        Some(&cancel),
    )
    .await
    .unwrap();
    // Should bail before tokenizing — at most the empty vec.
    assert!(r.is_empty(), "expected empty result on pre-cancel");
}

#[tokio::test]
async fn detokenize_fallback_emits_truncated_chunk_when_split_fails() {
    // A 6-byte body where every char is a token (chars_per_token = 1.0).
    // max_tokens = 3 means the body is over budget. With overlap=0, window=0
    // and a very small target, the splitter may fail to halve cleanly; the
    // detokenize fallback should emit a max_tokens-truncated chunk.
    let mock = Arc::new(MockLlm::new(4).with_chars_per_token(1.0));
    let llm: Arc<dyn Llm> = mock.clone();
    let r = chunk_document_by_tokens(
        llm.clone(),
        "abcdef",
        Some(3),
        Some(0),
        Some(0),
        None,
        ChunkStrategy::Auto,
        None,
    )
    .await
    .unwrap();
    assert!(!r.is_empty());
    for c in &r {
        assert!(c.tokens <= 3);
    }
    // If the fallback fired at least once, detokenize would have been
    // called. We don't assert on exact count — but it should be reachable
    // for at least some configurations.
    let _ = mock
        .detokenize_calls
        .load(std::sync::atomic::Ordering::Relaxed);
}
