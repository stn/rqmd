//! Real-GGUF embed-path test for `LlamaCpp`.
//!
//! Run with `cargo test -p rqmd-core -- --ignored integration_embed`.
//! Downloads embeddinggemma-300M on first call (~300 MB), then
//! exercises the worker pool through `Llm::embed_batch`.

use std::sync::Arc;

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::EmbedOptions;

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn embed_batch_returns_one_vector_per_input_with_consistent_dim() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(2),
        ..Default::default()
    });
    let inputs: Vec<String> = vec![
        "tokio is an async runtime for rust".into(),
        "ruby on rails is a web framework".into(),
        "json schema validates documents".into(),
    ];
    let results = llm
        .embed_batch(&inputs, EmbedOptions::default())
        .await
        .expect("embed_batch must succeed against the real embed model");

    assert_eq!(results.len(), inputs.len());
    let first_dim = results[0]
        .as_ref()
        .expect("first result Some")
        .embedding
        .len();
    assert!(first_dim > 0, "embedding must be non-empty");
    for (i, slot) in results.iter().enumerate() {
        let r = slot
            .as_ref()
            .unwrap_or_else(|| panic!("input {i} must produce an embedding"));
        assert_eq!(
            r.embedding.len(),
            first_dim,
            "all embeddings must have the same dim ({first_dim})"
        );
        assert!(
            r.embedding.iter().any(|v| *v != 0.0),
            "embedding {i} must contain nonzero values"
        );
        assert_eq!(r.model, llm.embed_model_uri());
    }

    llm.dispose().await;
}

/// Regression for the encoder `n_ubatch >= n_tokens` assertion: feed a
/// text whose token count exceeds llama.cpp's default n_ubatch (512) but
/// stays well under our pinned `embed_context_size` (2048). Before
/// `worker::make_pool_ctx_params` pinned `n_batch = n_ubatch = n_ctx`,
/// the embed pool only set `with_n_ctx`, leaving n_batch / n_ubatch at
/// the 512 default — any input > 512 tokens aborted with GGML_ASSERT.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn embed_handles_input_larger_than_default_ubatch() {
    // ~6 KB of repeated technical English → ~1400-1500 tokens (well > 512,
    // well < 2048). Repetition is fine; we only need the encoder to
    // process a sequence longer than llama.cpp's default n_ubatch.
    let paragraph = "The Raft consensus algorithm is designed to be more \
        understandable than Paxos. It separates leader election from log \
        replication, which makes each piece easier to reason about and verify. ";
    let long_text: String = paragraph.repeat(30);
    assert!(long_text.len() > 4_000, "test input must be large enough");

    let llm = LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(1),
        // Pin context size explicitly — without this, `RQMD_EMBED_CONTEXT_SIZE`
        // from the developer's shell could change n_ctx and either mask the
        // bug (if set very high, ubatch ends up wider than needed) or
        // introduce an unrelated failure (if set very low, the input itself
        // overflows the context). 2048 matches DEFAULT_EMBED_CONTEXT_SIZE.
        embed_context_size: Some(2048),
        ..Default::default()
    });
    let results = llm
        .embed_batch(&[long_text], EmbedOptions::default())
        .await
        .expect("embed_batch must not panic on inputs > default n_ubatch");

    assert_eq!(results.len(), 1);
    let emb = &results[0].as_ref().expect("embedding present").embedding;
    assert!(emb.len() > 100, "embedding must be non-empty");
    assert!(
        emb.iter().any(|v| *v != 0.0),
        "embedding must contain nonzero values"
    );

    llm.dispose().await;
}

// =============================================================================
// B-1: consistency / difference (TS: embed "returns consistent embeddings for
// same input" + "returns different embeddings for different inputs")
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn embed_is_consistent_for_same_input_and_differs_for_distinct_inputs() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(1),
        ..Default::default()
    });

    // Same input -> (near-)identical embeddings.
    let a = llm
        .embed("test text", EmbedOptions::default())
        .await
        .unwrap()
        .expect("embedding present");
    let b = llm
        .embed("test text", EmbedOptions::default())
        .await
        .unwrap()
        .expect("embedding present");
    assert_eq!(a.embedding.len(), b.embedding.len());
    for (x, y) in a.embedding.iter().zip(b.embedding.iter()) {
        assert!(
            (x - y).abs() < 1e-4,
            "same input must yield stable embeddings ({x} vs {y})"
        );
    }

    // Distinct inputs -> cosine similarity < 0.95 (meaningfully different).
    let c = llm
        .embed("cats are great", EmbedOptions::default())
        .await
        .unwrap()
        .expect("embedding present");
    let d = llm
        .embed("database optimization", EmbedOptions::default())
        .await
        .unwrap()
        .expect("embedding present");
    let sim = cosine(&c.embedding, &d.embedding);
    assert!(sim < 0.95, "distinct inputs must differ; cosine={sim}");

    llm.dispose().await;
}

// =============================================================================
// B-2: embedBatch parity (TS: "returns same results as individual embed calls",
// "handles empty array", concurrent-without-race)
// =============================================================================

/// Empty input returns `Ok(vec![])`. The empty check sits AFTER `ensure_not_ci`
/// but BEFORE any model load, so a non-CI instance returns immediately with no
/// GGUF present — hence this test needs no `#[ignore]`.
#[tokio::test]
async fn embed_batch_empty_returns_empty_without_loading_a_model() {
    let llm = LlamaCpp::new(LlamaCppConfig::default());
    let results = llm
        .embed_batch(&[], EmbedOptions::default())
        .await
        .expect("empty batch must succeed");
    assert!(results.is_empty());
    llm.dispose().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn embed_batch_matches_individual_embed_calls() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(2),
        ..Default::default()
    });
    let texts: Vec<String> = vec!["cats are great".into(), "dogs are awesome".into()];

    let batch = llm
        .embed_batch(&texts, EmbedOptions::default())
        .await
        .expect("embed_batch must succeed");
    assert_eq!(batch.len(), texts.len());

    for (i, t) in texts.iter().enumerate() {
        let single = llm
            .embed(t, EmbedOptions::default())
            .await
            .unwrap()
            .expect("embedding present");
        let bv = &batch[i]
            .as_ref()
            .expect("batch embedding present")
            .embedding;
        assert_eq!(bv.len(), single.embedding.len());
        for (x, y) in bv.iter().zip(single.embedding.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "batch and individual embeddings must match ({x} vs {y})"
            );
        }
    }

    llm.dispose().await;
}

/// Five concurrent `embed_batch` calls on a fresh instance must all succeed.
/// This exercises the embed-pool creation guard under contention — a broken
/// guard would surface as a panic, deadlock, or "context disposed" error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn concurrent_embed_batch_on_fresh_instance_all_succeed() {
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(2),
        ..Default::default()
    }));
    let texts: Vec<String> = (0..10).map(|i| format!("Document {i}")).collect();

    let mut handles = Vec::new();
    for chunk in texts.chunks(2) {
        let llm = llm.clone();
        let chunk: Vec<String> = chunk.to_vec();
        handles.push(tokio::spawn(async move {
            llm.embed_batch(&chunk, EmbedOptions::default()).await
        }));
    }

    let mut total = 0usize;
    for h in handles {
        let res = h
            .await
            .unwrap()
            .expect("each concurrent embed_batch must succeed");
        for slot in res {
            assert!(slot.is_some(), "every input must produce an embedding");
            total += 1;
        }
    }
    assert_eq!(total, 10);

    llm.dispose().await;
}
