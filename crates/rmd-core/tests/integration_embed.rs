//! Real-GGUF embed-path test for `LlamaCpp`.
//!
//! Run with `cargo test -p rmd-core -- --ignored integration_embed`.
//! Downloads embeddinggemma-300M on first call (~300 MB), then
//! exercises the worker pool through `Llm::embed_batch`.

use rmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rmd_core::llm::traits::Llm;
use rmd_core::llm::types::EmbedOptions;

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
    let first_dim = results[0].as_ref().expect("first result Some").embedding.len();
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
        // Pin context size explicitly — without this, `QMD_EMBED_CONTEXT_SIZE`
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
