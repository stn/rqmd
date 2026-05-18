//! Real-GGUF embed-path test for `LlamaCpp`.
//!
//! Run with `cargo test -p rmd-llm -- --ignored integration_embed`.
//! Downloads Qwen3-Embedding-0.6B on first call (~600 MB), then
//! exercises the worker pool through `Llm::embed_batch`.

use rmd_llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rmd_llm::traits::Llm;
use rmd_llm::types::EmbedOptions;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-Embedding-0.6B (~600 MB) and runs CPU inference"]
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
        .expect("embed_batch must succeed against the real Qwen3-Embedding model");

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
