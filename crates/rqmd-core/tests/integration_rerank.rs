//! Real-GGUF rerank test for `LlamaCpp`.
//!
//! Run with `cargo test -p rqmd-core -- --ignored integration_rerank`.
//! Mirrors `spike_03_rerank.rs`: with the Qwen3-Reranker chat-format
//! prompt, `tokio` must score higher than `rails` / `json` for the
//! query "rust async runtime".

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{RerankDocument, RerankOptions};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-Reranker-0.6B (~600 MB) and runs CPU inference"]
async fn rerank_ranks_relevant_doc_first() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        rerank_parallelism: Some(2),
        ..Default::default()
    });
    let docs = vec![
        RerankDocument {
            file: "tokio.md".into(),
            text: "Tokio is an asynchronous runtime for the Rust programming language.".into(),
            title: None,
        },
        RerankDocument {
            file: "rails.md".into(),
            text: "Ruby on Rails is a server-side web application framework written in Ruby."
                .into(),
            title: None,
        },
        RerankDocument {
            file: "json.md".into(),
            text: "JSON Schema is a vocabulary for annotating and validating JSON documents."
                .into(),
            title: None,
        },
    ];

    let result = llm
        .rerank("rust async runtime", &docs, RerankOptions::default())
        .await
        .expect("rerank must succeed against the real Qwen3-Reranker model");

    assert_eq!(result.results.len(), 3);
    let order: Vec<&str> = result.results.iter().map(|r| r.file.as_str()).collect();
    assert_eq!(
        order[0], "tokio.md",
        "expected tokio to rank first; got {order:?}"
    );
    assert_eq!(result.model, llm.rerank_model_uri());
    // The score separation should be large with the canonical prompt
    // (spike_03 measured 0.998 vs ~0.001 levels). Don't pin exact
    // values, just sanity-check the ordering reflects a real signal.
    assert!(
        result.results[0].score > result.results[2].score + 0.1,
        "expected meaningful score separation; got {:?}",
        result.results
    );

    llm.dispose().await;
}
