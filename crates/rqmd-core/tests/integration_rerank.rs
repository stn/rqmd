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

fn rerank_llm() -> LlamaCpp {
    LlamaCpp::new(LlamaCppConfig {
        rerank_parallelism: Some(2),
        ..Default::default()
    })
}

/// TS: rerank "scores authentication query correctly" — the two auth docs must
/// land in the top two regardless of order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-Reranker-0.6B (~600 MB) and runs CPU inference"]
async fn rerank_scores_authentication_query_correctly() {
    let llm = rerank_llm();
    let docs = vec![
        RerankDocument {
            file: "weather.md".into(),
            text: "The weather today is sunny with mild temperatures.".into(),
            title: None,
        },
        RerankDocument {
            file: "auth.md".into(),
            text: "Authentication can be configured by setting the AUTH_SECRET environment variable."
                .into(),
            title: None,
        },
        RerankDocument {
            file: "pizza.md".into(),
            text: "Our restaurant serves the best pizza in town.".into(),
            title: None,
        },
        RerankDocument {
            file: "jwt.md".into(),
            text: "JWT authentication requires a secret key and expiration time.".into(),
            title: None,
        },
    ];

    let result = llm
        .rerank(
            "How do I configure authentication?",
            &docs,
            RerankOptions::default(),
        )
        .await
        .expect("rerank must succeed");

    assert_eq!(result.results.len(), 4);
    let top_two: Vec<&str> = result.results.iter().take(2).map(|r| r.file.as_str()).collect();
    assert!(top_two.contains(&"auth.md"), "auth.md must rank top-2; got {top_two:?}");
    assert!(top_two.contains(&"jwt.md"), "jwt.md must rank top-2; got {top_two:?}");

    llm.dispose().await;
}

/// TS: rerank "handles single document", "preserves original file paths",
/// "returns scores between 0 and 1" — combined into one model load.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-Reranker-0.6B (~600 MB) and runs CPU inference"]
async fn rerank_single_doc_preserves_paths_and_bounds_scores() {
    let llm = rerank_llm();

    // Single document round-trips with its original path.
    let one = llm
        .rerank(
            "test",
            &[RerankDocument {
                file: "path/to/doc.md".into(),
                text: "content".into(),
                title: None,
            }],
            RerankOptions::default(),
        )
        .await
        .expect("rerank must succeed");
    assert_eq!(one.results.len(), 1);
    assert_eq!(one.results[0].file, "path/to/doc.md");

    // Scores stay within [0, 1] and original paths are preserved as a set.
    let docs = vec![
        RerankDocument {
            file: "another/path/doc1.md".into(),
            text: "The quick brown fox jumps over the lazy dog.".into(),
            title: None,
        },
        RerankDocument {
            file: "deep/nested/path/doc2.md".into(),
            text: "Machine learning algorithms process data efficiently.".into(),
            title: None,
        },
        RerankDocument {
            file: "doc3.md".into(),
            text: "React components use JSX syntax for rendering.".into(),
            title: None,
        },
    ];
    let result = llm
        .rerank("Tell me about animals", &docs, RerankOptions::default())
        .await
        .expect("rerank must succeed");

    for r in &result.results {
        assert!(
            (0.0..=1.0).contains(&r.score) && !r.score.is_nan(),
            "score out of [0,1]: {}",
            r.score
        );
    }
    let mut files: Vec<&str> = result.results.iter().map(|r| r.file.as_str()).collect();
    files.sort_unstable();
    assert_eq!(
        files,
        vec!["another/path/doc1.md", "deep/nested/path/doc2.md", "doc3.md"]
    );

    llm.dispose().await;
}

/// TS: rerank "truncates and reranks document exceeding 2048 token context".
///
/// rqmd has no token-truncation seam (unlike TS), so this can't assert
/// truncation; the default rerank context is 4096
/// (`DEFAULT_RERANK_CONTEXT_SIZE`). The repeated paragraph below is ~3200
/// tokens — large but within budget — so this verifies a large document is
/// handled (no GGML assert / crash, scores valid) rather than a truncation
/// boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-Reranker-0.6B (~600 MB) and runs CPU inference"]
async fn rerank_handles_large_document_within_context() {
    let llm = rerank_llm();

    let paragraph = "The quick brown fox jumps over the lazy dog near the riverbank. \
        Authentication tokens must be validated on every request to ensure security. \
        Database queries should use prepared statements to prevent SQL injection attacks. \
        The deployment pipeline includes linting, testing, building, and publishing stages. ";
    let long_text = paragraph.repeat(40); // ~12.8 KB ≈ 3200 tokens, < 4096 ctx

    let docs = vec![
        RerankDocument {
            file: "short-relevant.md".into(),
            text: "Authentication can be configured by setting AUTH_SECRET.".into(),
            title: None,
        },
        RerankDocument {
            file: "long-doc.md".into(),
            text: long_text,
            title: None,
        },
        RerankDocument {
            file: "short-irrelevant.md".into(),
            text: "The weather is sunny today.".into(),
            title: None,
        },
    ];

    let result = llm
        .rerank(
            "How do I configure authentication?",
            &docs,
            RerankOptions::default(),
        )
        .await
        .expect("rerank must not crash on a large (in-budget) document");

    assert_eq!(result.results.len(), 3);
    for r in &result.results {
        assert!(
            (0.0..=1.0).contains(&r.score) && !r.score.is_nan(),
            "score out of [0,1]: {}",
            r.score
        );
    }

    llm.dispose().await;
}
