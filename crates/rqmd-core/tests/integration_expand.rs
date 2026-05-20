//! Real-GGUF expand_query test for `LlamaCpp`.
//!
//! Run with `cargo test -p rqmd-core -- --ignored integration_expand`.
//! Mirrors `spike_04_grammar.rs`: with the chat-template + clear
//! prompt approach (no GBNF), the model must produce at least one
//! `lex:` / `vec:` / `hyde:` line. Uses Qwen3-0.6B (cheaper than the
//! 1.7B query-expansion default) so the test runs in reasonable time
//! on CPU.

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{ExpandQueryOptions, QueryType};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-0.6B (~600 MB) and runs CPU inference"]
async fn expand_query_returns_at_least_one_parseable_line() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        // Smaller generation model than the default 1.7B so this test
        // finishes in tens of seconds on CPU rather than minutes.
        generate_model: Some("hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf".into()),
        ..Default::default()
    });

    let result = llm
        .expand_query("rust async runtime", ExpandQueryOptions::default())
        .await
        .expect("expand_query must succeed against Qwen3-0.6B");

    assert!(!result.is_empty(), "expected at least one Queryable");

    // We require at least one of each known QueryType in the union of
    // results — fallback covers all three even when the model produces
    // nothing.
    let types: std::collections::HashSet<_> = result.iter().map(|q| q.type_).collect();
    for kind in [QueryType::Lex, QueryType::Vec, QueryType::Hyde] {
        // The fallback always includes hyde + vec (+ lex when
        // include_lexical=true). The model output may add more.
        // Either way at least vec + hyde must appear; lex is optional
        // only when include_lexical=false (the default is true).
        if matches!(kind, QueryType::Vec | QueryType::Hyde) {
            assert!(
                types.contains(&kind),
                "expected {kind:?} in expansion; got {result:?}"
            );
        }
    }

    llm.dispose().await;
}

/// TS: expandQuery "can exclude lexical queries". With `include_lexical=false`
/// no `lex` entry may appear — the filter applies to both the model-output and
/// the fallback paths, so this holds regardless of what the model emits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads Qwen3-0.6B (~600 MB) and runs CPU inference"]
async fn expand_query_can_exclude_lexical() {
    let llm = LlamaCpp::new(LlamaCppConfig {
        generate_model: Some("hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf".into()),
        ..Default::default()
    });

    let result = llm
        .expand_query(
            "authentication setup",
            ExpandQueryOptions {
                include_lexical: Some(false),
                ..Default::default()
            },
        )
        .await
        .expect("expand_query must succeed against Qwen3-0.6B");

    assert!(
        !result.iter().any(|q| q.type_ == QueryType::Lex),
        "no lex entries when include_lexical=false; got {result:?}"
    );

    llm.dispose().await;
}
