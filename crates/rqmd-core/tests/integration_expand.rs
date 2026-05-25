//! Real-GGUF expand_query test for `LlamaCpp`.
//!
//! Run with `cargo test -p rqmd-core --features cuda -- --ignored integration_expand`.
//! Exercises the GBNF-constrained expansion path: the model output is forced
//! to one-or-more `lex:` / `vec:` / `hyde:` lines via `EXPAND_QUERY_GRAMMAR`,
//! applied through `sample_token`'s grammar-first flow. Uses the default
//! fine-tuned `qmd-query-expansion-1.7B` model — under the grammar a generic
//! base model is no longer a valid stand-in: it cannot pick `vec`/`hyde` and
//! never emits EOG, so `line+` runs to the token cap. The fine-tuned model is
//! trained to emit the three typed lines and stop, which the grammar permits at
//! each line boundary.

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{ExpandQueryOptions, QueryType};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads qmd-query-expansion-1.7B (~1.1 GB) and runs inference"]
async fn expand_query_returns_at_least_one_parseable_line() {
    let llm = LlamaCpp::new(LlamaCppConfig::default());

    let result = llm
        .expand_query("rust async runtime", ExpandQueryOptions::default())
        .await
        .expect("expand_query must succeed against the fine-tuned model");

    assert!(!result.is_empty(), "expected at least one Queryable");

    // The fine-tuned model, constrained by the grammar, emits typed lines
    // directly (no `<think>` preamble). At minimum vec + hyde must appear;
    // lex is optional only when include_lexical=false (default is true).
    let types: std::collections::HashSet<_> = result.iter().map(|q| q.type_).collect();
    for kind in [QueryType::Vec, QueryType::Hyde] {
        assert!(
            types.contains(&kind),
            "expected {kind:?} in expansion; got {result:?}"
        );
    }

    llm.dispose().await;
}

/// TS: expandQuery "can exclude lexical queries". With `include_lexical=false`
/// no `lex` entry may appear — the filter applies to both the model-output and
/// the fallback paths, so this holds regardless of what the model emits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads qmd-query-expansion-1.7B (~1.1 GB) and runs inference"]
async fn expand_query_can_exclude_lexical() {
    let llm = LlamaCpp::new(LlamaCppConfig::default());

    let result = llm
        .expand_query(
            "authentication setup",
            ExpandQueryOptions {
                include_lexical: Some(false),
                ..Default::default()
            },
        )
        .await
        .expect("expand_query must succeed against the fine-tuned model");

    assert!(
        !result.iter().any(|q| q.type_ == QueryType::Lex),
        "no lex entries when include_lexical=false; got {result:?}"
    );

    llm.dispose().await;
}
