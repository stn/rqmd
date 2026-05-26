//! Integration tests for `rqmd_core::llm::prompt`.

use rqmd_core::llm::prompt::{
    QWEN3_RERANKER_INSTRUCT, build_expand_query_user_message, build_qwen3_rerank_prompt,
    fallback_queryables, filter_with_query_terms, parse_expand_query_output,
};
use rqmd_core::llm::types::{QueryType, Queryable};

// =============================================================================
// Qwen3-Reranker prompt
// =============================================================================

#[test]
fn rerank_prompt_includes_all_required_landmarks() {
    let prompt = build_qwen3_rerank_prompt("rust async runtime", "Tokio is an async runtime.");

    // Both the query and the document must appear verbatim.
    assert!(
        prompt.contains("rust async runtime"),
        "query missing from prompt"
    );
    assert!(
        prompt.contains("Tokio is an async runtime."),
        "document missing from prompt"
    );

    // Chat-template brackets and the canonical instruction.
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("<|im_start|>assistant"));
    assert!(prompt.contains("<|im_end|>"));
    assert!(prompt.contains(QWEN3_RERANKER_INSTRUCT));

    // The empty `<think>` block prevents reasoning from being part of the
    // pooled rank output. Without it, Qwen3 emits chain-of-thought tokens
    // that change the pooling result. (Spike #3 finding.)
    assert!(prompt.contains("<think>\n\n</think>"));
}

// =============================================================================
// expand_query helpers
// =============================================================================

#[test]
fn expand_query_user_message_without_intent() {
    let msg = build_expand_query_user_message("rust async runtime", None, "/no_think");
    assert_eq!(
        msg,
        "/no_think Expand this search query: rust async runtime"
    );
}

#[test]
fn expand_query_user_message_with_intent() {
    let msg = build_expand_query_user_message(
        "rust async runtime",
        Some("performance comparison"),
        "/no_think",
    );
    assert_eq!(
        msg,
        "/no_think Expand this search query: rust async runtime\nQuery intent: performance comparison"
    );
}

#[test]
fn parse_expand_query_output_picks_up_three_prefixes() {
    let raw = "lex: tokio runtime\nvec: rust async runtime alternatives\nhyde: \
               Tokio is an asynchronous runtime for Rust.\n";
    let parsed = parse_expand_query_output(raw);
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].type_, QueryType::Lex);
    assert_eq!(parsed[0].text, "tokio runtime");
    assert_eq!(parsed[1].type_, QueryType::Vec);
    assert_eq!(parsed[2].type_, QueryType::Hyde);
}

#[test]
fn parse_expand_query_output_skips_think_block_and_noise() {
    // Real spike #4 output started with `<think>\n\n</think>` and blank
    // lines before the actual three lines.
    let raw = "<think>\n\n</think>\n\nlex: tokio\nvec: rust async\nhyde: Tokio docs\n";
    let parsed = parse_expand_query_output(raw);
    assert_eq!(parsed.len(), 3);
    assert_eq!(
        parsed.iter().map(|q| q.type_).collect::<Vec<_>>(),
        vec![QueryType::Lex, QueryType::Vec, QueryType::Hyde]
    );
}

#[test]
fn parse_expand_query_output_accepts_loose_spacing_after_colon() {
    // Spike #4 also produced `lex:tokio` (no space) under some samples.
    // We accept any whitespace (trim_start_matches type prefix, trim rest).
    let raw = "lex:tokio\nvec:   rust async\nhyde:Tokio\n";
    let parsed = parse_expand_query_output(raw);
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0].text, "tokio");
    assert_eq!(parsed[1].text, "rust async");
    assert_eq!(parsed[2].text, "Tokio");
}

#[test]
fn parse_expand_query_output_returns_empty_when_no_lines_match() {
    let raw = "I'm not going to do that, Dave.";
    assert!(parse_expand_query_output(raw).is_empty());
}

#[test]
fn filter_with_query_terms_keeps_only_on_topic_lines() {
    let queryables = vec![
        Queryable {
            type_: QueryType::Lex,
            text: "tokio runtime".into(),
        },
        Queryable {
            type_: QueryType::Vec,
            text: "ruby on rails".into(),
        },
        Queryable {
            type_: QueryType::Hyde,
            text: "An async runtime for Rust".into(),
        },
    ];
    let filtered = filter_with_query_terms("rust async runtime", queryables);
    // "ruby on rails" gets filtered because it contains none of "rust", "async", "runtime".
    let texts: Vec<_> = filtered.iter().map(|q| q.text.as_str()).collect();
    assert!(
        texts.contains(&"tokio runtime"),
        "missing matching line: {texts:?}"
    );
    assert!(texts.contains(&"An async runtime for Rust"));
    assert!(!texts.contains(&"ruby on rails"));
}

#[test]
fn filter_with_query_terms_passes_everything_when_query_is_empty() {
    let queryables = vec![Queryable {
        type_: QueryType::Lex,
        text: "anything".into(),
    }];
    let filtered = filter_with_query_terms("", queryables);
    assert_eq!(filtered.len(), 1);
}

// =============================================================================
// Fallback queryables
// =============================================================================

#[test]
fn fallback_includes_hyde_lex_vec_when_lexical_enabled() {
    let fallback = fallback_queryables("hello world", true, "Information about {query}");
    let types: Vec<_> = fallback.iter().map(|q| q.type_).collect();
    assert_eq!(types, vec![QueryType::Hyde, QueryType::Lex, QueryType::Vec]);
    assert_eq!(fallback[0].text, "Information about hello world");
    assert_eq!(fallback[1].text, "hello world");
    assert_eq!(fallback[2].text, "hello world");
}

#[test]
fn fallback_drops_lex_when_lexical_disabled() {
    let fallback = fallback_queryables("hello world", false, "Information about {query}");
    let types: Vec<_> = fallback.iter().map(|q| q.type_).collect();
    assert_eq!(types, vec![QueryType::Hyde, QueryType::Vec]);
}
