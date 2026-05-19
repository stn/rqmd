//! Integration tests for `rmd_core::llm::format`.

use rmd_core::llm::format::{
    format_doc_for_embedding, format_query_for_embedding, is_qwen3_embedding_model,
};

#[test]
fn detects_qwen3_embedding_models_in_either_order() {
    // Canonical layout: "qwen" before "embed"
    assert!(is_qwen3_embedding_model(
        "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
    ));
    // Reversed: "embed" before "qwen"
    assert!(is_qwen3_embedding_model(
        "hf:someone/embed-qwen-variant-GGUF/embed-qwen-variant.gguf"
    ));
    // Case-insensitive
    assert!(is_qwen3_embedding_model("QWEN-EMBEDDING-1.5"));
}

#[test]
fn rejects_non_qwen_embedding_models() {
    assert!(!is_qwen3_embedding_model(
        "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
    ));
    assert!(!is_qwen3_embedding_model(
        "hf:nomic-ai/nomic-embed-text-v1.5"
    ));
    assert!(!is_qwen3_embedding_model("/local/path/some-model.gguf"));
    assert!(!is_qwen3_embedding_model(""));
    // "qwen" alone is not enough — needs both qwen and embed somewhere
    assert!(!is_qwen3_embedding_model(
        "hf:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B.gguf"
    ));
}

#[test]
fn nomic_query_format_uses_task_search_template() {
    let q = format_query_for_embedding(
        "rust async runtime",
        "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    );
    assert_eq!(q, "task: search result | query: rust async runtime");
}

#[test]
fn qwen3_query_format_uses_instruct_template() {
    let q = format_query_for_embedding(
        "rust async runtime",
        "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
    );
    assert_eq!(
        q,
        "Instruct: Retrieve relevant documents for the given query\nQuery: rust async runtime"
    );
}

#[test]
fn nomic_doc_format_uses_title_text_template() {
    let no_title = format_doc_for_embedding("hello world", None, "hf:foo/embeddinggemma/x.gguf");
    assert_eq!(no_title, "title: none | text: hello world");

    let with_title = format_doc_for_embedding(
        "hello world",
        Some("Greeting"),
        "hf:foo/embeddinggemma/x.gguf",
    );
    assert_eq!(with_title, "title: Greeting | text: hello world");
}

#[test]
fn qwen3_doc_format_emits_raw_text_or_title_plus_text() {
    let no_title = format_doc_for_embedding(
        "hello world",
        None,
        "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/x.gguf",
    );
    assert_eq!(no_title, "hello world");

    let with_title = format_doc_for_embedding(
        "hello world",
        Some("Greeting"),
        "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/x.gguf",
    );
    assert_eq!(with_title, "Greeting\nhello world");
}

#[test]
fn empty_string_title_is_treated_as_no_title() {
    // TS: `title || "none"` / `title ? ... : text` — empty string is falsy.
    // The Rust port treats `Some("")` like `None` for both model styles so
    // callers can pass `doc.title.as_deref()` without normalizing.
    let nomic = "hf:ggml-org/embeddinggemma-300M-GGUF/x.gguf";
    let qwen = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/x.gguf";

    assert_eq!(
        format_doc_for_embedding("body", Some(""), nomic),
        "title: none | text: body",
    );
    assert_eq!(format_doc_for_embedding("body", Some(""), qwen), "body");
}
