//! Embedding-input formatting helpers.
//!
//! Mirrors `tobi/qmd/src/llm.ts` lines 73–106. nomic-style embedding
//! models (`embeddinggemma`, …) want a `task: ... | query: ...` /
//! `title: ... | text: ...` template. Qwen3-Embedding wants an
//! `Instruct: ... \n Query: ...` template for queries and raw text for
//! documents.
//!
//! The TS API accepted an optional `modelUri` and fell back to
//! [`crate::llm::config::resolve_embed_model`]. The Rust API takes the model
//! URI explicitly — at every real call site we already know the URI
//! (LlamaCpp stores it on the struct), and forcing callers to be
//! explicit avoids a quiet dependency on global state.

use std::sync::LazyLock;

use regex::Regex;

static QWEN_EMBED_LEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)qwen.*embed").expect("static regex"));
static QWEN_EMBED_TRAILING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)embed.*qwen").expect("static regex"));

/// Returns true when `model_uri` refers to a Qwen3-Embedding-style model.
/// Matches either ordering of "qwen" and "embed" in the URI.
pub fn is_qwen3_embedding_model(model_uri: &str) -> bool {
    QWEN_EMBED_LEADING.is_match(model_uri) || QWEN_EMBED_TRAILING.is_match(model_uri)
}

/// Format a query string for embedding under the given model.
pub fn format_query_for_embedding(query: &str, model_uri: &str) -> String {
    if is_qwen3_embedding_model(model_uri) {
        format!("Instruct: Retrieve relevant documents for the given query\nQuery: {query}")
    } else {
        format!("task: search result | query: {query}")
    }
}

/// Format a document for embedding under the given model.
///
/// `title` is incorporated into the nomic-style template; Qwen3-Embedding
/// uses raw text (optionally prefixed by the title on its own line).
///
/// Empty-string titles are treated the same as `None` to match TS truthy
/// semantics (`title || "none"`). This matters because callers commonly
/// pass `doc.title.as_deref()` and frontmatter sometimes parses a missing
/// title field as `""` rather than `None`.
pub fn format_doc_for_embedding(text: &str, title: Option<&str>, model_uri: &str) -> String {
    let title = title.filter(|t| !t.is_empty());
    if is_qwen3_embedding_model(model_uri) {
        match title {
            Some(t) => format!("{t}\n{text}"),
            None => text.to_owned(),
        }
    } else {
        let title = title.unwrap_or("none");
        format!("title: {title} | text: {text}")
    }
}
