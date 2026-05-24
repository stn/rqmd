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
use sha2::{Digest, Sha256};

use crate::store::{CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS};

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

// Probe strings folded into the embedding fingerprint. They never reach a
// model — they only make the query/doc *templates* visible to the hash, so a
// change in `format_query_for_embedding` / `format_doc_for_embedding` (or the
// model branch they pick) flips the fingerprint. Mirror qmd
// `EMBED_FINGERPRINT_PROBE_*` (`store.ts:53–55`).
const EMBED_FINGERPRINT_PROBE_QUERY: &str = "__qmd_embedding_query_probe__";
const EMBED_FINGERPRINT_PROBE_TITLE: &str = "__qmd_embedding_title_probe__";
const EMBED_FINGERPRINT_PROBE_DOC: &str = "__qmd_embedding_document_probe__";

/// Stable 6-hex-char signature over everything that, if changed, invalidates
/// stored embeddings: the model, the query template, the doc template, and the
/// chunking parameters. Stored in `content_vectors.embed_fingerprint`; when it
/// changes, existing vectors are treated as pending and re-embedded.
///
/// Mirrors qmd `getEmbeddingFingerprint` (`store.ts:68–77`) byte-for-byte: the
/// significant lines are joined with `\n`, SHA-256'd, hex-encoded, and sliced to
/// the first 6 hex chars.
pub fn embedding_fingerprint(model: &str) -> String {
    let significant = [
        format!("model:{model}"),
        format!(
            "query:{}",
            format_query_for_embedding(EMBED_FINGERPRINT_PROBE_QUERY, model)
        ),
        format!(
            "doc:{}",
            format_doc_for_embedding(
                EMBED_FINGERPRINT_PROBE_DOC,
                Some(EMBED_FINGERPRINT_PROBE_TITLE),
                model,
            )
        ),
        format!("chunk_tokens:{CHUNK_SIZE_TOKENS}"),
        format!("chunk_overlap_tokens:{CHUNK_OVERLAP_TOKENS}"),
    ]
    .join("\n");

    let mut hasher = Sha256::new();
    hasher.update(significant.as_bytes());
    let digest = hasher.finalize();
    // First 3 bytes = 6 hex chars = qmd's `.digest("hex").slice(0, 6)`.
    let mut s = String::with_capacity(6);
    for b in &digest[..3] {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_fingerprint_is_six_lowercase_hex_and_stable() {
        let fp = embedding_fingerprint("hf:test/embed.gguf");
        assert_eq!(fp.len(), 6);
        assert!(
            fp.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
        // Pure function — two calls agree.
        assert_eq!(fp, embedding_fingerprint("hf:test/embed.gguf"));
    }

    #[test]
    fn embedding_fingerprint_matches_upstream_qmd_golden() {
        // Golden values captured from upstream qmd's `getEmbeddingFingerprint`
        // (`bun -e "...getEmbeddingFingerprint(m)"` against tobi/qmd). This
        // pins byte-for-byte parity: changing a probe constant, a format
        // template, or a chunk constant breaks this test.
        assert_eq!(embedding_fingerprint("hf:test/embed.gguf"), "2846ff");
        assert_eq!(
            embedding_fingerprint("hf:Qwen/Qwen3-Embedding-0.6B.gguf"),
            "8bbf95"
        );
    }

    #[test]
    fn embedding_fingerprint_differs_by_model_and_template_branch() {
        let nomic = embedding_fingerprint("hf:test/embed.gguf");
        let other_nomic = embedding_fingerprint("hf:other/embed.gguf");
        let qwen = embedding_fingerprint("hf:Qwen/Qwen3-Embedding-0.6B.gguf");
        assert_ne!(nomic, other_nomic); // model URI is part of the hash
        assert_ne!(nomic, qwen); // Qwen vs nomic flips the template branch
    }
}
