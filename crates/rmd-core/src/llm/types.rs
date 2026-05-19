//! Public data types exchanged with `Llm` implementations.
//!
//! Mirrors the type declarations near the top of `tobi/qmd/src/llm.ts`
//! (lines 109–235). These are plain data structures with no behavior;
//! anything needing a `LlamaModel` / `LlamaContext` lives in PR2 modules
//! (`worker`, `llama_cpp`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// =============================================================================
// Embedding / generation / reranking results
// =============================================================================

/// Token with log probability (`TokenLogProb` in TS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    pub token: String,
    pub logprob: f64,
}

/// Result of an `embed` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub model: String,
}

/// Result of a `generate` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResult {
    pub text: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TokenLogProb>>,
    pub done: bool,
}

/// Per-document score from `rerank`. `index` refers back to the position
/// in the input slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankDocumentResult {
    pub file: String,
    pub score: f32,
    pub index: usize,
}

/// Aggregated rerank output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub results: Vec<RerankDocumentResult>,
    pub model: String,
}

/// Status of a model file. `path` is set only for local-filesystem models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

// =============================================================================
// Call options
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Override the embedding model URI for this call.
    pub model: Option<String>,
    /// True when the text is a search query (uses query-style formatting).
    /// False (default) treats the text as a document.
    pub is_query: bool,
    /// Optional document title (used by nomic-style formatting only).
    pub title: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct RerankOptions {
    pub model: Option<String>,
}

/// Options for `expand_query`. `include_lexical` defaults to true when None.
#[derive(Debug, Clone, Default)]
pub struct ExpandQueryOptions {
    pub context: Option<String>,
    pub include_lexical: Option<bool>,
    pub intent: Option<String>,
}

// =============================================================================
// Query expansion
// =============================================================================

/// Which search backend a `Queryable` targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryType {
    Lex,
    Vec,
    Hyde,
}

impl QueryType {
    /// String representation matching the TS literal (`"lex" | "vec" | "hyde"`).
    pub fn as_str(self) -> &'static str {
        match self {
            QueryType::Lex => "lex",
            QueryType::Vec => "vec",
            QueryType::Hyde => "hyde",
        }
    }
}

/// One query variation produced by `expand_query`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Queryable {
    #[serde(rename = "type")]
    pub type_: QueryType,
    pub text: String,
}

// =============================================================================
// Rerank input
// =============================================================================

/// One document for `rerank`. `file` is an opaque identifier (typically a
/// virtual path) that flows through to `RerankDocumentResult.file`.
#[derive(Debug, Clone)]
pub struct RerankDocument {
    pub file: String,
    pub text: String,
    pub title: Option<String>,
}

// =============================================================================
// Model resolution / pulling
// =============================================================================

/// Override individual model URIs (otherwise env vars / defaults apply).
#[derive(Debug, Clone, Default)]
pub struct ModelResolutionConfig {
    pub embed: Option<String>,
    pub generate: Option<String>,
    pub rerank: Option<String>,
}

/// Options for [`crate::llm::pull::pull_models`].
#[derive(Debug, Clone, Default)]
pub struct PullOptions {
    /// Force re-download even if a cached copy exists.
    pub refresh: bool,
    /// Override the cache directory. None = use [`crate::llm::config::default_model_cache_dir`].
    pub cache_dir: Option<PathBuf>,
}

/// Per-model result from [`crate::llm::pull::pull_models`].
#[derive(Debug, Clone)]
pub struct PullResult {
    pub model: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    /// True when this call removed a previously cached snapshot and
    /// re-downloaded. False when [`PullOptions::refresh`] was off, or
    /// when `refresh` was on but nothing was actually cached yet.
    pub refreshed: bool,
}
