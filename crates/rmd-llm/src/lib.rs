//! `rmd-llm` — local GGUF model integration for the `rmd` workspace.
//!
//! Rust port of `tobi/qmd`'s `src/llm.ts`. PR1 (this commit) lands the
//! pure utility layer: error types, public data shapes, embedding /
//! prompt formatting, environment-variable resolution, GPU mode parsing,
//! the process-wide `LlamaBackend` singleton, and HuggingFace download +
//! GGUF validation. PR2 will add the `LlamaCpp` core (worker pool +
//! session + singleton) on top.
//!
//! The public surface mirrors `llm.ts` exports where the names map
//! cleanly to Rust (functions/constants) and uses idiomatic Rust shapes
//! (`Result<T, Error>`, RAII guards, `&str` over `Option<String>`)
//! where the JS API was inherently dynamic.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod backend;
pub mod config;
pub mod error;
pub mod format;
pub mod gpu;
pub mod prompt;
pub mod pull;
pub mod types;

// Convenience re-exports of the most commonly used items.
pub use error::{Error, Result};
pub use types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    ModelResolutionConfig, PullOptions, PullResult, QueryType, Queryable, RerankDocument,
    RerankDocumentResult, RerankOptions, RerankResult, TokenLogProb,
};
