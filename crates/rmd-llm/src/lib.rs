//! `rmd-llm` — local GGUF model integration for the `rmd` workspace.
//!
//! Rust port of `tobi/qmd`'s `src/llm.ts`. The crate is split into
//! two layers:
//!
//! * PR1 utility layer: error types, public data shapes, embedding /
//!   prompt formatting, environment-variable resolution, GPU mode
//!   parsing, the process-wide `LlamaBackend` singleton, and
//!   HuggingFace download + GGUF validation.
//! * PR2 core: the [`LlamaCpp`] struct + [`Llm`] async trait,
//!   dedicated-thread worker pools that isolate
//!   `LlamaContext<'_>`'s `!Send` lifetime, the scoped
//!   [`LlmSession`] handle, and the default-instance singleton.
//!
//! Public surface mirrors `llm.ts` exports where the names map
//! cleanly to Rust (functions/constants) and uses idiomatic Rust
//! shapes (`Result<T, Error>`, RAII guards, `&str` over
//! `Option<String>`) where the JS API was inherently dynamic.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// PR1
pub mod backend;
pub mod config;
pub mod error;
pub mod format;
pub mod gpu;
pub mod prompt;
pub mod pull;
pub mod types;

// PR2
pub mod llama_cpp;
pub mod session;
pub mod singleton;
pub mod traits;
pub mod worker;

// Convenience re-exports of the most commonly used items.
pub use error::{Error, Result};
pub use llama_cpp::{LlamaCpp, LlamaCppConfig};
pub use session::{LlmSession, LlmSessionOptions, with_llm_session};
pub use singleton::{default_llama_cpp, dispose_default_llama_cpp, set_default_llama_cpp};
pub use traits::{LlamaToken, Llm};
pub use types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    ModelResolutionConfig, PullOptions, PullResult, QueryType, Queryable, RerankDocument,
    RerankDocumentResult, RerankOptions, RerankResult, TokenLogProb,
};
