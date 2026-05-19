//! Local GGUF model integration. Rust port of `tobi/qmd`'s `src/llm.ts`.
//!
//! Two layers:
//! - Foundation: error types, shapes, formatting, environment resolution,
//!   GPU mode, singleton, HF download (`error`, `types`, `format`, `config`,
//!   `gpu`, `prompt`, `pull`, `backend`).
//! - Runtime: [`llama_cpp::LlamaCpp`] struct + [`traits::Llm`] trait, worker
//!   pools, [`session::LlmSession`] handle, default singleton instance
//!   (`llama_cpp`, `session`, `singleton`, `traits`, `worker`).
//!
//! The public surface mirrors `llm.ts` exports. Async functions are tokio
//! runtime-bound; callers must already be inside one.

pub mod backend;
pub mod config;
pub mod error;
pub mod format;
pub mod gpu;
pub mod prompt;
pub mod pull;
pub mod types;
pub mod llama_cpp;
pub mod session;
pub mod singleton;
pub mod traits;
pub mod worker;

// Module-level convenience re-exports (mirrors the old `llm.ts` exports).
pub use error::{Error, Result};
