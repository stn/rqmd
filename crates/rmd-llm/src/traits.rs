//! Public async trait for LLM operations.
//!
//! Mirrors `interface LLM` from `tobi/qmd/src/llm.ts` (lines 444–476).
//! Two notable additions over the TS interface:
//!
//! * `tokenize` / `detokenize` / `count_tokens` are part of the trait so
//!   downstream callers (next-PR `Store::chunk_document_by_tokens`) can
//!   use a model's tokenizer through this single API surface, instead
//!   of pulling in `llama-cpp-2` themselves. The TS source exposes
//!   these as separate methods on the `LlamaCpp` class but not on the
//!   `LLM` interface — we promote them.
//! * `count_tokens` has a default impl in terms of `tokenize`. Impls
//!   that can count without a full materialized token vec are free to
//!   override.
//!
//! `LlamaToken` is re-exported from `llama-cpp-2` so callers don't need
//! a direct dep on it. The type is `pub` precisely so `Store` can hold
//! `Vec<LlamaToken>` while building chunk boundaries.

use async_trait::async_trait;

pub use llama_cpp_2::token::LlamaToken;

use crate::error::Result;
use crate::types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    Queryable, RerankDocument, RerankOptions, RerankResult,
};

/// Abstract LLM interface. The only production impl is
/// [`crate::llama_cpp::LlamaCpp`]; tests can substitute a fake.
///
/// All methods are `async` because the LlamaCpp impl uses
/// `tokio::task::spawn_blocking` to wrap synchronous llama-cpp-2 FFI
/// calls. Returning `Option<_>` from `embed` / `generate` mirrors TS,
/// where these methods return `null` on a soft failure rather than
/// throwing (callers want to skip a single bad chunk without aborting
/// the whole batch).
#[async_trait]
pub trait Llm: Send + Sync {
    /// Compute an embedding for one text. Returns `Ok(None)` on a
    /// soft failure (e.g. the worker thread refused the job for a
    /// recoverable reason); `Err(_)` for hard errors (`Disposed`,
    /// `CiDisabled`, model load failure).
    async fn embed(&self, text: &str, opts: EmbedOptions) -> Result<Option<EmbeddingResult>>;

    /// Batch variant — order is preserved, individual failures surface
    /// as `None` slots so callers can skip-and-continue per chunk.
    async fn embed_batch(
        &self,
        texts: &[String],
        opts: EmbedOptions,
    ) -> Result<Vec<Option<EmbeddingResult>>>;

    /// Free-form text generation. `opts.max_tokens` / `temperature` map
    /// to the sampler chain.
    async fn generate(
        &self,
        prompt: &str,
        opts: GenerateOptions,
    ) -> Result<Option<GenerateResult>>;

    /// Existence check for a model URI. HF URIs are assumed to exist
    /// without a network round-trip (matches TS); local paths are
    /// checked against the filesystem.
    async fn model_exists(&self, model: &str) -> Result<ModelInfo>;

    /// Expand a search query into `lex` / `vec` / `hyde` variations.
    /// Returns the fallback set (`[hyde, lex?, vec]`) when the model
    /// produces nothing usable.
    async fn expand_query(
        &self,
        query: &str,
        opts: ExpandQueryOptions,
    ) -> Result<Vec<Queryable>>;

    /// Score `docs` against `query` using the reranker model under
    /// pooling=Rank. Results come back sorted by descending score.
    async fn rerank(
        &self,
        query: &str,
        docs: &[RerankDocument],
        opts: RerankOptions,
    ) -> Result<RerankResult>;

    /// Tokenize a string using the embedding model's tokenizer.
    /// Used by `Store::chunk_document_by_tokens` in the next PR to
    /// split documents on token boundaries rather than chars.
    async fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>>;

    /// Detokenize a slice of token IDs back to text.
    async fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String>;

    /// Count tokens without necessarily building the full token vec.
    /// The default impl materializes the vec; override for efficiency
    /// when the impl can count more cheaply.
    async fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.tokenize(text).await?.len())
    }

    /// Tear down workers and release native resources. Idempotent.
    /// In-flight operations are given up to ~30s to drain before
    /// dispose proceeds (see [`crate::llama_cpp::LlamaCpp::dispose`]).
    async fn dispose(&self);
}
