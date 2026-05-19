//! Error type for the `rmd_core::llm` module.
//!
//! PR1 contributed the file-and-env variants (`BackendInit`, `HfApi`,
//! `Io`, `InvalidHfUri`, `ModelNotFound`, `InvalidGguf`,
//! `InvalidEnvVar`). PR2 adds lifecycle and llama-cpp-2 wrapping
//! variants used by the core `LlamaCpp` and its `Llm` impl.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `LlamaBackend::init()` failed (e.g. failed to allocate native
    /// resources). `BackendAlreadyInitialized` is handled internally by
    /// [`crate::llm::backend::get`] and never surfaces here.
    #[error("llama backend init failed: {0}")]
    BackendInit(String),

    /// HuggingFace API error during model download or metadata fetch.
    #[error("huggingface api error: {0}")]
    HfApi(String),

    /// Filesystem I/O failure with the path that triggered it.
    #[error("io error at {path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The HF URI (`hf:<repo>/<file>`) is malformed.
    #[error("invalid HuggingFace URI: '{0}'")]
    InvalidHfUri(String),

    /// `pull_models` was given a `model_uri` without `hf:` prefix and the
    /// path did not exist locally. Distinct from `InvalidHfUri` so PR2
    /// callers can offer "did you mean to write `hf:...`?" suggestions.
    #[error("model file not found: '{0}' (no hf: prefix and no local file)")]
    ModelNotFound(String),

    /// A downloaded file is not a valid GGUF model.
    /// `looks_like_html` is true when the file starts with `<!doctype`
    /// or `<html`, which usually means a proxy or captive portal hijacked
    /// the download.
    #[error("not a valid GGUF file at {path}: {reason}", path = path.display())]
    InvalidGguf {
        path: PathBuf,
        reason: String,
        looks_like_html: bool,
    },

    /// An environment variable contained a value that could not be parsed
    /// as the expected type.
    #[error("invalid env var {var}=\"{value}\": {reason}")]
    InvalidEnvVar {
        var: &'static str,
        value: String,
        reason: String,
    },

    // ----- PR2: lifecycle ----------------------------------------------------
    /// LLM operations are disabled because `CI=true` was set at struct
    /// construction (or [`crate::llm::llama_cpp::LlamaCppConfig::ci_mode`] was
    /// `true`). Mirrors `tobi/qmd/src/llm.ts` `_ciMode` guards.
    #[error("LLM operations are disabled in CI (set CI=true at construction)")]
    CiDisabled,

    /// The session has been released (via drop / explicit release / max
    /// duration). Equivalent to TS `SessionReleasedError`.
    #[error("LLM session has been released or aborted: {0}")]
    SessionReleased(String),

    /// `LlamaCpp::dispose` has already run. New operations cannot start.
    #[error("LlamaCpp has been disposed")]
    Disposed,

    /// `LlamaModel::load_from_file` failed. The message includes the
    /// model path or URI for diagnosis.
    #[error("failed to load model '{uri}': {source}")]
    ModelLoad {
        uri: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// `chat_template(None)` returned `MissingTemplate` (the model has no
    /// embedded chat template), or `apply_chat_template` failed.
    #[error("chat template error: {0}")]
    ChatTemplate(String),

    /// `str_to_token` / `token_to_piece` / `detokenize` failed.
    #[error("tokenization error: {0}")]
    Tokenize(String),

    /// Generic wrapper for llama-cpp-2 errors that don't fit a more
    /// specific variant (decode failures, batch overflow, etc.). The
    /// inner string preserves the original Display text.
    #[error("llama backend error: {0}")]
    Llama(String),

    /// A worker thread's mpsc channel was closed before we could send or
    /// receive on it — typically because the worker panicked or the
    /// `LlamaCpp` is mid-dispose.
    #[error("LLM worker channel closed")]
    WorkerClosed,

    /// A spawned blocking task panicked. The `JoinError` carries the
    /// panic payload for downstream logging.
    #[error("blocking task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

pub type Result<T> = std::result::Result<T, Error>;
