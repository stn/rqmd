//! Error type for `rmd-llm`.
//!
//! Only the variants used by PR1 modules (`backend`, `config`, `gpu`,
//! `pull`, `prompt`) are declared here. PR2 will append variants for
//! session/lifecycle errors (`SessionReleased`, `CiDisabled`, `Join`,
//! `ModelLoad`, `ChatTemplate`, generic `Llama`).

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `LlamaBackend::init()` failed (e.g. failed to allocate native
    /// resources). `BackendAlreadyInitialized` is handled internally by
    /// [`crate::backend::get`] and never surfaces here.
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
}

pub type Result<T> = std::result::Result<T, Error>;
