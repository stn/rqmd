//! Process-wide `LlamaBackend` accessor.
//!
//! `LlamaBackend::init()` is one-shot per process — calling it a second
//! time returns `LlamaCppError::BackendAlreadyInitialized` (verified by
//! `examples/spike_01_backend.rs`). The backend must therefore live in
//! a static, and every part of the `llm` module that needs a `&LlamaBackend`
//! goes through [`get`].
//!
//! The static is initialized lazily on first call. Initialization
//! failures other than `BackendAlreadyInitialized` propagate from
//! [`try_get`]; [`get`] panics on those, which is fine because they
//! indicate a fatal environment problem (no GPU drivers, etc.) that the
//! caller cannot recover from.

use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::llm::error::{Error, Result};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Get the shared `LlamaBackend`, initializing it on the first call.
/// Panics if initialization fails (see [`try_get`] for a fallible version).
pub fn get() -> &'static LlamaBackend {
    try_get().expect("LlamaBackend::init() must succeed once per process")
}

/// Fallible variant of [`get`]. Returns the cached backend if already
/// initialized; otherwise tries to initialize and caches the result.
pub fn try_get() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    match LlamaBackend::init() {
        Ok(b) => Ok(BACKEND.get_or_init(|| b)),
        Err(e) => {
            // BackendAlreadyInitialized only happens when another path
            // (e.g. an upstream example or a test using llama-cpp-2
            // directly) raced us to init. In that case the static may
            // not be populated, but the backend exists at the C layer.
            // We can't recover a `&'static LlamaBackend` for it, so
            // surface a clear error.
            Err(Error::BackendInit(e.to_string()))
        }
    }
}
