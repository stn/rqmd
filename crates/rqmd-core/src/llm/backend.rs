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

use std::sync::{Once, OnceLock};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};

use crate::llm::error::{Error, Result};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Guards the one-shot install of the native log callback.
static LOG_REDIRECT: Once = Once::new();

/// Route llama.cpp + ggml native C logs into the `tracing` ecosystem.
///
/// Without this, llama.cpp logs via a default callback that writes straight
/// to the C `stderr` (`llama_context:`, `sched_reserve:`, `decode: ...`).
/// That bypasses both Rust's `print!` capture and `tracing`, so the noise
/// shows up even under `cargo test`. Sending it to `tracing` instead means
/// the output obeys the process log level — and stays silent in tests, which
/// install no subscriber.
///
/// Idempotent and safe to call before `LlamaBackend::init()` (it only sets
/// global C function pointers), so we install it as early as possible to also
/// capture backend-init chatter.
fn install_log_redirect() {
    LOG_REDIRECT.call_once(|| {
        send_logs_to_tracing(LogOptions::default());
    });
}

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
    // Install before init so backend-enumeration logs are captured too.
    install_log_redirect();
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
