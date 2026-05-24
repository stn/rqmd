//! Native llama.cpp / ggml backend device enumeration for `rqmd doctor`.
//!
//! `llama-cpp-2` already exposes a safe wrapper over the `ggml_backend_dev_*`
//! FFI (`list_llama_ggml_backend_devices`), so no `unsafe` is needed here. The
//! only requirement is that the backend registry is initialized first, which
//! [`crate::llm::backend::try_get`] guarantees (`LlamaBackend::init()` once per
//! process). On a default build (no `metal`/`cuda`/`vulkan` feature) only CPU
//! devices are registered — that is expected, not an error.

pub use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType};

use crate::llm::error::Result;

/// Enumerate the available ggml backend devices.
///
/// Initializes the shared `LlamaBackend` first (required for device
/// registration); propagates [`crate::llm::error::Error::BackendInit`] if that
/// fails so the caller (`doctor`) can report a probe failure instead of
/// aborting.
pub fn probe_devices() -> Result<Vec<LlamaBackendDevice>> {
    crate::llm::backend::try_get()?;
    Ok(llama_cpp_2::list_llama_ggml_backend_devices())
}
