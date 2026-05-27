//! GPU mode and parallelism resolution.
//!
//! `llama-cpp-2` does NOT support runtime backend selection — the backend
//! is decided by Cargo features at build time. So the runtime knob is a
//! two-state `Auto | Off`:
//!
//! * `Off` means "force `n_gpu_layers = 0`" (CPU-only inference, even
//!   when a GPU backend is compiled in).
//! * `Auto` means "use whichever backend was compiled in" — Cargo
//!   features `metal` / `cuda` / `vulkan` on the `rqmd-core` crate
//!   (also re-exported by the `rqmd` CLI).
//!
//! `RQMD_FORCE_CPU` (set to anything other than `false`/`off`/`none`/`0`)
//! forces `Off`. Otherwise `RQMD_LLAMA_GPU` is consulted: unset / empty /
//! `auto` → `Auto`; any off-string → `Off`; the legacy backend names
//! `metal` / `vulkan` / `cuda` map to `Auto` with a warning explaining
//! the compile-time-only restriction; anything else → `Auto` with a
//! stronger warning about the invalid value.

use std::env;

use crate::env_keys;

// Strings that mean "off" for both RQMD_FORCE_CPU and RQMD_LLAMA_GPU.
const OFF_VALUES: &[&str] = &["false", "off", "none", "disable", "disabled", "0"];
const KNOWN_BACKEND_VALUES: &[&str] = &["metal", "vulkan", "cuda"];

/// What llama.cpp should do with GPU offload. The full backend choice is
/// compile-time in `llama-cpp-2`; this enum only controls the runtime
/// on/off switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaGpuMode {
    /// Use whichever backend was compiled in. With no GPU features,
    /// behaves the same as `Off`.
    Auto,
    /// Force `n_gpu_layers = 0` regardless of compile-time features.
    Off,
}

/// Resolve [`LlamaGpuMode`] from the rqmd env vars.
///
/// `RQMD_FORCE_CPU` wins when set to anything OTHER than an off-string
/// (i.e. "1", "true", "yes", "force" all mean "force CPU"); when set to
/// an off-string it is treated as unset so the explicit "off" semantics
/// behave intuitively.
///
/// Then `RQMD_LLAMA_GPU` is consulted: unset / empty / "auto" → `Auto`;
/// any off-string → `Off`; "metal" / "vulkan" / "cuda" → `Auto` with a
/// warning; anything else → `Auto` with a stronger warning about the
/// invalid value.
pub fn resolve_llama_gpu_mode(gpu_env: Option<&str>, force_cpu_env: Option<&str>) -> LlamaGpuMode {
    if let Some(raw) = force_cpu_env {
        let normalized = raw.trim().to_lowercase();
        if !normalized.is_empty() && !OFF_VALUES.contains(&normalized.as_str()) {
            return LlamaGpuMode::Off;
        }
    }

    let raw = match gpu_env {
        Some(v) => v,
        None => return LlamaGpuMode::Auto,
    };
    let normalized = raw.trim().to_lowercase();
    if normalized.is_empty() || normalized == "auto" {
        return LlamaGpuMode::Auto;
    }
    if OFF_VALUES.contains(&normalized.as_str()) {
        return LlamaGpuMode::Off;
    }
    if KNOWN_BACKEND_VALUES.contains(&normalized.as_str()) {
        tracing::warn!(
            "RQMD_LLAMA_GPU=\"{raw}\" requests runtime backend selection, but llama-cpp-2 \
             chooses backends at compile time. Rebuild with `cargo --features \
             rqmd-core/{backend}` instead. Treating this as `auto`.",
            raw = raw,
            backend = normalized,
        );
        return LlamaGpuMode::Auto;
    }
    tracing::warn!(
        "invalid RQMD_LLAMA_GPU=\"{raw}\"; falling back to auto",
        raw = raw,
    );
    LlamaGpuMode::Auto
}

/// Convenience wrapper that reads the env directly.
pub fn resolve_llama_gpu_mode_from_env() -> LlamaGpuMode {
    let gpu = env::var(env_keys::LLAMA_GPU).ok();
    let force = env::var(env_keys::FORCE_CPU).ok();
    resolve_llama_gpu_mode(gpu.as_deref(), force.as_deref())
}

// =============================================================================
// Parallelism
// =============================================================================

/// Parse `RQMD_EMBED_PARALLELISM`. Returns `None` for unset / empty /
/// invalid; clamps valid values to `[1, 8]`.
pub fn resolve_parallelism_override(env_value: Option<&str>) -> Option<usize> {
    let raw = env_value?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n >= 1 => Some(n.min(8)),
        _ => {
            tracing::warn!(
                "invalid RQMD_EMBED_PARALLELISM=\"{raw}\", using automatic parallelism",
                raw = raw,
            );
            None
        }
    }
}

/// Options for [`resolve_safe_parallelism`].
#[derive(Debug, Clone)]
pub struct ParallelismOptions<'a> {
    /// Raw `RQMD_EMBED_PARALLELISM` value, if any.
    pub env_value: Option<&'a str>,
    /// Caller-computed parallelism (e.g. derived from VRAM or CPU cores).
    pub computed: usize,
    /// True when the build targets Windows AND the CUDA backend is
    /// compiled in. The combination is unstable in llama.cpp (see TS
    /// comment lines 543–548), so we serialize to 1 context.
    pub serialize_windows_cuda: bool,
}

/// Resolve the final parallel context count.
///
/// Priority: env override > Windows+CUDA serialization > `max(1, computed)`.
pub fn resolve_safe_parallelism(opts: ParallelismOptions<'_>) -> usize {
    if let Some(n) = resolve_parallelism_override(opts.env_value) {
        return n;
    }
    if opts.serialize_windows_cuda {
        return 1;
    }
    opts.computed.max(1)
}

/// Whether the current build serializes contexts because of the
/// Windows+CUDA llama.cpp bug. PR2 callers should pass the result of
/// this helper to [`ParallelismOptions::serialize_windows_cuda`].
pub const fn windows_cuda_serialization_required() -> bool {
    cfg!(all(target_os = "windows", feature = "cuda"))
}
