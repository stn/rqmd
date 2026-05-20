//! Integration tests for `rqmd_core::llm::gpu`.
//!
//! `resolve_llama_gpu_mode` takes explicit `Option<&str>` arguments (no
//! env reads), so most tests don't need `#[serial]`. The single
//! `from_env` test wraps the env mutation in an `unsafe` block and is
//! marked `#[serial]` to avoid contention with other env-touching tests.

use serial_test::serial;

use rqmd_core::llm::gpu::{
    resolve_llama_gpu_mode, resolve_llama_gpu_mode_from_env, resolve_parallelism_override,
    resolve_safe_parallelism, windows_cuda_serialization_required, LlamaGpuMode,
    ParallelismOptions,
};

// =============================================================================
// GPU mode
// =============================================================================

#[test]
fn gpu_mode_defaults_to_auto_when_unset() {
    assert_eq!(resolve_llama_gpu_mode(None, None), LlamaGpuMode::Auto);
    assert_eq!(resolve_llama_gpu_mode(Some(""), None), LlamaGpuMode::Auto);
    assert_eq!(
        resolve_llama_gpu_mode(Some("auto"), None),
        LlamaGpuMode::Auto
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("  AUTO  "), None),
        LlamaGpuMode::Auto
    );
}

#[test]
fn gpu_mode_recognizes_all_off_variants() {
    for off in &["false", "off", "none", "disable", "disabled", "0"] {
        assert_eq!(
            resolve_llama_gpu_mode(Some(off), None),
            LlamaGpuMode::Off,
            "expected `{off}` to disable GPU",
        );
    }
    // Case-insensitive
    assert_eq!(
        resolve_llama_gpu_mode(Some("FALSE"), None),
        LlamaGpuMode::Off
    );
}

#[test]
fn gpu_mode_runtime_backend_values_warn_but_return_auto() {
    // metal / vulkan / cuda are accepted for env compatibility but treated
    // as Auto (with a tracing::warn the test can't easily observe — we
    // assert the return value only).
    //
    // NOTE: This is an INTENTIONAL divergence from TS. `tobi/qmd`'s
    // `resolveLlamaGpuMode` passes these backends through unchanged
    // (`"metal" -> "metal"`); rqmd's `LlamaGpuMode` is a two-state
    // `{ Off, Auto }` enum (the model-loading path can't consume an explicit
    // backend), so named backends collapse to `Auto`. Do not "fix" this to
    // match TS — keep the behaviors deliberately non-identical.
    assert_eq!(
        resolve_llama_gpu_mode(Some("metal"), None),
        LlamaGpuMode::Auto
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("vulkan"), None),
        LlamaGpuMode::Auto
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("cuda"), None),
        LlamaGpuMode::Auto
    );
}

#[test]
fn gpu_mode_unknown_value_warns_and_returns_auto() {
    assert_eq!(
        resolve_llama_gpu_mode(Some("rocm"), None),
        LlamaGpuMode::Auto
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("yes"), None),
        LlamaGpuMode::Auto
    );
}

#[test]
fn force_cpu_overrides_gpu_setting_when_truthy() {
    // Any non-off value of QMD_FORCE_CPU forces Off regardless of QMD_LLAMA_GPU.
    assert_eq!(
        resolve_llama_gpu_mode(Some("metal"), Some("1")),
        LlamaGpuMode::Off
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("auto"), Some("true")),
        LlamaGpuMode::Off
    );
    assert_eq!(resolve_llama_gpu_mode(None, Some("yes")), LlamaGpuMode::Off);
}

#[test]
fn force_cpu_with_off_value_is_treated_as_unset() {
    // QMD_FORCE_CPU=false means "don't force CPU"; QMD_LLAMA_GPU still wins.
    assert_eq!(
        resolve_llama_gpu_mode(Some("auto"), Some("false")),
        LlamaGpuMode::Auto
    );
    assert_eq!(
        resolve_llama_gpu_mode(Some("off"), Some("0")),
        LlamaGpuMode::Off
    );
}

#[test]
#[serial]
fn from_env_reads_qmd_env_vars() {
    let prev_gpu = std::env::var("QMD_LLAMA_GPU").ok();
    let prev_cpu = std::env::var("QMD_FORCE_CPU").ok();
    unsafe {
        std::env::remove_var("QMD_LLAMA_GPU");
        std::env::remove_var("QMD_FORCE_CPU");
    }
    assert_eq!(resolve_llama_gpu_mode_from_env(), LlamaGpuMode::Auto);

    unsafe {
        std::env::set_var("QMD_LLAMA_GPU", "off");
    }
    assert_eq!(resolve_llama_gpu_mode_from_env(), LlamaGpuMode::Off);

    // restore
    unsafe {
        match prev_gpu {
            Some(v) => std::env::set_var("QMD_LLAMA_GPU", v),
            None => std::env::remove_var("QMD_LLAMA_GPU"),
        }
        match prev_cpu {
            Some(v) => std::env::set_var("QMD_FORCE_CPU", v),
            None => std::env::remove_var("QMD_FORCE_CPU"),
        }
    }
}

// =============================================================================
// Parallelism
// =============================================================================

#[test]
fn parallelism_override_parses_valid_values_and_clamps_at_eight() {
    assert_eq!(resolve_parallelism_override(Some("1")), Some(1));
    assert_eq!(resolve_parallelism_override(Some("4")), Some(4));
    assert_eq!(resolve_parallelism_override(Some("8")), Some(8));
    assert_eq!(resolve_parallelism_override(Some("99")), Some(8));
    assert_eq!(resolve_parallelism_override(Some("  6  ")), Some(6));
}

#[test]
fn parallelism_override_returns_none_for_invalid_or_unset() {
    assert_eq!(resolve_parallelism_override(None), None);
    assert_eq!(resolve_parallelism_override(Some("")), None);
    assert_eq!(resolve_parallelism_override(Some("garbage")), None);
    assert_eq!(resolve_parallelism_override(Some("0")), None);
    assert_eq!(resolve_parallelism_override(Some("-3")), None);
}

#[test]
fn safe_parallelism_env_override_wins() {
    let n = resolve_safe_parallelism(ParallelismOptions {
        env_value: Some("4"),
        computed: 16, // would normally be 16; override forces 4
        serialize_windows_cuda: false,
    });
    assert_eq!(n, 4);
}

#[test]
fn safe_parallelism_windows_cuda_serializes_to_one() {
    let n = resolve_safe_parallelism(ParallelismOptions {
        env_value: None,
        computed: 8,
        serialize_windows_cuda: true,
    });
    assert_eq!(n, 1);
}

#[test]
fn safe_parallelism_clamps_computed_at_one_floor() {
    let n = resolve_safe_parallelism(ParallelismOptions {
        env_value: None,
        computed: 0,
        serialize_windows_cuda: false,
    });
    assert_eq!(n, 1);
}

#[test]
fn safe_parallelism_passes_through_normal_computed() {
    let n = resolve_safe_parallelism(ParallelismOptions {
        env_value: None,
        computed: 4,
        serialize_windows_cuda: false,
    });
    assert_eq!(n, 4);
}

#[test]
fn windows_cuda_serialization_required_matches_compile_features() {
    // Sanity check: this is a const fn, so it's just evaluating
    // cfg!(all(target_os = "windows", feature = "cuda")).
    let expected = cfg!(all(target_os = "windows", feature = "cuda"));
    assert_eq!(windows_cuda_serialization_required(), expected);
}
