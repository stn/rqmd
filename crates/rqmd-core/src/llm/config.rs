//! Model URI / cache dir / context size resolution.
//!
//! Mirrors `tobi/qmd/src/llm.ts` lines 241–296 (model resolution),
//! 286–289 (cache dir), and 587–606 (expand context size). Plus the
//! per-context-size env knobs that live next to `LlamaCpp` in TS
//! (lines 1019–1027).

use std::path::PathBuf;

use crate::llm::error::{Error, Result};
use crate::llm::types::ModelResolutionConfig;

// =============================================================================
// Default model URIs
// =============================================================================

/// Default embedding model (`embeddinggemma-300M`, nomic-style).
pub const DEFAULT_EMBED_MODEL: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

/// Default reranker (`Qwen3-Reranker-0.6B`).
pub const DEFAULT_RERANK_MODEL: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

/// Default query-expansion model (fine-tuned 1.7B from upstream qmd author).
pub const DEFAULT_GENERATE_MODEL: &str =
    "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

/// Alternative generate models suitable as a fine-tuning base.
pub const LFM2_GENERATE_MODEL: &str = "hf:LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf";
pub const LFM2_INSTRUCT_MODEL: &str =
    "hf:LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

// =============================================================================
// Context size defaults
// =============================================================================

pub const DEFAULT_EMBED_CONTEXT_SIZE: usize = 2048;
pub const DEFAULT_RERANK_CONTEXT_SIZE: usize = 4096;
pub const DEFAULT_EXPAND_CONTEXT_SIZE: usize = 2048;

/// Default inactivity timeout (5 minutes) before unloading contexts.
pub const DEFAULT_INACTIVITY_TIMEOUT_MS: u64 = 5 * 60 * 1000;

// =============================================================================
// Model URI resolution
// =============================================================================

/// Resolve the embedding model URI. Priority: config arg > `QMD_EMBED_MODEL`
/// env var > [`DEFAULT_EMBED_MODEL`].
pub fn resolve_embed_model(config: Option<&ModelResolutionConfig>) -> String {
    resolve_with_env(
        config.and_then(|c| c.embed.as_deref()),
        "QMD_EMBED_MODEL",
        DEFAULT_EMBED_MODEL,
    )
}

/// Resolve the generation model URI. Priority: config > `QMD_GENERATE_MODEL`
/// > [`DEFAULT_GENERATE_MODEL`].
pub fn resolve_generate_model(config: Option<&ModelResolutionConfig>) -> String {
    resolve_with_env(
        config.and_then(|c| c.generate.as_deref()),
        "QMD_GENERATE_MODEL",
        DEFAULT_GENERATE_MODEL,
    )
}

/// Resolve the reranker model URI. Priority: config > `QMD_RERANK_MODEL`
/// > [`DEFAULT_RERANK_MODEL`].
pub fn resolve_rerank_model(config: Option<&ModelResolutionConfig>) -> String {
    resolve_with_env(
        config.and_then(|c| c.rerank.as_deref()),
        "QMD_RERANK_MODEL",
        DEFAULT_RERANK_MODEL,
    )
}

/// Resolve all three model URIs at once. Mirrors `resolveModels` in TS.
pub fn resolve_models(config: Option<&ModelResolutionConfig>) -> ResolvedModels {
    ResolvedModels {
        embed: resolve_embed_model(config),
        generate: resolve_generate_model(config),
        rerank: resolve_rerank_model(config),
    }
}

/// All three resolved model URIs (TS: `Required<ModelResolutionConfig>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModels {
    pub embed: String,
    pub generate: String,
    pub rerank: String,
}

fn resolve_with_env(config_value: Option<&str>, env_var: &str, default: &str) -> String {
    if let Some(v) = config_value
        && !v.is_empty()
    {
        return v.to_owned();
    }
    if let Ok(v) = std::env::var(env_var)
        && !v.is_empty()
    {
        return v;
    }
    default.to_owned()
}

// =============================================================================
// Model cache directory
// =============================================================================

/// Default model cache directory.
///
/// Priority:
///   1. `XDG_CACHE_HOME/qmd/models`
///   2. `dirs::cache_dir()/qmd/models` (platform-specific user cache)
///   3. `dirs::home_dir()/.cache/qmd/models` (POSIX fallback)
///
/// Returns `None` when no usable cache directory can be determined (very
/// unusual; only happens if both `dirs::cache_dir` and `dirs::home_dir`
/// fail, which on Windows would mean the user has no profile path).
pub fn default_model_cache_dir() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg).join("qmd").join("models"));
    }
    if let Some(cache) = dirs::cache_dir() {
        return Some(cache.join("qmd").join("models"));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("qmd").join("models"))
}

// =============================================================================
// Context size resolution
// =============================================================================

/// Resolve the embed context size from env (`QMD_EMBED_CONTEXT_SIZE`),
/// falling back to [`DEFAULT_EMBED_CONTEXT_SIZE`]. Invalid env values
/// produce a `tracing::warn!` and the default is used.
pub fn resolve_embed_context_size() -> usize {
    resolve_context_size_env("QMD_EMBED_CONTEXT_SIZE", DEFAULT_EMBED_CONTEXT_SIZE)
}

/// Resolve the rerank context size from env (`QMD_RERANK_CONTEXT_SIZE`).
pub fn resolve_rerank_context_size() -> usize {
    resolve_context_size_env("QMD_RERANK_CONTEXT_SIZE", DEFAULT_RERANK_CONTEXT_SIZE)
}

/// Resolve the expand-query context size.
///
/// `config_value` takes priority. Then `QMD_EXPAND_CONTEXT_SIZE` env var.
/// Then [`DEFAULT_EXPAND_CONTEXT_SIZE`].
///
/// Errors when `config_value` is `Some(0)` or otherwise invalid; matches
/// TS which throws on a bad configValue. Invalid env values warn and fall
/// back to default.
pub fn resolve_expand_context_size(config_value: Option<usize>) -> Result<usize> {
    if let Some(v) = config_value {
        if v == 0 {
            return Err(Error::InvalidEnvVar {
                var: "expandContextSize (config)",
                value: v.to_string(),
                reason: "must be a positive integer".into(),
            });
        }
        return Ok(v);
    }
    Ok(resolve_context_size_env(
        "QMD_EXPAND_CONTEXT_SIZE",
        DEFAULT_EXPAND_CONTEXT_SIZE,
    ))
}

fn resolve_context_size_env(var: &'static str, default: usize) -> usize {
    let raw = match std::env::var(var) {
        Ok(v) => v,
        Err(_) => return default,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default;
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n > 0 => n,
        _ => {
            tracing::warn!(
                "invalid {var}=\"{raw}\", using default {default}",
                var = var,
                raw = raw,
                default = default
            );
            default
        }
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Port of qmd's `test/cli.test.ts` "CLI Embed" resolution cases
    //! (`prefers QMD_EMBED_MODEL`, `falls back to the default embed model`).
    //! These are pure resolution unit tests in qmd too — the resolved URI is
    //! never printed by any rqmd command, so they cannot be e2e (process-spawn)
    //! tests here. Extended to all three model slots.

    use super::*;
    use crate::llm::types::ModelResolutionConfig;
    use serial_test::serial;

    /// Model-resolution env vars consulted by `resolve_with_env`. Snapshotted
    /// and restored wholesale so env-mutating tests (all `#[serial]`) cannot
    /// leak state — mirrors the `EnvGuard` pattern in `store::path::tests`.
    /// `set_var`/`remove_var` are `unsafe` since Rust 2024; pair with
    /// `#[serial]` so no other test reads these vars concurrently.
    const ENV_KEYS: &[&str] = &["QMD_EMBED_MODEL", "QMD_GENERATE_MODEL", "QMD_RERANK_MODEL"];

    struct EnvGuard(Vec<(&'static str, Option<String>)>);

    impl EnvGuard {
        fn new() -> Self {
            EnvGuard(
                ENV_KEYS
                    .iter()
                    .map(|k| (*k, std::env::var(k).ok()))
                    .collect(),
            )
        }
        fn set(&self, k: &str, v: &str) {
            unsafe { std::env::set_var(k, v) };
        }
        fn unset(&self, k: &str) {
            unsafe { std::env::remove_var(k) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                unsafe {
                    match v {
                        Some(val) => std::env::set_var(k, val),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    #[test]
    #[serial]
    fn prefers_env_qmd_embed_model_over_default() {
        let g = EnvGuard::new();
        g.set("QMD_EMBED_MODEL", "hf:env/embed-model.gguf");
        assert_eq!(resolve_embed_model(None), "hf:env/embed-model.gguf");
    }

    #[test]
    #[serial]
    fn falls_back_to_default_embed_model_when_unset() {
        let g = EnvGuard::new();
        g.unset("QMD_EMBED_MODEL");
        assert_eq!(resolve_embed_model(None), DEFAULT_EMBED_MODEL);
    }

    #[test]
    #[serial]
    fn config_value_takes_priority_over_env_and_default() {
        let g = EnvGuard::new();
        g.set("QMD_EMBED_MODEL", "hf:env/embed-model.gguf");
        let cfg = ModelResolutionConfig {
            embed: Some("hf:config/embed-model.gguf".to_string()),
            generate: None,
            rerank: None,
        };
        assert_eq!(
            resolve_embed_model(Some(&cfg)),
            "hf:config/embed-model.gguf"
        );
    }

    #[test]
    #[serial]
    fn generate_and_rerank_resolve_env_then_default() {
        let g = EnvGuard::new();
        g.set("QMD_GENERATE_MODEL", "hf:env/generate.gguf");
        g.unset("QMD_RERANK_MODEL");
        assert_eq!(resolve_generate_model(None), "hf:env/generate.gguf");
        assert_eq!(resolve_rerank_model(None), DEFAULT_RERANK_MODEL);
    }
}
