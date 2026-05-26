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
// Expand-query prompt + sampling defaults
// =============================================================================

/// Default prefix prepended to the `expand_query` user message. `/no_think`
/// is a Qwen3 control token that suppresses the chain-of-thought block —
/// matches qmd's hardcoded behaviour (`src/llm.ts:1449`). Non-Qwen finetunes
/// should override to `""`.
pub const DEFAULT_EXPAND_USER_MESSAGE_PREFIX: &str = "/no_think";

/// Default HyDE template used when `expand_query` produces no usable output.
/// `{query}` is substituted with the original query.
pub const DEFAULT_EXPAND_FALLBACK_HYDE_TEMPLATE: &str = "Information about {query}";

/// Default sampler temperature for `expand_query`. Mirrors qmd's hardcoded
/// Qwen3-1.7B tuning.
pub const DEFAULT_EXPAND_TEMP: f32 = 0.7;

/// Default sampler `top_k` for `expand_query`.
pub const DEFAULT_EXPAND_TOP_K: i32 = 20;

/// Default sampler `top_p` for `expand_query`.
pub const DEFAULT_EXPAND_TOP_P: f32 = 0.8;

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

// =============================================================================
// Expand-query prompt + sampling resolution
// =============================================================================

/// Resolve the `expand_query` user-message prefix. Priority:
/// YAML config > `QMD_EXPAND_USER_MESSAGE_PREFIX` env > [`DEFAULT_EXPAND_USER_MESSAGE_PREFIX`].
/// `Some("")` from either YAML or env is honoured as an explicit empty prefix.
pub fn resolve_expand_user_message_prefix(config: Option<&str>) -> String {
    resolve_with_env_allow_empty(
        config,
        "QMD_EXPAND_USER_MESSAGE_PREFIX",
        Some(DEFAULT_EXPAND_USER_MESSAGE_PREFIX),
    )
    .expect("default is Some, result must be Some")
}

/// Resolve the optional `expand_query` system message. Priority:
/// YAML config > `QMD_EXPAND_SYSTEM_MESSAGE` env > `None`.
/// `Some("")` is honoured as an explicit empty system message (suppresses
/// Llama-3 / similar templates' default system prompt).
pub fn resolve_expand_system_message(config: Option<&str>) -> Option<String> {
    resolve_with_env_allow_empty(config, "QMD_EXPAND_SYSTEM_MESSAGE", None)
}

/// Resolve the HyDE fallback template. Priority:
/// YAML config > `QMD_EXPAND_FALLBACK_HYDE_TEMPLATE` env > [`DEFAULT_EXPAND_FALLBACK_HYDE_TEMPLATE`].
pub fn resolve_expand_fallback_hyde_template(config: Option<&str>) -> String {
    resolve_with_env_allow_empty(
        config,
        "QMD_EXPAND_FALLBACK_HYDE_TEMPLATE",
        Some(DEFAULT_EXPAND_FALLBACK_HYDE_TEMPLATE),
    )
    .expect("default is Some, result must be Some")
}

/// Resolve the `expand_query` sampler temperature. Priority:
/// YAML config > `QMD_EXPAND_TEMP` env > [`DEFAULT_EXPAND_TEMP`].
pub fn resolve_expand_temp(config: Option<f32>) -> f32 {
    resolve_numeric_env(config, "QMD_EXPAND_TEMP", DEFAULT_EXPAND_TEMP)
}

/// Resolve the `expand_query` sampler `top_k`. Priority:
/// YAML config > `QMD_EXPAND_TOP_K` env > [`DEFAULT_EXPAND_TOP_K`].
pub fn resolve_expand_top_k(config: Option<i32>) -> i32 {
    resolve_numeric_env(config, "QMD_EXPAND_TOP_K", DEFAULT_EXPAND_TOP_K)
}

/// Resolve the `expand_query` sampler `top_p`. Priority:
/// YAML config > `QMD_EXPAND_TOP_P` env > [`DEFAULT_EXPAND_TOP_P`].
pub fn resolve_expand_top_p(config: Option<f32>) -> f32 {
    resolve_numeric_env(config, "QMD_EXPAND_TOP_P", DEFAULT_EXPAND_TOP_P)
}

/// Generic numeric env-var resolver: YAML config > env (parsed) > default.
/// Invalid env values produce a `tracing::warn!` and fall back to default
/// (mirrors [`resolve_context_size_env`]). Empty string env = unset (parse fails).
fn resolve_numeric_env<T>(config: Option<T>, var: &'static str, default: T) -> T
where
    T: std::str::FromStr + Copy,
{
    if let Some(v) = config {
        return v;
    }
    let raw = match std::env::var(var) {
        Ok(v) => v,
        Err(_) => return default,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default;
    }
    match trimmed.parse::<T>() {
        Ok(n) => n,
        Err(_) => {
            tracing::warn!(
                "invalid {var}=\"{raw}\", using default",
                var = var,
                raw = raw,
            );
            default
        }
    }
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

/// Like [`resolve_with_env`] but **treats `Some("")` and `env_var=""` as
/// meaningful explicit empty values**. Only falls through to `default` when
/// `config_value` is `None` AND the env var is unset. `default = None`
/// returns `None`; `default = Some(...)` returns `Some(...)`.
///
/// Used by `expand_query` config knobs where the user must be able to set
/// `user_message_prefix: ""` to suppress the default `/no_think` prefix.
fn resolve_with_env_allow_empty(
    config_value: Option<&str>,
    env_var: &str,
    default: Option<&str>,
) -> Option<String> {
    if let Some(v) = config_value {
        return Some(v.to_owned());
    }
    if let Ok(v) = std::env::var(env_var) {
        return Some(v);
    }
    default.map(str::to_owned)
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
            ..Default::default()
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

    // -----------------------------------------------------------------
    // Expand-query knob resolvers (prefix, system_message, fallback, sampling)
    // -----------------------------------------------------------------

    /// Env vars touched by the `expand_*` resolvers. Snapshotted and
    /// restored per test like the URI guard above.
    const EXPAND_ENV_KEYS: &[&str] = &[
        "QMD_EXPAND_USER_MESSAGE_PREFIX",
        "QMD_EXPAND_SYSTEM_MESSAGE",
        "QMD_EXPAND_FALLBACK_HYDE_TEMPLATE",
        "QMD_EXPAND_TEMP",
        "QMD_EXPAND_TOP_K",
        "QMD_EXPAND_TOP_P",
    ];

    struct ExpandEnvGuard(Vec<(&'static str, Option<String>)>);

    impl ExpandEnvGuard {
        fn new() -> Self {
            ExpandEnvGuard(
                EXPAND_ENV_KEYS
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
        fn unset_all(&self) {
            for k in EXPAND_ENV_KEYS {
                self.unset(k);
            }
        }
    }

    impl Drop for ExpandEnvGuard {
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
    fn expand_user_message_prefix_falls_back_to_default() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        assert_eq!(
            resolve_expand_user_message_prefix(None),
            DEFAULT_EXPAND_USER_MESSAGE_PREFIX,
        );
    }

    #[test]
    #[serial]
    fn expand_user_message_prefix_env_overrides_default() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_USER_MESSAGE_PREFIX", "### Task:");
        assert_eq!(resolve_expand_user_message_prefix(None), "### Task:");
    }

    #[test]
    #[serial]
    fn expand_user_message_prefix_empty_env_is_honoured() {
        // This is the load-bearing case for Llama Swallow / non-Qwen users:
        // `QMD_EXPAND_USER_MESSAGE_PREFIX=""` must produce an empty prefix,
        // not fall through to the `/no_think` default.
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_USER_MESSAGE_PREFIX", "");
        assert_eq!(resolve_expand_user_message_prefix(None), "");
    }

    #[test]
    #[serial]
    fn expand_user_message_prefix_config_beats_env() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_USER_MESSAGE_PREFIX", "from-env");
        assert_eq!(
            resolve_expand_user_message_prefix(Some("from-config")),
            "from-config",
        );
    }

    #[test]
    #[serial]
    fn expand_user_message_prefix_config_empty_is_honoured() {
        // YAML side: `expand.user_message_prefix: ""` must produce an empty
        // prefix and not fall through.
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_USER_MESSAGE_PREFIX", "from-env");
        assert_eq!(resolve_expand_user_message_prefix(Some("")), "");
    }

    #[test]
    #[serial]
    fn expand_system_message_default_is_none() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        assert_eq!(resolve_expand_system_message(None), None);
    }

    #[test]
    #[serial]
    fn expand_system_message_env_empty_is_some_empty() {
        // Critical: env=""` is meaningful — it suppresses Llama-3 default
        // system prompt — and must round-trip to `Some(String::new())`,
        // not `None`.
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_SYSTEM_MESSAGE", "");
        assert_eq!(resolve_expand_system_message(None), Some(String::new()),);
    }

    #[test]
    #[serial]
    fn expand_system_message_config_overrides_env() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_SYSTEM_MESSAGE", "from-env");
        assert_eq!(
            resolve_expand_system_message(Some("from-config")),
            Some("from-config".to_string()),
        );
    }

    #[test]
    #[serial]
    fn expand_fallback_hyde_template_default_is_english() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        assert_eq!(
            resolve_expand_fallback_hyde_template(None),
            DEFAULT_EXPAND_FALLBACK_HYDE_TEMPLATE,
        );
    }

    #[test]
    #[serial]
    fn expand_fallback_hyde_template_japanese_via_env() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_FALLBACK_HYDE_TEMPLATE", "{query}に関する情報");
        assert_eq!(
            resolve_expand_fallback_hyde_template(None),
            "{query}に関する情報",
        );
    }

    #[test]
    #[serial]
    fn expand_temp_default_when_unset() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        assert!((resolve_expand_temp(None) - DEFAULT_EXPAND_TEMP).abs() < f32::EPSILON);
    }

    #[test]
    #[serial]
    fn expand_temp_env_overrides_default() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_TEMP", "0.42");
        assert!((resolve_expand_temp(None) - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    #[serial]
    fn expand_temp_invalid_env_warns_and_falls_back() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_TEMP", "not-a-number");
        assert!((resolve_expand_temp(None) - DEFAULT_EXPAND_TEMP).abs() < f32::EPSILON);
    }

    #[test]
    #[serial]
    fn expand_top_k_config_beats_env() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_TOP_K", "50");
        assert_eq!(resolve_expand_top_k(Some(99)), 99);
    }

    #[test]
    #[serial]
    fn expand_top_p_env_overrides_default() {
        let g = ExpandEnvGuard::new();
        g.unset_all();
        g.set("QMD_EXPAND_TOP_P", "0.95");
        assert!((resolve_expand_top_p(None) - 0.95).abs() < f32::EPSILON);
    }
}
