//! Integration tests for `rqmd_core::llm::config`.
//!
//! Tests that read or mutate `RQMD_*` environment variables are marked
//! `#[serial]` to avoid interleaving with each other (and with any
//! other env-touching test in the crate). Rust 2024 made
//! `std::env::set_var` `unsafe`, so the few mutating tests use an
//! `unsafe` block deliberately, scoped to the test body.

use serial_test::serial;

use rqmd_core::env_keys;
use rqmd_core::llm::config::{
    DEFAULT_EMBED_CONTEXT_SIZE, DEFAULT_EMBED_MODEL, DEFAULT_EXPAND_CONTEXT_SIZE,
    DEFAULT_GENERATE_MODEL, DEFAULT_RERANK_CONTEXT_SIZE, DEFAULT_RERANK_MODEL,
    resolve_embed_context_size, resolve_embed_model, resolve_expand_context_size,
    resolve_generate_model, resolve_models, resolve_rerank_context_size, resolve_rerank_model,
};
use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::types::ModelResolutionConfig;

const EMBED_ENV: &str = env_keys::EMBED_MODEL;
const GENERATE_ENV: &str = env_keys::GENERATE_MODEL;
const RERANK_ENV: &str = env_keys::RERANK_MODEL;
const EMBED_CTX_ENV: &str = env_keys::EMBED_CONTEXT_SIZE;
const RERANK_CTX_ENV: &str = env_keys::RERANK_CONTEXT_SIZE;
const EXPAND_CTX_ENV: &str = env_keys::EXPAND_CONTEXT_SIZE;

/// Run `f` with `var` removed from the environment, then restore it.
fn with_unset<F: FnOnce()>(var: &str, f: F) {
    let prev = std::env::var(var).ok();
    unsafe {
        std::env::remove_var(var);
    }
    f();
    unsafe {
        match prev {
            Some(v) => std::env::set_var(var, v),
            None => std::env::remove_var(var),
        }
    }
}

/// Run `f` with `var` set to `value`, then restore the previous value.
fn with_set<F: FnOnce()>(var: &str, value: &str, f: F) {
    let prev = std::env::var(var).ok();
    unsafe {
        std::env::set_var(var, value);
    }
    f();
    unsafe {
        match prev {
            Some(v) => std::env::set_var(var, v),
            None => std::env::remove_var(var),
        }
    }
}

// =============================================================================
// Model URI resolution
// =============================================================================

#[test]
#[serial]
fn resolve_embed_model_falls_back_to_default() {
    with_unset(EMBED_ENV, || {
        assert_eq!(resolve_embed_model(None), DEFAULT_EMBED_MODEL);
    });
}

#[test]
#[serial]
fn resolve_embed_model_honors_env_var() {
    with_set(EMBED_ENV, "hf:custom/embed/file.gguf", || {
        assert_eq!(resolve_embed_model(None), "hf:custom/embed/file.gguf");
    });
}

#[test]
#[serial]
fn resolve_embed_model_config_wins_over_env_and_default() {
    with_set(EMBED_ENV, "hf:env/embed/file.gguf", || {
        let config = ModelResolutionConfig {
            embed: Some("hf:config/embed/file.gguf".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_embed_model(Some(&config)),
            "hf:config/embed/file.gguf"
        );
    });
}

#[test]
#[serial]
fn resolve_embed_model_treats_empty_strings_as_unset() {
    with_set(EMBED_ENV, "", || {
        let config = ModelResolutionConfig {
            embed: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(resolve_embed_model(Some(&config)), DEFAULT_EMBED_MODEL);
    });
}

#[test]
#[serial]
fn resolve_generate_and_rerank_work_the_same_way() {
    with_unset(GENERATE_ENV, || {
        assert_eq!(resolve_generate_model(None), DEFAULT_GENERATE_MODEL);
    });
    with_unset(RERANK_ENV, || {
        assert_eq!(resolve_rerank_model(None), DEFAULT_RERANK_MODEL);
    });
}

#[test]
#[serial]
fn resolve_models_returns_all_three_uris() {
    with_unset(EMBED_ENV, || {
        with_unset(GENERATE_ENV, || {
            with_unset(RERANK_ENV, || {
                let resolved = resolve_models(None);
                assert_eq!(resolved.embed, DEFAULT_EMBED_MODEL);
                assert_eq!(resolved.generate, DEFAULT_GENERATE_MODEL);
                assert_eq!(resolved.rerank, DEFAULT_RERANK_MODEL);
            });
        });
    });
}

// =============================================================================
// Context size resolution
// =============================================================================

#[test]
#[serial]
fn resolve_embed_context_size_defaults_when_unset() {
    with_unset(EMBED_CTX_ENV, || {
        assert_eq!(resolve_embed_context_size(), DEFAULT_EMBED_CONTEXT_SIZE);
    });
}

#[test]
#[serial]
fn resolve_rerank_context_size_defaults_when_unset() {
    with_unset(RERANK_CTX_ENV, || {
        assert_eq!(resolve_rerank_context_size(), DEFAULT_RERANK_CONTEXT_SIZE);
    });
}

#[test]
#[serial]
fn resolve_embed_context_size_parses_valid_env() {
    with_set(EMBED_CTX_ENV, "8192", || {
        assert_eq!(resolve_embed_context_size(), 8192);
    });
}

#[test]
#[serial]
fn resolve_embed_context_size_warns_and_falls_back_on_invalid_env() {
    with_set(EMBED_CTX_ENV, "garbage", || {
        assert_eq!(resolve_embed_context_size(), DEFAULT_EMBED_CONTEXT_SIZE);
    });
    with_set(EMBED_CTX_ENV, "0", || {
        assert_eq!(resolve_embed_context_size(), DEFAULT_EMBED_CONTEXT_SIZE);
    });
    with_set(EMBED_CTX_ENV, "  ", || {
        assert_eq!(resolve_embed_context_size(), DEFAULT_EMBED_CONTEXT_SIZE);
    });
}

#[test]
#[serial]
fn resolve_expand_context_size_config_wins_over_env() {
    with_set(EXPAND_CTX_ENV, "9999", || {
        assert_eq!(resolve_expand_context_size(Some(1234)).unwrap(), 1234);
    });
}

#[test]
#[serial]
fn resolve_expand_context_size_env_used_when_no_config() {
    with_set(EXPAND_CTX_ENV, "3072", || {
        assert_eq!(resolve_expand_context_size(None).unwrap(), 3072);
    });
}

#[test]
#[serial]
fn resolve_expand_context_size_defaults_when_neither_set() {
    with_unset(EXPAND_CTX_ENV, || {
        assert_eq!(
            resolve_expand_context_size(None).unwrap(),
            DEFAULT_EXPAND_CONTEXT_SIZE
        );
    });
}

#[test]
#[serial]
fn resolve_expand_context_size_errors_on_zero_config() {
    let err = resolve_expand_context_size(Some(0)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("must be a positive integer"), "got: {msg}");
}

#[test]
#[serial]
fn resolve_expand_context_size_warns_and_falls_back_on_invalid_env() {
    // TS parity: "falls back to default and warns when RQMD_EXPAND_CONTEXT_SIZE
    // is invalid". Mirrors the embed-version test above: a non-positive /
    // non-numeric / whitespace env value falls back to the default.
    for bad in ["bad", "0", "  "] {
        with_set(EXPAND_CTX_ENV, bad, || {
            assert_eq!(
                resolve_expand_context_size(None).unwrap(),
                DEFAULT_EXPAND_CONTEXT_SIZE,
                "RQMD_EXPAND_CONTEXT_SIZE={bad:?} must fall back to default",
            );
        });
    }
}

// =============================================================================
// Constructor ↔ resolver parity
// =============================================================================

#[test]
#[serial]
fn constructor_resolves_model_uris_via_the_same_resolver() {
    // TS parity: `new LlamaCpp(config).embedModelName === resolveEmbedModel(config)`.
    // This is somewhat tautological in Rust because `LlamaCpp::new` calls
    // `resolve_*_model` internally — it's a low-cost regression guard against a
    // future refactor that bypasses the resolver, NOT a meaningful behavior test.
    with_unset(EMBED_ENV, || {
        with_unset(GENERATE_ENV, || {
            with_unset(RERANK_ENV, || {
                let llm = LlamaCpp::new(LlamaCppConfig {
                    embed_model: Some("hf:config/embed/file.gguf".into()),
                    generate_model: Some("hf:config/generate/file.gguf".into()),
                    rerank_model: Some("hf:config/rerank/file.gguf".into()),
                    ..Default::default()
                });
                let res = ModelResolutionConfig {
                    embed: Some("hf:config/embed/file.gguf".into()),
                    generate: Some("hf:config/generate/file.gguf".into()),
                    rerank: Some("hf:config/rerank/file.gguf".into()),
                    ..Default::default()
                };
                assert_eq!(llm.embed_model_uri(), resolve_embed_model(Some(&res)));
                assert_eq!(llm.generate_model_uri(), resolve_generate_model(Some(&res)));
                assert_eq!(llm.rerank_model_uri(), resolve_rerank_model(Some(&res)));
            });
        });
    });
}
