//! `ci_mode` guard tests for the `Llm` impl on `LlamaCpp`.
//!
//! These tests construct `LlamaCpp` with `ci_mode: true` injected
//! directly into [`LlamaCppConfig`] — they intentionally do NOT touch
//! `std::env::set_var("CI", "true")`. Rust 2024 edition made
//! `set_var` `unsafe`, and `cargo test` runs tests in parallel by
//! default, so a shared `CI` env var would race with PR1 config tests
//! that also touch env vars.
//!
//! The serial_test attribute is here as a belt-and-suspenders measure:
//! if a future refactor moves any of these tests to actually mutate
//! env, they'll already be marked.

use std::sync::Arc;

use rqmd_core::llm::error::Error;
use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::{
    EmbedOptions, ExpandQueryOptions, GenerateOptions, RerankDocument, RerankOptions,
};
use serial_test::serial;

fn ci_llm() -> LlamaCpp {
    LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    })
}

#[tokio::test]
#[serial]
async fn embed_batch_returns_ci_disabled_when_ci_mode_set() {
    let llm = ci_llm();
    let err = llm
        .embed_batch(&["hello".into()], EmbedOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}

#[tokio::test]
#[serial]
async fn generate_returns_ci_disabled_when_ci_mode_set() {
    let llm = ci_llm();
    let err = llm
        .generate("hello", GenerateOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}

#[tokio::test]
#[serial]
async fn expand_query_returns_ci_disabled_when_ci_mode_set() {
    let llm = ci_llm();
    let err = llm
        .expand_query("rust async runtime", ExpandQueryOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}

#[tokio::test]
#[serial]
async fn rerank_returns_ci_disabled_when_ci_mode_set() {
    let llm = ci_llm();
    let err = llm
        .rerank(
            "rust async",
            &[RerankDocument {
                file: "a.md".into(),
                text: "tokio is a runtime".into(),
                title: None,
            }],
            RerankOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}

#[tokio::test]
#[serial]
async fn rerank_with_empty_docs_short_circuits_without_ci_check() {
    // Empty input is checked AFTER the CI guard in the impl — verify that
    // CI mode wins. (If we ever reorder, this test catches the regression.)
    let llm = ci_llm();
    let err = llm
        .rerank("anything", &[], RerankOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}

#[tokio::test]
#[serial]
async fn embed_single_does_not_check_ci_mode_matching_ts() {
    // The TS source explicitly omits the CI guard on `embed()` (only
    // `embedBatch` / `generate` / `expandQuery` / `rerank` check it).
    // We mirror that. `embed()` will instead try to load the model and
    // — in CI with no GGUF available — fail with an HfApi /
    // ModelLoad error, NOT CiDisabled.
    //
    // We can't easily run the load path in unit-test scope (no network,
    // no model), so just assert the error is NOT CiDisabled. Any other
    // error variant proves CI guard was skipped.
    let llm = ci_llm();
    let result = llm.embed("hello", EmbedOptions::default()).await;
    if let Err(Error::CiDisabled) = result {
        panic!("embed() must not check ci_mode per TS parity")
    }
    // any other outcome (Ok / non-CiDisabled Err) is fine.
}

#[tokio::test]
#[serial]
async fn tokenize_does_not_check_ci_mode() {
    // Tokenize bypasses the CI guard the same as embed() — it's a
    // model-tokenizer query, not an inference call. With no GGUF
    // available it'll fail with a non-CiDisabled error.
    let llm = ci_llm();
    let result = llm.tokenize("hello").await;
    if let Err(Error::CiDisabled) = result {
        panic!("tokenize() must not check ci_mode");
    }
}

#[tokio::test]
#[serial]
async fn ci_mode_field_is_observable() {
    let normal = LlamaCpp::new(LlamaCppConfig::default());
    assert!(!normal.ci_mode());
    let ci = ci_llm();
    assert!(ci.ci_mode());
}

#[tokio::test]
#[serial]
async fn dispose_is_idempotent() {
    let llm = ci_llm();
    assert!(!llm.is_disposed());
    llm.dispose().await;
    assert!(llm.is_disposed());
    // Second dispose is a no-op (and must not panic).
    llm.dispose().await;
    assert!(llm.is_disposed());
}

#[tokio::test]
#[serial]
async fn disposed_returns_disposed_error_on_methods_that_acquire() {
    let llm = ci_llm();
    llm.dispose().await;

    // CI guard fires first in the method body, so we'd get CiDisabled
    // here. Use `embed()` (which skips CI) to actually surface the
    // Disposed error.
    let err = llm.embed("hi", EmbedOptions::default()).await.unwrap_err();
    assert!(matches!(err, Error::Disposed), "got {err:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn in_flight_count_returns_to_zero_after_concurrent_method_calls() {
    // Even though ci_mode short-circuits BEFORE acquire (so the counter
    // never actually goes above 0 here), this test guards against a
    // regression where a future refactor moves the CI guard AFTER
    // acquire and leaks the counter on the error path.
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }));
    assert_eq!(llm.in_flight_count(), 0);

    let mut handles = Vec::new();
    for _ in 0..16 {
        let llm = llm.clone();
        handles.push(tokio::spawn(async move {
            llm.embed_batch(&["x".into()], EmbedOptions::default())
                .await
        }));
    }
    for h in handles {
        let _ = h.await;
    }
    assert_eq!(
        llm.in_flight_count(),
        0,
        "in-flight counter must drain to 0 even when methods fail fast"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn dispose_during_concurrent_failing_calls_drains_cleanly() {
    // Spawn N concurrent embed() calls (which bypass the ci_mode
    // guard) against a ci-mode instance. They'll race to acquire
    // their in-flight guard then fail at ensure_embed_pool ->
    // ensure_embed_model -> load_model (no network in test scope).
    // Dispose runs concurrently; we assert that:
    //   1. dispose returns within a bounded time (i.e. drains).
    //   2. in_flight is 0 after.
    //   3. is_disposed reports true.
    //
    // This doesn't fully exercise the dispose-vs-acquire race
    // (which needs real long-running ops) but does catch counter
    // leaks under contention.
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let llm = llm.clone();
        handles.push(tokio::spawn(async move {
            // embed() bypasses ci_mode so this actually exercises the
            // model-load failure path under concurrent dispose.
            let _ = llm.embed("x", EmbedOptions::default()).await;
        }));
    }
    // Let some spawn before dispose.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let dispose_handle = {
        let llm = llm.clone();
        tokio::spawn(async move { llm.dispose().await })
    };

    for h in handles {
        let _ = h.await;
    }
    // Cap dispose wait so a regression that causes dispose to hang
    // surfaces as a test failure rather than CI timeout.
    let timed = tokio::time::timeout(std::time::Duration::from_secs(5), dispose_handle).await;
    assert!(timed.is_ok(), "dispose must return within 5s");
    timed.unwrap().unwrap();

    assert_eq!(llm.in_flight_count(), 0);
    assert!(llm.is_disposed());
}
