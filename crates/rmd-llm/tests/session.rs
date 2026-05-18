//! Session lifecycle tests.
//!
//! These run against a `ci_mode: true` LlamaCpp so no GGUF model is
//! ever loaded. We're testing the session machinery itself (cancel
//! tokens, release semantics, max_duration), not LLM inference.

use std::sync::Arc;
use std::time::Duration;

use rmd_llm::error::Error;
use rmd_llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rmd_llm::session::{LlmSession, LlmSessionOptions, with_llm_session};
use rmd_llm::types::EmbedOptions;

fn ci_llm() -> Arc<LlamaCpp> {
    Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }))
}

#[tokio::test]
async fn fresh_session_is_valid_and_has_uncancelled_token() {
    let llm = ci_llm();
    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            name: Some("fresh".into()),
            ..Default::default()
        },
    );
    assert!(session.is_valid());
    assert!(!session.signal().is_cancelled());
    assert_eq!(session.name(), "fresh");
}

#[tokio::test]
async fn explicit_release_invalidates_session_and_cancels_token() {
    let llm = ci_llm();
    let session = LlmSession::new(llm, LlmSessionOptions::default());
    assert!(session.is_valid());

    session.release();

    assert!(!session.is_valid());
    assert!(session.signal().is_cancelled());
}

#[tokio::test]
async fn release_is_idempotent() {
    let llm = ci_llm();
    let session = LlmSession::new(llm, LlmSessionOptions::default());
    session.release();
    session.release();
    session.release();
    assert!(!session.is_valid());
}

#[tokio::test]
async fn drop_releases_session_and_cancels_token() {
    let llm = ci_llm();
    let token = {
        let session = LlmSession::new(llm, LlmSessionOptions::default());
        let token = session.signal().clone();
        assert!(!token.is_cancelled());
        token
        // session dropped here
    };
    assert!(token.is_cancelled(), "drop must cancel the session token");
}

#[tokio::test]
async fn method_call_on_released_session_returns_session_released() {
    let llm = ci_llm();
    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            name: Some("released-session".into()),
            ..Default::default()
        },
    );
    session.release();

    let err = session
        .embed("anything", EmbedOptions::default())
        .await
        .unwrap_err();
    match err {
        Error::SessionReleased(name) => {
            assert_eq!(name, "released-session");
        }
        other => panic!("expected SessionReleased, got {other:?}"),
    }
}

#[tokio::test]
async fn max_duration_cancels_session_after_elapsed() {
    let llm = ci_llm();
    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            max_duration: Some(Duration::from_millis(50)),
            name: Some("timed".into()),
        },
    );
    let token = session.signal().clone();
    assert!(!token.is_cancelled());

    // Wait a bit more than max_duration for the spawned timer to fire.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(token.is_cancelled(), "max_duration timer must cancel the token");
    assert!(!session.is_valid());
}

#[tokio::test]
async fn with_llm_session_runs_callback_and_releases_after() {
    let llm = ci_llm();
    let captured_token = with_llm_session(
        llm,
        LlmSessionOptions {
            name: Some("scoped".into()),
            ..Default::default()
        },
        |session| async move {
            assert!(session.is_valid());
            assert_eq!(session.name(), "scoped");
            // Return the token so we can check post-release state outside.
            Ok::<_, Error>(session.signal().clone())
        },
    )
    .await
    .unwrap();

    assert!(captured_token.is_cancelled(), "session must be released after closure returns");
}

#[tokio::test]
async fn with_llm_session_propagates_callback_error() {
    let llm = ci_llm();
    let result: Result<(), Error> = with_llm_session(
        llm,
        LlmSessionOptions::default(),
        |_session| async move { Err(Error::Disposed) },
    )
    .await;
    assert!(matches!(result, Err(Error::Disposed)));
}

#[tokio::test]
async fn with_llm_session_releases_even_when_callback_errors() {
    let llm = ci_llm();
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let captured_for_closure = captured.clone();

    let _ = with_llm_session(
        llm,
        LlmSessionOptions::default(),
        |session| async move {
            *captured_for_closure.lock().unwrap() = Some(session.signal().clone());
            Err::<(), _>(Error::Disposed)
        },
    )
    .await;

    let token = captured.lock().unwrap().take().expect("token captured");
    assert!(token.is_cancelled());
}

#[tokio::test]
async fn delegated_methods_propagate_ci_disabled_when_underlying_llm_is_ci_mode() {
    let llm = ci_llm();
    let session = LlmSession::new(llm, LlmSessionOptions::default());

    // embed_batch goes through the CI guard on the underlying LlamaCpp.
    let err = session
        .embed_batch(&["hello".into()], EmbedOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::CiDisabled), "got {err:?}");
}
