//! Compile-time regression for `Arc<LlmSession> → Arc<dyn Llm>` coercion.
//! These tests do no I/O — they exist purely so the type relationships
//! `store_ops` depends on cannot regress silently.

use std::sync::Arc;

use rqmd_core::LlamaCppConfig;
use rqmd_core::llm::llama_cpp::LlamaCpp;
use rqmd_core::llm::session::{LlmSession, LlmSessionOptions, with_llm_session};
use rqmd_core::llm::traits::Llm;

#[allow(dead_code)]
async fn takes_llm(_llm: Arc<dyn Llm>) {}

#[tokio::test]
async fn arc_session_coerces_at_variable_binding() {
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }));
    let session: Arc<LlmSession> = LlmSession::new(llm, LlmSessionOptions::default());
    // Explicit binding to Arc<dyn Llm>. Compile-only check.
    let _dyn_session: Arc<dyn Llm> = session.clone();
    session.release();
}

#[tokio::test]
async fn arc_session_coerces_at_closure_boundary() {
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }));
    // Inside `with_llm_session` the closure receives Arc<LlmSession>; the
    // canonical 1-line `let _: Arc<dyn Llm> = session;` cast is enough.
    let _ = with_llm_session(
        llm,
        LlmSessionOptions::default(),
        |session: Arc<LlmSession>| async move {
            let _dyn_session: Arc<dyn Llm> = session;
            Ok::<(), rqmd_core::llm::error::Error>(())
        },
    )
    .await;
}
