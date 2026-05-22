//! Unit tests for `LlamaCpp::model_exists` (TS parity:
//! `describe("LlamaCpp.modelExists")` in `tobi/qmd/test/llm.test.ts`).
//!
//! `model_exists` does no model loading and has no CI guard — it only
//! checks the `hf:` URI prefix or `Path::exists()` (see
//! `llama_cpp.rs::model_exists`). So these run against a `ci_mode: true`
//! instance with no GGUF available and need no `#[ignore]`.

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;

fn ci_llm() -> LlamaCpp {
    LlamaCpp::new(LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    })
}

#[tokio::test]
async fn hf_uri_reports_exists_true() {
    let llm = ci_llm();
    let info = llm
        .model_exists("hf:org/repo/model.gguf")
        .await
        .expect("model_exists must not error for hf URIs");
    assert!(info.exists, "hf: URIs are always considered to exist");
    assert_eq!(info.name, "hf:org/repo/model.gguf");
}

#[tokio::test]
async fn nonexistent_local_path_reports_exists_false() {
    let llm = ci_llm();
    let path = "/nonexistent/path/model.gguf";
    let info = llm
        .model_exists(path)
        .await
        .expect("model_exists must not error for local paths");
    assert!(
        !info.exists,
        "a missing local path must report exists=false"
    );
    assert_eq!(info.name, path);
    assert!(
        info.path.is_none(),
        "no path is returned when the file is absent"
    );
}
