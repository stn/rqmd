//! Tests for the default-instance singleton (`rmd_core::llm::singleton`).
//!
//! All tests are `#[serial]` because they share the process-global
//! `DEFAULT` slot. Test isolation pattern: each test installs its
//! own LlamaCpp via `install_default(...)` (or clears via
//! `set_default_llama_cpp(None)`) at the start, and the next test
//! does the same. The drop of the previous Arc happens whenever the
//! last holder goes out of scope — we don't try to dispose between
//! tests because dispose is async.

use std::sync::Arc;

use rmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rmd_core::llm::singleton::{
    default_llama_cpp, dispose_default_llama_cpp, install_default, set_default_llama_cpp,
};
use serial_test::serial;

fn ci_config() -> LlamaCppConfig {
    LlamaCppConfig {
        ci_mode: true,
        ..Default::default()
    }
}

#[tokio::test]
#[serial]
async fn default_llama_cpp_returns_the_same_arc_twice() {
    // Clear any leftover state from prior tests.
    set_default_llama_cpp(None);
    let installed = install_default(ci_config());

    let a = default_llama_cpp();
    let b = default_llama_cpp();
    assert!(Arc::ptr_eq(&a, &b), "both reads must yield the same Arc");
    assert!(Arc::ptr_eq(&a, &installed));
}

#[tokio::test]
#[serial]
async fn set_default_llama_cpp_replaces_existing_instance() {
    set_default_llama_cpp(None);
    let first = install_default(ci_config());
    let second = Arc::new(LlamaCpp::new(ci_config()));
    set_default_llama_cpp(Some(second.clone()));

    let got = default_llama_cpp();
    assert!(
        Arc::ptr_eq(&got, &second),
        "default_llama_cpp must return the most recently installed instance"
    );
    assert!(!Arc::ptr_eq(&got, &first));
}

#[tokio::test]
#[serial]
async fn set_default_llama_cpp_none_clears_slot() {
    set_default_llama_cpp(None);
    install_default(ci_config());

    set_default_llama_cpp(None);

    // The next read will lazy-construct a fresh instance from env. We
    // can't assert what it is, only that it's different from the one
    // we just cleared (we already let that one drop).
    let fresh = default_llama_cpp();
    assert!(!fresh.is_disposed());
    // Clean up so the next test starts from a known state.
    set_default_llama_cpp(None);
}

#[tokio::test]
#[serial]
async fn dispose_default_llama_cpp_is_safe_when_unset() {
    set_default_llama_cpp(None);
    // No panic, no error path.
    dispose_default_llama_cpp().await;
}

#[tokio::test]
#[serial]
async fn dispose_default_llama_cpp_disposes_installed_instance() {
    set_default_llama_cpp(None);
    let installed = install_default(ci_config());
    assert!(!installed.is_disposed());

    dispose_default_llama_cpp().await;

    assert!(
        installed.is_disposed(),
        "dispose_default must call dispose() on the installed instance"
    );

    // Subsequent default_llama_cpp() lazy-constructs a fresh instance.
    let fresh = default_llama_cpp();
    assert!(!Arc::ptr_eq(&fresh, &installed));
    assert!(!fresh.is_disposed());

    set_default_llama_cpp(None);
}
