//! Spike #1 — verify critical assumption #3 from the plan:
//! `LlamaBackend::init()` is process-wide one-shot, so it cannot live as an
//! `Arc<LlamaBackend>` field on `LlamaCpp`. The real implementation must use
//! a `static OnceLock<LlamaBackend>` accessed via helper.
//!
//! Pass criteria:
//!   * Two `backend()` calls return the same pointer.
//!   * A direct second `LlamaBackend::init()` returns an error.
//!   * Record the exact error variant for use in the v2 plan.

use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("first backend init must succeed"))
}

fn main() {
    let b1 = backend();
    let b2 = backend();
    assert!(
        std::ptr::eq(b1, b2),
        "OnceLock must return the same backend instance on every call"
    );
    println!("OK: shared backend = {b1:p}");

    // Direct second init should fail. We deliberately call the *raw* init here
    // (not the cached `backend()` helper) to observe the failure mode that
    // future code must avoid.
    match LlamaBackend::init() {
        Ok(_b) => panic!("expected second LlamaBackend::init() to fail, but it succeeded"),
        Err(e) => {
            println!("OK: second init returned error variant = {e:?}");
            println!("OK: error display = {e}");
        }
    }

    println!();
    println!("Notes for v2 plan:");
    println!("  - Use `static OnceLock<LlamaBackend>` in rmd-llm.");
    println!("  - LlamaCpp struct must NOT hold its own LlamaBackend.");
    println!("  - Tests that construct multiple LlamaCpp instances must share the static.");
}
