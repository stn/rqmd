//! Default-instance singleton for [`crate::llm::llama_cpp::LlamaCpp`].
//!
//! Mirrors TS `getDefaultLlamaCpp` / `setDefaultLlamaCpp` /
//! `disposeDefaultLlamaCpp`. The TS source uses a mutable
//! module-level `let defaultLlamaCpp: LlamaCpp | null`. In Rust we
//! use `arc_swap::ArcSwapOption<LlamaCpp>` to get:
//!
//! * Lock-free reads (no mutex held across `.await`).
//! * Atomic swap for `set_default` and `dispose_default`.
//! * The `rcu` pattern for "construct-if-empty" without a TOCTOU race.
//!
//! `LlamaCpp::dispose()` is async, so `dispose_default_llama_cpp()`
//! is too. There is no synchronous public API for disposal because
//! draining in-flight ops needs `tokio::time::sleep`.

use std::sync::Arc;

use arc_swap::ArcSwapOption;

use crate::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use crate::llm::traits::Llm;

static DEFAULT: ArcSwapOption<LlamaCpp> = ArcSwapOption::const_empty();

/// Get (or lazily construct) the default `LlamaCpp` instance.
///
/// Uses `arc_swap::ArcSwapOption::rcu` so concurrent first-callers
/// race to construct an instance, but exactly one wins and is
/// installed; the losers drop their construction and return the
/// winner. Subsequent callers see the cached value with one atomic
/// load, no allocation.
pub fn default_llama_cpp() -> Arc<LlamaCpp> {
    // Fast path: already set.
    if let Some(existing) = DEFAULT.load_full() {
        return existing;
    }
    // Slow path: rcu. Note that `rcu`'s closure can be called more
    // than once if there's contention — that's why we don't observe
    // side effects (e.g. logging) inside it. Construction allocates
    // but doesn't load any model (those are lazy on first call).
    DEFAULT.rcu(|cur| match cur.as_ref() {
        Some(existing) => Some(Arc::clone(existing)),
        None => Some(Arc::new(LlamaCpp::with_env())),
    });
    DEFAULT
        .load_full()
        .expect("rcu just stored Some(LlamaCpp)")
}

/// Replace the default instance. Used by tests to inject a
/// `LlamaCpp` constructed with `ci_mode: true`, or by long-running
/// binaries that want to dispose-and-replace.
///
/// Passing `None` clears the slot without disposing. Use
/// [`dispose_default_llama_cpp`] for the dispose-and-clear path.
pub fn set_default_llama_cpp(llm: Option<Arc<LlamaCpp>>) {
    DEFAULT.store(llm);
}

/// Atomically swap out the default and dispose it.
///
/// If no default has been constructed yet, this is a no-op.
/// Otherwise the previous instance has [`Llm::dispose`] awaited on
/// it before returning. Subsequent calls to
/// [`default_llama_cpp`] construct a fresh instance.
pub async fn dispose_default_llama_cpp() {
    if let Some(prev) = DEFAULT.swap(None) {
        prev.dispose().await;
    }
}

/// Test helper: build a `LlamaCpp` with the given config and install
/// it as the default. Returns the new `Arc<LlamaCpp>`. Exposed
/// outside `#[cfg(test)]` because integration tests live in
/// `crates/rqmd-core/tests/` and can't see
/// test-only items.
pub fn install_default(config: LlamaCppConfig) -> Arc<LlamaCpp> {
    let llm = Arc::new(LlamaCpp::new(config));
    DEFAULT.store(Some(llm.clone()));
    llm
}
