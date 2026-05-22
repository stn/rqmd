//! Parity for the *containment* part of TS `native llama stdout containment`.
//!
//! `tobi/qmd` wraps llama init in `withNativeStdoutRedirectedToStderr` so the
//! node-llama-cpp native chatter doesn't pollute stdout. The rqmd equivalent
//! is `backend::install_log_redirect` calling
//! `llama_cpp_2::send_logs_to_tracing(...)`, which reroutes llama.cpp + ggml
//! native C logs (`llama_context:`, `sched_reserve:`, `decode: ...`) through
//! `tracing` instead of letting them hit the C stderr directly.
//!
//! This test installs a process-global tracing subscriber, loads a model
//! (which emits many native log lines), and asserts the subscriber captured
//! them. A regression that drops the redirect would capture zero events.
//!
//! NOTE: The other two TS tests in that block are intentionally NOT ported —
//! they have no rqmd behavior to exercise. llama-cpp-2 picks the GPU backend
//! at COMPILE time (Cargo features `metal`/`cuda`/`vulkan`), so there is no
//! runtime "try CUDA, fail, cache, fall back to CPU" path to keep off stdout
//! or to warn about once-per-process (see `rqmd_core::llm::gpu`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::EmbedOptions;

/// Minimal global subscriber that just counts emitted events. Hand-rolled to
/// avoid a `tracing-subscriber` dev-dependency. Native logs are emitted from
/// the FFI worker threads, which have no thread-local default dispatcher, so
/// the subscriber MUST be installed via `set_global_default`.
struct CountingSubscriber {
    events: Arc<AtomicUsize>,
}

impl tracing::Subscriber for CountingSubscriber {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {
        self.events.fetch_add(1, Ordering::Relaxed);
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "loads embeddinggemma-300M (~300 MB) and runs CPU inference"]
async fn native_llama_logs_are_routed_into_tracing() {
    let events = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::set_global_default(CountingSubscriber {
        events: events.clone(),
    })
    .expect("this test owns the process-global tracing subscriber");

    let llm = LlamaCpp::new(LlamaCppConfig {
        embed_parallelism: Some(1),
        ..Default::default()
    });

    // Loading + running the embed model emits many native llama/ggml log lines.
    // With the redirect installed they flow through `tracing` (our subscriber);
    // without it they would go straight to the C stderr and never be counted.
    let _ = llm
        .embed("contain this native noise", EmbedOptions::default())
        .await
        .expect("embed must succeed");
    llm.dispose().await;

    let n = events.load(Ordering::Relaxed);
    assert!(
        n > 0,
        "expected llama.cpp native logs to be captured by tracing (got {n}); \
         the backend log redirect may be missing"
    );
}
