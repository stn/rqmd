//! Scoped LLM session with cancel-on-drop semantics.
//!
//! Mirrors `tobi/qmd/src/llm.ts` lines 1562–1805. A session is a thin
//! handle around an `Arc<LlamaCpp>` that adds:
//!
//! * A `CancellationToken` callers can pass to long-running operations.
//! * An optional `max_duration` deadline — when it fires, the session's
//!   token is cancelled.
//! * Cancel-on-drop: dropping the session aborts its token, so a
//!   `with_llm_session` scope cleanly signals shutdown when its
//!   callback returns or panics.
//!
//! What this module **does not** do (deliberately deferred to a
//! follow-up PR, per v3 plan):
//!
//! * Session-counter integration with an inactivity timer. PR2 ships
//!   no inactivity timer, so there is no need for the TS
//!   `canUnloadLLM` mechanism. `LlamaCpp::dispose` already drains the
//!   per-method `in_flight` counter, which is sufficient for the
//!   scoped-call shutdown story.
//! * Mid-FFI cancellation. Calling `signal().cancel()` after a method
//!   has already dispatched a job to a worker thread does NOT abort
//!   the C++ `decode`; the abort only short-circuits *future* method
//!   calls on the session. This limitation is fundamental to
//!   llama-cpp-2 (see v3 plan §Risks) and the doc on
//!   [`LlmSession::signal`] makes it explicit.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use async_trait::async_trait;

use crate::llm::error::{Error, Result};
use crate::llm::traits::{LlamaToken, Llm};
use crate::llm::types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    Queryable, RerankDocument, RerankOptions, RerankResult,
};

/// Options for [`with_llm_session`].
#[derive(Debug, Clone, Default)]
pub struct LlmSessionOptions {
    /// Maximum wall-clock duration the session is valid for. When it
    /// elapses, [`LlmSession::signal`] is cancelled and subsequent
    /// method calls return [`Error::SessionReleased`].
    pub max_duration: Option<Duration>,
    /// Debug label used in error messages.
    pub name: Option<String>,
}

/// Sentinel returned in [`Error::SessionReleased`] when the session
/// was released cleanly (vs. timed out / externally aborted).
pub const SESSION_RELEASED_REASON: &str = "session released";

/// A scoped handle to an [`Arc<dyn Llm>`]. See module docs.
///
/// Holds a trait object rather than a concrete `LlamaCpp` so orchestrators
/// (e.g. `generate_embeddings`) and tests can drive the session with any
/// [`Llm`] impl, including fakes that inject embedding failures.
pub struct LlmSession {
    llm: Arc<dyn Llm>,
    name: String,
    released: AtomicBool,
    abort: CancellationToken,
}

impl LlmSession {
    /// Construct a new session. Prefer [`with_llm_session`] for the
    /// scoped-RAII shape; this lower-level constructor is exposed for
    /// long-lived sessions managed externally.
    pub fn new(llm: Arc<dyn Llm>, options: LlmSessionOptions) -> Arc<Self> {
        let session = Arc::new(Self {
            llm,
            name: options.name.unwrap_or_else(|| "unnamed".into()),
            released: AtomicBool::new(false),
            abort: CancellationToken::new(),
        });

        if let Some(max_dur) = options.max_duration {
            let weak = Arc::downgrade(&session);
            tokio::spawn(async move {
                tokio::time::sleep(max_dur).await;
                if let Some(s) = weak.upgrade()
                    && !s.released.load(Ordering::Acquire)
                {
                    s.abort.cancel();
                }
            });
        }

        session
    }

    /// Debug label.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Cancellation token for this session. Becomes cancelled when the
    /// session is released (drop), its `max_duration` elapses, or a
    /// caller invokes [`Self::release`].
    ///
    /// **Cancellation contract**: cancelling the token short-circuits
    /// new method calls on this session, but does NOT abort an
    /// in-flight C++ decode loop. Callers needing hard cancellation
    /// of long generations should size `max_duration` accordingly.
    pub fn signal(&self) -> &CancellationToken {
        &self.abort
    }

    /// True iff the session has not been released AND its abort token
    /// has not fired.
    pub fn is_valid(&self) -> bool {
        !self.released.load(Ordering::Acquire) && !self.abort.is_cancelled()
    }

    /// Explicit release. Safe to call multiple times; only the first
    /// call has an effect.
    pub fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.abort.cancel();
        }
    }

    fn check_valid(&self) -> Result<()> {
        if !self.is_valid() {
            return Err(Error::SessionReleased(self.name.clone()));
        }
        Ok(())
    }

    // ----- Delegated Llm methods --------------------------------------------

    pub async fn embed(&self, text: &str, opts: EmbedOptions) -> Result<Option<EmbeddingResult>> {
        self.check_valid()?;
        self.llm.embed(text, opts).await
    }

    pub async fn embed_batch(
        &self,
        texts: &[String],
        opts: EmbedOptions,
    ) -> Result<Vec<Option<EmbeddingResult>>> {
        self.check_valid()?;
        self.llm.embed_batch(texts, opts).await
    }

    pub async fn generate(
        &self,
        prompt: &str,
        opts: GenerateOptions,
    ) -> Result<Option<GenerateResult>> {
        self.check_valid()?;
        self.llm.generate(prompt, opts).await
    }

    pub async fn expand_query(
        &self,
        query: &str,
        opts: ExpandQueryOptions,
    ) -> Result<Vec<Queryable>> {
        self.check_valid()?;
        self.llm.expand_query(query, opts).await
    }

    pub async fn rerank(
        &self,
        query: &str,
        docs: &[RerankDocument],
        opts: RerankOptions,
    ) -> Result<RerankResult> {
        self.check_valid()?;
        self.llm.rerank(query, docs, opts).await
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        self.check_valid()?;
        self.llm.tokenize(text).await
    }

    pub async fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String> {
        self.check_valid()?;
        self.llm.detokenize(tokens).await
    }
}

impl Drop for LlmSession {
    fn drop(&mut self) {
        // Cancel-on-drop. If the session was already released
        // explicitly, this is a no-op.
        if !self.released.swap(true, Ordering::AcqRel) {
            self.abort.cancel();
        }
    }
}

/// `LlmSession` as an `Llm` trait object. Lets `store_ops` orchestrators
/// accept `Arc<dyn Llm>` for both bare `Arc<LlamaCpp>` and
/// `Arc<LlmSession>` callers. Method bodies delegate to the inherent
/// methods, which add session-validity guards on top of the underlying
/// `LlamaCpp`.
///
/// `dispose` is a deliberate no-op: an `Arc<LlmSession>` does not own
/// the `LlamaCpp` outright (other handles may exist), and the session's
/// `Drop` already cancels its abort token. Tearing down the underlying
/// `LlamaCpp` from a trait-object dispose call would be a footgun.
/// `model_exists` delegates to the wrapped `LlamaCpp`.
#[async_trait]
impl Llm for LlmSession {
    fn embed_context_size(&self) -> usize {
        self.llm.embed_context_size()
    }

    async fn embed(&self, text: &str, opts: EmbedOptions) -> Result<Option<EmbeddingResult>> {
        Self::embed(self, text, opts).await
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        opts: EmbedOptions,
    ) -> Result<Vec<Option<EmbeddingResult>>> {
        Self::embed_batch(self, texts, opts).await
    }

    async fn generate(
        &self,
        prompt: &str,
        opts: GenerateOptions,
    ) -> Result<Option<GenerateResult>> {
        Self::generate(self, prompt, opts).await
    }

    async fn model_exists(&self, model: &str) -> Result<ModelInfo> {
        self.llm.model_exists(model).await
    }

    async fn expand_query(&self, query: &str, opts: ExpandQueryOptions) -> Result<Vec<Queryable>> {
        Self::expand_query(self, query, opts).await
    }

    async fn rerank(
        &self,
        query: &str,
        docs: &[RerankDocument],
        opts: RerankOptions,
    ) -> Result<RerankResult> {
        Self::rerank(self, query, docs, opts).await
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        Self::tokenize(self, text).await
    }

    async fn detokenize(&self, tokens: &[LlamaToken]) -> Result<String> {
        Self::detokenize(self, tokens).await
    }

    /// No-op: see type-level docs.
    async fn dispose(&self) {}
}

/// Run `f` with a scoped [`LlmSession`]. The session is automatically
/// released when `f` returns (or panics) — its abort token fires, the
/// `max_duration` timer task observes the released flag and exits, and
/// any further calls on the session return [`Error::SessionReleased`].
///
/// Equivalent to TS `withLLMSession`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
/// # use rqmd_core::llm::session::{LlmSessionOptions, with_llm_session};
/// # use rqmd_core::llm::types::EmbedOptions;
/// # async fn demo() -> rqmd_core::llm::error::Result<()> {
/// let llm = Arc::new(LlamaCpp::new(LlamaCppConfig::default()));
/// let result = with_llm_session(
///     llm,
///     LlmSessionOptions { name: Some("demo".into()), ..Default::default() },
///     |session| async move {
///         session.embed("hello", EmbedOptions::default()).await
///     },
/// ).await?;
/// # let _ = result;
/// # Ok(())
/// # }
/// ```
pub async fn with_llm_session<F, Fut, T>(
    llm: Arc<dyn Llm>,
    options: LlmSessionOptions,
    f: F,
) -> Result<T>
where
    F: FnOnce(Arc<LlmSession>) -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let session = LlmSession::new(llm, options);
    let result = f(session.clone()).await;
    session.release();
    result
}
