//! Centralised env-var name constants for the rqmd runtime.
//!
//! All runtime-tunable knobs use the `RQMD_` prefix so that rqmd installs
//! never read configuration intended for upstream `qmd`. Production code,
//! tests, and `doctor` output all reference these constants rather than
//! string literals so a future rename can't drift between sites.
//!
//! Only the env vars touched by the 0.2.0 rename live here; the older
//! path/cache vars (`RQMD_INDEX_PATH`, `RQMD_CONFIG_DIR`, `RQMD_CACHE_DIR`,
//! `RQMD_SKILLS_DIR`, `RQMD_EDITOR_URI`, `RQMD_SKIP_LLM_TESTS`) remain at
//! their original call sites and may be consolidated in a follow-up.

// Model URIs.
pub const EMBED_MODEL: &str = "RQMD_EMBED_MODEL";
pub const GENERATE_MODEL: &str = "RQMD_GENERATE_MODEL";
pub const RERANK_MODEL: &str = "RQMD_RERANK_MODEL";

// GPU / device.
pub const FORCE_CPU: &str = "RQMD_FORCE_CPU";
pub const LLAMA_GPU: &str = "RQMD_LLAMA_GPU";
pub const DOCTOR_DEVICE_PROBE: &str = "RQMD_DOCTOR_DEVICE_PROBE";

// Parallelism.
pub const EMBED_PARALLELISM: &str = "RQMD_EMBED_PARALLELISM";
pub const RERANK_PARALLELISM: &str = "RQMD_RERANK_PARALLELISM";

// Context sizes.
pub const EMBED_CONTEXT_SIZE: &str = "RQMD_EMBED_CONTEXT_SIZE";
pub const RERANK_CONTEXT_SIZE: &str = "RQMD_RERANK_CONTEXT_SIZE";
pub const EXPAND_CONTEXT_SIZE: &str = "RQMD_EXPAND_CONTEXT_SIZE";

// Expand-query prompt + sampling.
pub const EXPAND_USER_MESSAGE_PREFIX: &str = "RQMD_EXPAND_USER_MESSAGE_PREFIX";
pub const EXPAND_SYSTEM_MESSAGE: &str = "RQMD_EXPAND_SYSTEM_MESSAGE";
pub const EXPAND_FALLBACK_HYDE_TEMPLATE: &str = "RQMD_EXPAND_FALLBACK_HYDE_TEMPLATE";
pub const EXPAND_TEMP: &str = "RQMD_EXPAND_TEMP";
pub const EXPAND_TOP_K: &str = "RQMD_EXPAND_TOP_K";
pub const EXPAND_TOP_P: &str = "RQMD_EXPAND_TOP_P";
