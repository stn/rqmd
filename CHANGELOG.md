# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-28

### Added

- `RqmdStore::doctor_report` and `RqmdStore::adopt_legacy_embeddings` expose
  the full `rqmd doctor` diagnostics — SQLite/sqlite-vec versions, resolved
  vs. configured models, env overrides, device probe, fingerprint groups,
  legacy-row count, embedding sample check — as an SDK-callable
  `DoctorReport` struct. The CLI is now a thin formatter over these methods.

### Changed

- Crate-root `rqmd_core::Error` / `Result` now alias the `rqmd_store`
  aggregator (Store ∪ Collections ∪ Llm ∪ StoreOps) instead of the
  collections-only error. Bare `rqmd_core::Error` previously looked
  general-purpose but silently caught only collections errors.
- Removed the `RqmdStoreError` / `RqmdStoreResult` aliases; use
  `rqmd_core::Error` / `Result` directly.
- Re-exported `CollectionListing`, `FindDocumentOptions`,
  `FindDocumentOutcome`, and `FindDocumentsOptions` at the crate root so
  callers no longer need deep `store::lookup::*` / `store::context::*`
  paths.

## [0.2.0] - 2026-05-27

### Breaking changes

- **All runtime env vars renamed from `QMD_*` to `RQMD_*`** so qmd and rqmd can
  coexist on the same machine without configuration crosstalk. No fallback to
  the old names is provided. Replace any occurrence of:
  - `QMD_EMBED_MODEL` → `RQMD_EMBED_MODEL`
  - `QMD_GENERATE_MODEL` → `RQMD_GENERATE_MODEL`
  - `QMD_RERANK_MODEL` → `RQMD_RERANK_MODEL`
  - `QMD_FORCE_CPU` → `RQMD_FORCE_CPU`
  - `QMD_LLAMA_GPU` → `RQMD_LLAMA_GPU`
  - `QMD_DOCTOR_DEVICE_PROBE` → `RQMD_DOCTOR_DEVICE_PROBE`
  - `QMD_EMBED_PARALLELISM` → `RQMD_EMBED_PARALLELISM`
  - `QMD_RERANK_PARALLELISM` → `RQMD_RERANK_PARALLELISM`
  - `QMD_EMBED_CONTEXT_SIZE` → `RQMD_EMBED_CONTEXT_SIZE`
  - `QMD_RERANK_CONTEXT_SIZE` → `RQMD_RERANK_CONTEXT_SIZE`
  - `QMD_EXPAND_CONTEXT_SIZE` → `RQMD_EXPAND_CONTEXT_SIZE`
  - `QMD_EXPAND_USER_MESSAGE_PREFIX` → `RQMD_EXPAND_USER_MESSAGE_PREFIX`
  - `QMD_EXPAND_SYSTEM_MESSAGE` → `RQMD_EXPAND_SYSTEM_MESSAGE`
  - `QMD_EXPAND_FALLBACK_HYDE_TEMPLATE` → `RQMD_EXPAND_FALLBACK_HYDE_TEMPLATE`
  - `QMD_EXPAND_TEMP` → `RQMD_EXPAND_TEMP`
  - `QMD_EXPAND_TOP_K` → `RQMD_EXPAND_TOP_K`
  - `QMD_EXPAND_TOP_P` → `RQMD_EXPAND_TOP_P`

### Removed

- The `QMD_EDITOR_URI` fallback for `RQMD_EDITOR_URI` was dropped — only the
  `RQMD_` form is read now.
- The retired `QMD_STATUS_DEVICE_PROBE` reference was removed from the
  `status` regression test.

### Internal

- Added `rqmd_core::env_keys` — a single module that defines the constants
  for the 17 renamed env var names. Production code, tests, and `doctor`
  output all reference these constants so a future rename can't drift
  between sites.

## [0.1.1]

Previous releases — see Git history.
