//! `rqmd pull` — download configured GGUF models from HuggingFace.
//!
//! Maps to qmd's `pullModels` CLI handler (`src/cli/qmd.ts` lines 3569–3587)
//! and `pullModels` in `src/llm.ts` (lines 377–435). ETag-based incremental
//! download is handled by `rqmd_core::llm::pull::pull_models`.

use anyhow::{Context, Result};
use rqmd_core::llm::pull::pull_models;
use rqmd_core::llm::types::PullOptions;

use crate::cli::PullArgs;
use crate::color::Palette;
use crate::format_helpers::format_bytes;
use crate::state::IndexState;

pub async fn run(args: PullArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let uris = state.resolved_model_uris()?;
    let models = vec![uris.embed, uris.generate, uris.rerank];

    eprintln!("{}Pulling models{}", p.bold(), p.reset());
    let opts = PullOptions {
        refresh: args.refresh,
        cache_dir: None,
    };

    // `pull_models` is sync (hf-hub uses blocking ureq); offload to a blocking
    // task so we don't park a tokio worker thread for the full download.
    // Mirrors the pattern in `LlamaCpp::load_model`.
    //
    // We preserve the underlying error chain via `?` + `.context()` so that
    // `main.rs`'s `{err:#}` formatting can surface specific guidance like
    // `Error::InvalidGguf { looks_like_html, .. }` from `rqmd-core/src/llm/pull.rs`.
    let results = tokio::task::spawn_blocking(move || pull_models(&models, &opts))
        .await
        .context("pull task panicked")?
        .context("pull failed")?;

    for r in results {
        let note = if r.refreshed {
            "refreshed"
        } else {
            "cached/checked"
        };
        println!(
            "{}-{} {} -> {} ({}, {})",
            p.dim(),
            p.reset(),
            r.model,
            r.path.display(),
            format_bytes(r.size_bytes),
            note,
        );
    }
    Ok(())
}
