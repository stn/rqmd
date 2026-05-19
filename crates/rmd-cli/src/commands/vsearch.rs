//! `rmd vsearch` — vector similarity search with automatic query expansion.
//!
//! Maps to qmd's `vectorSearch` CLI handler (`src/cli/qmd.ts` lines 2443–2492)
//! plus `vectorSearchQuery` in `src/store.ts` lines 4582–4629.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rmd_core::Store;
use rmd_core::store_ops::{
    vector_search_query, ExpandedQuery, SearchHooks, VectorSearchOptions,
};
use rmd_core::llm::traits::Llm;

use crate::cli::VsearchArgs;
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{print_hits, vector_result_to_hit};
use crate::state::IndexState;

pub async fn run(args: VsearchArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    if args.flags.collection.len() > 1 {
        bail!(
            "vsearch accepts at most one --collection (got {})",
            args.flags.collection.len()
        );
    }
    let q = args.query.join(" ");
    let fmt = OutputFormat::from(&args.format);

    // Borrow order matters: take Arc<LlamaCpp> first (consumes &mut self only
    // for the call), then re-borrow store_mut's &mut Store as &Store for the
    // immutable `vector_search_query` API.
    let llm = state.llama_cpp()?;
    let store: &Store = state.store_mut()?;

    let opts = VectorSearchOptions {
        collection: args.flags.collection.first().cloned(),
        limit: Some(if args.flags.all {
            500
        } else {
            args.flags.limit.unwrap_or(10)
        }),
        min_score: Some(args.flags.min_score.unwrap_or(0.3)),
        intent: args.intent.clone(),
        hooks: build_vsearch_hooks(fmt),
    };

    // Arc<LlamaCpp> -> Arc<dyn Llm> via let-binding unsized coercion.
    // Fallback if inference fails: `Arc::clone(&llm) as Arc<dyn Llm>`.
    let llm_dyn: Arc<dyn Llm> = llm;
    let results = vector_search_query(store, llm_dyn, &q, opts)
        .await
        .context("vsearch failed")?;

    let hits: Vec<_> = results
        .iter()
        .map(|r| vector_result_to_hit(r, &q, args.intent.as_deref(), args.flags.full))
        .collect();

    print_hits(&hits, fmt, p, args.flags.line_numbers)?;
    Ok(())
}

/// `vector_search_query` only fires `on_expand`; other hooks would be ignored.
/// See `crates/rmd-core/src/store_ops/vector_search.rs:32-34`.
/// Verbose logging only in the human CLI mode — any machine-readable format
/// runs silently so stderr stays clean.
fn build_vsearch_hooks(fmt: OutputFormat) -> SearchHooks {
    if fmt != OutputFormat::Cli {
        return SearchHooks::default();
    }
    SearchHooks {
        on_expand: Some(Arc::new(|orig, expanded: &[ExpandedQuery], ms| {
            eprintln!(
                "Expanded \"{orig}\" -> {} queries ({ms}ms)",
                expanded.len()
            );
            for e in expanded {
                eprintln!("  [{:?}] {}", e.type_, e.query);
            }
        })),
        ..Default::default()
    }
}
