//! `rmd query` — hybrid search (BM25 + vector + LLM expansion + reranking).
//!
//! Maps to qmd's `querySearch` CLI handler (`src/cli/qmd.ts` lines 2494–2630)
//! plus `hybridQuery` in `src/store.ts` lines 4272–4553.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rmd_core::Store;
use rmd_core::store_ops::{
    hybrid_query, ExpandedQuery, HybridQueryOptions, HybridQueryResult, SearchHooks,
};
use rmd_core::llm::traits::Llm;
use serde_json::json;

use crate::cli::QueryArgs;
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{hybrid_result_to_hit, print_hits, ExplainView, Hit};
use crate::state::IndexState;

pub async fn run(args: QueryArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    if args.flags.collection.len() > 1 {
        bail!(
            "query accepts at most one --collection (got {})",
            args.flags.collection.len()
        );
    }
    let q = args.query.join(" ");
    let fmt = OutputFormat::from(&args.format);

    let llm = state.llama_cpp()?;
    let store: &Store = state.store_mut()?;

    let opts = HybridQueryOptions {
        collection: args.flags.collection.first().cloned(),
        limit: Some(if args.flags.all {
            500
        } else {
            args.flags.limit.unwrap_or(10)
        }),
        min_score: Some(args.flags.min_score.unwrap_or(0.0)),
        candidate_limit: args.candidate_limit,
        explain: args.explain,
        intent: args.intent.clone(),
        skip_rerank: args.no_rerank,
        chunk_strategy: args.chunk_strategy.map(Into::into),
        hooks: build_query_hooks(fmt),
    };

    let llm_dyn: Arc<dyn Llm> = llm;
    let results = hybrid_query(store, llm_dyn, &q, opts)
        .await
        .context("query failed")?;

    let hits: Vec<Hit> = results
        .iter()
        .map(|r| hybrid_result_to_hit(r, args.flags.full))
        .collect();

    // `--explain` is only honoured for JSON (full trace) and CLI (stderr
    // summary); other formats render the hits and silently drop the trace —
    // CSV/MD/XML/files have no natural slot for it.
    if fmt == OutputFormat::Json && args.explain {
        // Build a top-level object that pairs each Hit with its trace.
        // ExplainView borrows from `results`, so this stays alloc-light.
        let explains: Vec<ExplainView<'_>> = results
            .iter()
            .filter_map(|r| r.explain.as_ref().map(|e| ExplainView::new(&r.file, e)))
            .collect();
        let s = serde_json::to_string_pretty(&json!({
            "hits": hits,
            "explain": explains,
        }))?;
        println!("{s}");
    } else {
        print_hits(&hits, fmt, p, args.flags.line_numbers)?;
        if fmt == OutputFormat::Cli && args.explain {
            print_explain_summary(&results, p);
        }
    }
    Ok(())
}

/// One-line per-hit summary written to stderr (CLI mode only). Detailed
/// trace lives behind `--explain --json`.
fn print_explain_summary(results: &[HybridQueryResult], p: &Palette) {
    for r in results {
        let Some(e) = &r.explain else { continue };
        eprintln!(
            "  {}{}{}  rrf.total={:.3}  rerank={:.3}  blended={:.3}",
            p.dim(),
            r.display_path,
            p.reset(),
            e.rrf.total_score,
            e.rerank_score,
            e.blended_score,
        );
    }
}

/// Verbose progress logging only in the human CLI mode — any
/// machine-readable format runs silently so stderr stays clean.
fn build_query_hooks(fmt: OutputFormat) -> SearchHooks {
    if fmt != OutputFormat::Cli {
        return SearchHooks::default();
    }
    SearchHooks {
        on_strong_signal: Some(Arc::new(|top| {
            eprintln!("Strong BM25 signal ({top:.2}) — skipping expansion");
        })),
        on_expand_start: Some(Arc::new(|| {
            eprintln!("Expanding query...");
        })),
        on_expand: Some(Arc::new(|orig, expanded: &[ExpandedQuery], ms| {
            eprintln!(
                "Expanded \"{orig}\" -> {} queries ({ms}ms)",
                expanded.len()
            );
            for e in expanded {
                eprintln!("  [{:?}] {}", e.type_, e.query);
            }
        })),
        on_embed_start: Some(Arc::new(|n| {
            eprintln!("Embedding {n} queries...");
        })),
        on_embed_done: Some(Arc::new(|ms| {
            eprintln!("  embedded ({ms}ms)");
        })),
        on_rerank_start: Some(Arc::new(|n| {
            eprintln!("Reranking {n} candidates...");
        })),
        on_rerank_done: Some(Arc::new(|ms| {
            eprintln!("  reranked ({ms}ms)");
        })),
    }
}
