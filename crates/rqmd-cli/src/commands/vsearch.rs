//! `rqmd vsearch` — vector similarity search with automatic query expansion.
//!
//! Maps to qmd's `vectorSearch` CLI handler (`src/cli/qmd.ts` lines 2443–2492)
//! plus `vectorSearchQuery` in `src/store.ts` lines 4582–4629.

use std::io::IsTerminal;
use std::sync::Arc;

use anyhow::{Context, Result};
use rqmd_core::llm::traits::Llm;
use rqmd_core::store::virtual_path::resolve_virtual_path;
use rqmd_core::store_ops::{vector_search_query, ExpandedQuery, SearchHooks, VectorSearchOptions};
use rqmd_core::Store;

use crate::cli::VsearchArgs;
use crate::collection_filter::{filter_by_collections, resolve_collection_filter, single_collection};
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{editor_uri_template, print_hits, vector_result_to_hit, CliLinkCtx};
use crate::state::IndexState;

pub async fn run(args: VsearchArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let q = args.query.join(" ");
    let fmt = OutputFormat::from(&args.format);

    // Active index name for `?index=` link annotation (captured before the
    // store borrow). `idx` is the non-default name (or None).
    let index_name = state.index_name().to_string();
    let idx = (index_name != "index").then_some(index_name.as_str());

    // `-c` omitted → default collections; explicit names validated. Resolve
    // before borrowing the LLM/store (the helper returns an owned Vec, releasing
    // the `&mut Config` borrow). TS `resolveCollectionFilter(opts.collection,
    // true)`, qmd.ts:2448.
    let collection_names =
        resolve_collection_filter(state.config_mut()?, &args.flags.collection, true)?;
    let collection = single_collection(&collection_names);

    // Clickable-link context for the cli format (resolved before the LLM/store
    // borrow).
    let link = CliLinkCtx {
        editor_template: editor_uri_template(state.config_mut()?.data().editor_uri.as_deref()),
        stdout_tty: std::io::stdout().is_terminal(),
    };

    // Borrow order matters: take Arc<LlamaCpp> first (consumes &mut self only
    // for the call), then re-borrow store_mut's &mut Store as &Store for the
    // immutable `vector_search_query` API.
    let llm = state.llama_cpp()?;
    let store: &Store = state.store_mut()?;

    let opts = VectorSearchOptions {
        collection,
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

    // Multi-collection post-filter (TS `filterByCollections`, qmd.ts:2468-2473);
    // no-op for 0/1 collection.
    let results = filter_by_collections(results, &collection_names, |r| r.file.as_str());

    let mut hits: Vec<_> = results
        .iter()
        .map(|r| vector_result_to_hit(r, &q, args.intent.as_deref(), args.flags.full, idx))
        .collect();

    // Resolve absolute paths for the cli format's OSC-8 links (TTY only).
    if fmt == OutputFormat::Cli && link.stdout_tty {
        store.with_connection(|conn| {
            for h in &mut hits {
                h.abs_path = resolve_virtual_path(conn, &h.file)
                    .ok()
                    .flatten()
                    .map(|pp| pp.to_string_lossy().into_owned());
            }
        });
    }

    print_hits(&hits, fmt, p, args.flags.line_numbers, &q, &link)?;
    Ok(())
}

/// `vector_search_query` only fires `on_expand`; other hooks would be ignored.
/// See `crates/rqmd-core/src/store_ops/vector_search.rs:32-34`.
/// Verbose logging only in the human CLI mode — any machine-readable format
/// runs silently so stderr stays clean.
fn build_vsearch_hooks(fmt: OutputFormat) -> SearchHooks {
    if fmt != OutputFormat::Cli {
        return SearchHooks::default();
    }
    SearchHooks {
        on_expand: Some(Arc::new(|orig, expanded: &[ExpandedQuery], ms| {
            eprintln!("Expanded \"{orig}\" -> {} queries ({ms}ms)", expanded.len());
            for e in expanded {
                eprintln!("  [{:?}] {}", e.type_, e.query);
            }
        })),
        ..Default::default()
    }
}
