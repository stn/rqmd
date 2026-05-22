//! `rqmd search` — BM25 full-text search. Sync because no LLM is involved.
//!
//! Maps to qmd's `search` CLI handler (`src/cli/qmd.ts` lines 2392–2426).

use std::io::IsTerminal;

use anyhow::Result;
use rqmd_core::store::search::search_fts;
use rqmd_core::store::virtual_path::resolve_virtual_path;

use crate::cli::SearchArgs;
use crate::collection_filter::{
    filter_by_collections, resolve_collection_filter, single_collection,
};
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{CliLinkCtx, editor_uri_template, print_hits, search_result_to_hit};
use crate::state::IndexState;

pub fn run(args: SearchArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let q = args.query.join(" ");
    // Empty query: `search_fts` returns Ok(vec![]) via `build_fts5_query`'s
    // None path, so we deliberately do NOT bail — `rqmd search "$VAR" --json`
    // with an empty $VAR should produce `[]`, not error.

    let min_score = args.flags.min_score.unwrap_or(0.0);
    let fmt = OutputFormat::from(&args.format);

    // Active index name for `?index=` link annotation (qmd `getActiveIndexName`).
    // Captured as an owned String before borrowing the store, then `idx` is the
    // non-default name (or None) passed to the hit constructor.
    let index_name = state.index_name().to_string();
    let idx = (index_name != "index").then_some(index_name.as_str());

    // `-c` omitted → default collections; explicit names validated. Resolve
    // before borrowing the store (the helper returns an owned Vec, releasing the
    // `&mut Config` borrow). TS `resolveCollectionFilter(opts.collection, true)`,
    // qmd.ts:2397.
    let collection_names =
        resolve_collection_filter(state.config_mut()?, &args.flags.collection, true)?;
    let collection = single_collection(&collection_names);

    // Clickable-link context for the cli format (resolved before the store
    // borrow; `editor_uri` comes from env/config, see `editor_uri_template`).
    let link = CliLinkCtx {
        editor_template: editor_uri_template(state.config_mut()?.data().editor_uri.as_deref()),
        stdout_tty: std::io::stdout().is_terminal(),
    };

    // Over-fetch so the multi-collection post-filter still has enough rows to
    // fill `display_limit`, then truncate — mirrors TS qmd.ts:2401 plus the
    // `slice(0, opts.limit)` in `outputResults` (qmd.ts:2098). `--all` sets the
    // limit to 100_000 (qmd.ts:2730), i.e. effectively unbounded.
    let display_limit = if args.flags.all {
        100_000
    } else {
        args.flags.limit.unwrap_or(20)
    };
    let fetch_limit = if args.flags.all {
        100_000
    } else {
        std::cmp::max(50, display_limit.saturating_mul(2))
    };

    let store = state.store_mut()?;
    let results = store
        .with_connection(|conn| search_fts(conn, &q, Some(fetch_limit), collection.as_deref()))?;

    // Multi-collection post-filter (TS `filterByCollections`, qmd.ts:2402-2405);
    // no-op for 0/1 collection. `SearchResult` carries its path at `doc.filepath`.
    let results = filter_by_collections(results, &collection_names, |r| r.doc.filepath.as_str());

    // TS `outputResults`: `filter(score >= minScore).slice(0, limit)`.
    let mut hits: Vec<_> = results
        .iter()
        .filter(|r| r.score >= min_score)
        .take(display_limit)
        .map(|r| search_result_to_hit(r, &q, None, args.flags.full, idx))
        .collect();

    // Resolve absolute paths for the cli format's OSC-8 links while the
    // connection is open (only needed on a TTY; non-TTY uses the plain link).
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
