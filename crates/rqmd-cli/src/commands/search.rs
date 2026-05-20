//! `rqmd search` — BM25 full-text search. Sync because no LLM is involved.
//!
//! Maps to qmd's `search` CLI handler (`src/cli/qmd.ts` lines 2392–2426).

use anyhow::{bail, Result};
use rqmd_core::store::search::search_fts;

use crate::cli::{SearchArgs, SearchFlags};
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{print_hits, search_result_to_hit};
use crate::state::IndexState;

pub fn run(args: SearchArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    if args.flags.collection.len() > 1 {
        bail!(
            "search accepts at most one --collection (got {})",
            args.flags.collection.len()
        );
    }
    let collection = args.flags.collection.first().cloned();
    let q = args.query.join(" ");
    // Empty query: `search_fts` returns Ok(vec![]) via `build_fts5_query`'s
    // None path, so we deliberately do NOT bail — `rqmd search "$VAR" --json`
    // with an empty $VAR should produce `[]`, not error.

    let limit = effective_limit(&args.flags, 20);
    let min_score = args.flags.min_score.unwrap_or(0.0);

    let store = state.store_mut()?;
    let results =
        store.with_connection(|conn| search_fts(conn, &q, Some(limit), collection.as_deref()))?;

    let hits: Vec<_> = results
        .iter()
        .filter(|r| r.score >= min_score)
        .map(|r| search_result_to_hit(r, &q, None, args.flags.full))
        .collect();

    let fmt = OutputFormat::from(&args.format);
    print_hits(&hits, fmt, p, args.flags.line_numbers)?;
    Ok(())
}

fn effective_limit(f: &SearchFlags, default: usize) -> usize {
    if f.all {
        100_000
    } else {
        f.limit.unwrap_or(default)
    }
}
