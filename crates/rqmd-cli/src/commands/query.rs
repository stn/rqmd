//! `rqmd query` — hybrid search (BM25 + vector + LLM expansion + reranking).
//!
//! Maps to qmd's `querySearch` CLI handler (`src/cli/qmd.ts` lines 2494–2630)
//! plus `hybridQuery` in `src/store.ts` lines 4272–4553. Supports qmd's
//! structured query documents (`lex:`/`vec:`/`hyde:`/`intent:`/`expand:`
//! prefixes); structured input routes to `structured_search`, plain or
//! `expand:` input falls through to `hybrid_query`.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rqmd_core::Store;
use rqmd_core::store_ops::{
    hybrid_query, structured_search, ExpandedQuery, ExpandedQueryType, HybridQueryOptions,
    HybridQueryResult, SearchHooks, StructuredSearchOptions,
};
use rqmd_core::llm::traits::Llm;
use serde_json::json;

use crate::cli::QueryArgs;
use crate::collection_filter::{filter_by_collections, resolve_collection_filter, single_collection};
use crate::color::Palette;
use crate::output::OutputFormat;
use crate::search_view::{hybrid_result_to_hit, print_hits, ExplainView, Hit};
use crate::state::IndexState;

pub async fn run(args: QueryArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let q = args.query.join(" ");
    let fmt = OutputFormat::from(&args.format);

    // Detect structured query syntax before acquiring the LLM/store so a parse
    // error fails fast. (TS parses inside `withLLMSession` after the store/health
    // checks, qmd.ts:2502-2509 — deliberate ordering deviation, no LLM start on
    // a malformed query.)
    let parsed = parse_structured_query(&q)?;
    // The `--intent` flag wins over a parsed `intent:` line
    // (TS: `opts.intent || parsed?.intent`, qmd.ts:2507).
    let intent = args
        .intent
        .clone()
        .or_else(|| parsed.as_ref().and_then(|pq| pq.intent.clone()));

    let limit = Some(if args.flags.all {
        500
    } else {
        args.flags.limit.unwrap_or(10)
    });
    let min_score = Some(args.flags.min_score.unwrap_or(0.0));

    // Resolve the collection filter (TS `resolveCollectionFilter(opts.collection,
    // true)`, qmd.ts:2499) before borrowing the LLM/store — the helper returns an
    // owned Vec so the `&mut Config` borrow is released here. `-c` omitted →
    // default collections; explicit names are validated.
    let collection_names =
        resolve_collection_filter(state.config_mut()?, &args.flags.collection, true)?;
    let single = single_collection(&collection_names);

    let llm = state.llama_cpp()?;
    let store: &Store = state.store_mut()?;
    let llm_dyn: Arc<dyn Llm> = llm;

    let results = if let Some(pq) = &parsed {
        log_structured_summary(pq, intent.as_deref(), fmt, p);
        structured_search(
            store,
            llm_dyn,
            &pq.searches,
            StructuredSearchOptions {
                // Only the single-collection case filters at the DB level; with
                // multiple collections we search all and post-filter below
                // (TS: `singleCollection ? [singleCollection] : undefined`).
                collections: single.clone().map(|c| vec![c]),
                limit,
                min_score,
                candidate_limit: args.candidate_limit,
                explain: args.explain,
                intent,
                skip_rerank: args.no_rerank,
                chunk_strategy: args.chunk_strategy.map(Into::into),
                hooks: build_structured_hooks(fmt),
            },
        )
        .await
        .context("structured query failed")?
    } else {
        // Plain or single `expand:` query. Pass the original `q` unchanged — TS
        // does NOT strip the `expand:` prefix before `hybridQuery` (qmd.ts:2557),
        // so we reproduce that for parity (known latent quirk).
        let opts = HybridQueryOptions {
            collection: single.clone(),
            limit,
            min_score,
            candidate_limit: args.candidate_limit,
            explain: args.explain,
            intent,
            skip_rerank: args.no_rerank,
            chunk_strategy: args.chunk_strategy.map(Into::into),
            hooks: build_query_hooks(fmt),
        };
        hybrid_query(store, llm_dyn, &q, opts)
            .await
            .context("query failed")?
    };

    // Narrow to the requested collections when more than one is in play
    // (TS `filterByCollections`, qmd.ts:2597-2602). No-op for 0/1 collection.
    let results = filter_by_collections(results, &collection_names, |r| r.file.as_str());

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

/// Parsed multi-line structured query document. Mirrors TS
/// `ParsedStructuredQuery` (`qmd.ts:2317-2320`).
#[derive(Debug)]
struct ParsedStructuredQuery {
    searches: Vec<ExpandedQuery>,
    intent: Option<String>,
}

/// Port of `parseStructuredQuery` (`qmd.ts:2322-2390`). Returns `None` for a
/// plain single-line query or a single `expand:` line (both route through
/// `hybrid_query`); `Some` when the document contains `lex:`/`vec:`/`hyde:`
/// lines. Validation errors mirror the TS messages.
///
/// This is the `qmd.ts` variant (handles `expand:`, uses "query document"
/// wording). The `bench.rs` copy is the separate `bench.ts` variant — both
/// mirror the duplication in the TS source.
fn parse_structured_query(query: &str) -> Result<Option<ParsedStructuredQuery>> {
    // (1-based line number, trimmed text) for non-blank lines.
    let lines: Vec<(usize, &str)> = query
        .split('\n')
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim()))
        .filter(|(_, t)| !t.is_empty())
        .collect();

    if lines.is_empty() {
        return Ok(None);
    }

    let mut searches: Vec<ExpandedQuery> = Vec::new();
    let mut intent: Option<String> = None;

    for (number, trimmed) in &lines {
        let lower = trimmed.to_lowercase();

        // `expand:` — a single standalone expand query; route through hybrid.
        // The mix check precedes the empty-text check (qmd.ts:2339-2345).
        if lower.starts_with("expand:") {
            if lines.len() > 1 {
                bail!("Line {number} starts with expand:, but query documents cannot mix expand with typed lines. Submit a single expand query instead.");
            }
            let text = trimmed["expand:".len()..].trim();
            if text.is_empty() {
                bail!("expand: query must include text.");
            }
            return Ok(None); // treat as standalone expand query
        }

        // `intent:` — optional domain hint; at most one per document.
        if lower.starts_with("intent:") {
            if intent.is_some() {
                bail!("Line {number}: only one intent: line is allowed per query document.");
            }
            let text = trimmed["intent:".len()..].trim();
            if text.is_empty() {
                bail!("Line {number}: intent: must include text.");
            }
            intent = Some(text.to_string());
            continue;
        }

        // `lex:` / `vec:` / `hyde:` typed search lines.
        if let Some((type_, prefix)) = match_prefix(&lower) {
            let text = trimmed[prefix.len()..].trim();
            if text.is_empty() {
                let name = &prefix[..prefix.len() - 1]; // drop the ':'
                bail!("Line {number} ({name}:) must include text.");
            }
            // Defensive parity with TS (qmd.ts:2369-2371). After line-split +
            // trim a typed line can only carry an embedded `\r`, but keep the
            // check + message faithful to the source.
            if text.contains(['\r', '\n']) {
                let name = &prefix[..prefix.len() - 1];
                bail!("Line {number} ({name}:) contains a newline. Keep each query on a single line.");
            }
            searches.push(ExpandedQuery {
                type_,
                query: text.to_string(),
                line: Some(*number),
            });
            continue;
        }

        // A lone plain line is a plain query, not structured.
        if lines.len() == 1 {
            return Ok(None);
        }

        bail!("Line {number} is missing a lex:/vec:/hyde:/intent: prefix. Each line in a query document must start with one.");
    }

    // `intent:` alone is not a valid query — must have at least one search.
    if intent.is_some() && searches.is_empty() {
        bail!("intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.");
    }

    Ok(if searches.is_empty() {
        None
    } else {
        Some(ParsedStructuredQuery { searches, intent })
    })
}

/// Match a `lex:` / `vec:` / `hyde:` prefix on an already-lowercased line.
fn match_prefix(lower: &str) -> Option<(ExpandedQueryType, &'static str)> {
    if lower.starts_with("lex:") {
        Some((ExpandedQueryType::Lex, "lex:"))
    } else if lower.starts_with("vec:") {
        Some((ExpandedQueryType::Vec, "vec:"))
    } else if lower.starts_with("hyde:") {
        Some((ExpandedQueryType::Hyde, "hyde:"))
    } else {
        None
    }
}

/// Lowercase label for a query type. rqmd-core's `type_name`
/// (`structured.rs`) is private; a 3-arm match here avoids a re-export for one
/// CLI label.
fn type_label(t: ExpandedQueryType) -> &'static str {
    match t {
        ExpandedQueryType::Lex => "lex",
        ExpandedQueryType::Vec => "vec",
        ExpandedQueryType::Hyde => "hyde",
    }
}

/// CLI-mode-only structured-search header on stderr (qmd.ts:2515-2527).
fn log_structured_summary(
    pq: &ParsedStructuredQuery,
    intent: Option<&str>,
    fmt: OutputFormat,
    p: &Palette,
) {
    if fmt != OutputFormat::Cli {
        return;
    }
    let labels: Vec<&str> = pq.searches.iter().map(|s| type_label(s.type_)).collect();
    eprintln!(
        "{}Structured search: {} queries ({}){}",
        p.dim(),
        pq.searches.len(),
        labels.join("+"),
        p.reset()
    );
    if let Some(i) = intent {
        eprintln!("{}├─ intent: {i}{}", p.dim(), p.reset());
    }
    for s in &pq.searches {
        let oneline = s.query.replace('\n', " ");
        // TS truncates with `substring(0, 69) + "..."` over UTF-16 code units;
        // `chars()` is the multi-byte-safe analog (identical for ASCII input).
        let preview = if oneline.chars().count() > 72 {
            format!("{}...", oneline.chars().take(69).collect::<String>())
        } else {
            oneline
        };
        eprintln!("{}├─ {}: {preview}{}", p.dim(), type_label(s.type_), p.reset());
    }
    eprintln!("{}└─ Searching...{}", p.dim(), p.reset());
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

/// Like [`build_query_hooks`] but without the expansion/strong-signal hooks.
/// `structured_search` calls `on_expand("", &[], 0)` (structured.rs:210-213);
/// TS's structured hook set omits `onExpand` (qmd.ts:2538-2553), so dropping it
/// avoids a spurious `Expanded "" -> 0 queries` line.
fn build_structured_hooks(fmt: OutputFormat) -> SearchHooks {
    if fmt != OutputFormat::Cli {
        return SearchHooks::default();
    }
    SearchHooks {
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
        ..SearchHooks::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(q: &str) -> Result<Option<ParsedStructuredQuery>> {
        parse_structured_query(q)
    }

    #[test]
    fn plain_single_line_is_not_structured() {
        assert!(parse("CAP theorem").unwrap().is_none());
    }

    #[test]
    fn blank_query_is_not_structured() {
        assert!(parse("   \n  ").unwrap().is_none());
    }

    #[test]
    fn single_expand_is_not_structured() {
        assert!(parse("expand: auth stuff").unwrap().is_none());
    }

    #[test]
    fn expand_mixed_with_typed_errors() {
        let err = parse("expand: question\nlex: keywords").unwrap_err();
        assert!(err.to_string().contains("cannot mix expand"));
    }

    #[test]
    fn expand_mixed_with_intent_errors() {
        let err = parse("intent: web\nexpand: performance").unwrap_err();
        assert!(err.to_string().contains("cannot mix expand"));
    }

    #[test]
    fn empty_expand_errors() {
        let err = parse("expand:   ").unwrap_err();
        assert!(err
            .to_string()
            .contains("expand: query must include text"));
    }

    #[test]
    fn empty_expand_with_typed_reports_mix_first() {
        // Mix check precedes the empty-text check (qmd.ts:2339-2345).
        let err = parse("expand:\nlex: x").unwrap_err();
        assert!(err.to_string().contains("cannot mix expand"));
    }

    #[test]
    fn parses_prefixes_and_intent() {
        let parsed = parse("lex: auth\nvec: secure sessions\nintent: security")
            .unwrap()
            .expect("structured");
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.searches[0].type_, ExpandedQueryType::Lex);
        assert_eq!(parsed.searches[0].query, "auth");
        assert_eq!(parsed.searches[1].type_, ExpandedQueryType::Vec);
        assert_eq!(parsed.searches[1].query, "secure sessions");
        assert_eq!(parsed.intent.as_deref(), Some("security"));
    }

    #[test]
    fn intent_after_typed_lines() {
        let parsed = parse("lex: performance\nintent: web page load times\nvec: latency")
            .unwrap()
            .expect("structured");
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.intent.as_deref(), Some("web page load times"));
    }

    #[test]
    fn prefix_is_case_insensitive() {
        let parsed = parse("HYDE: expanded text").unwrap().expect("structured");
        assert_eq!(parsed.searches[0].type_, ExpandedQueryType::Hyde);
        assert_eq!(parsed.searches[0].query, "expanded text");
    }

    #[test]
    fn intent_is_case_insensitive() {
        let parsed = parse("Intent: foo\nlex: bar").unwrap().expect("structured");
        assert_eq!(parsed.intent.as_deref(), Some("foo"));
    }

    #[test]
    fn duplicate_intent_errors() {
        let err = parse("intent: a\nintent: b").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("only one intent:"));
        assert!(msg.contains("query document"));
    }

    #[test]
    fn intent_only_errors() {
        let err = parse("intent: security").unwrap_err();
        assert!(err.to_string().contains("intent: cannot appear alone"));
    }

    #[test]
    fn empty_intent_errors() {
        let err = parse("intent:   \nlex: x").unwrap_err();
        assert!(err.to_string().contains("intent: must include text"));
    }

    #[test]
    fn multiline_missing_prefix_errors() {
        let err = parse("lex: auth\nplain line").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("missing a lex:/vec:/hyde:/intent: prefix"));
        assert!(msg.contains("query document"));
    }

    #[test]
    fn empty_prefix_text_errors() {
        let err = parse("lex:   \nvec: x").unwrap_err();
        assert!(err.to_string().contains("(lex:) must include text"));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let parsed = parse("\nlex: auth\n\nvec: sessions\n")
            .unwrap()
            .expect("structured");
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.searches[0].line, Some(2));
        assert_eq!(parsed.searches[1].line, Some(4));
    }

    #[test]
    fn colon_in_text_is_preserved() {
        let parsed = parse("lex: time: 12:30 PM").unwrap().expect("structured");
        assert_eq!(parsed.searches[0].query, "time: 12:30 PM");
    }

    #[test]
    fn surrounding_whitespace_trimmed() {
        let parsed = parse("  lex:   spaced query  ").unwrap().expect("structured");
        assert_eq!(parsed.searches[0].query, "spaced query");
    }
}
