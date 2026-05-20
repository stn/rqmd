//! `rqmd bench` — run a fixture of labelled queries through the four search
//! backends (bm25 / vector / hybrid / full) and report precision@k, recall,
//! recall@1/3/5, MRR, F1, and latency.
//!
//! Port of qmd's `src/bench/bench.ts` (the runner half; pure scoring + the
//! fixture/result types live in `rqmd_core::bench`).

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;

use rqmd_core::bench::{
    score_results, BackendResult, BenchmarkFixture, BenchmarkQuery, BenchmarkResult, QueryResult,
    SummaryStats,
};
use rqmd_core::llm::config::resolve_embed_model;
use rqmd_core::llm::traits::Llm;
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::search::search_fts;
use rqmd_core::store_ops::{
    hybrid_query, search_vec, structured_search, ExpandedQuery, ExpandedQueryType,
    HybridQueryOptions, StructuredSearchOptions,
};
use rqmd_core::Store;

use crate::cli::BenchArgs;
use crate::state::IndexState;

/// Backends run for every query, in fixed order (also the JSON key order).
const BACKENDS: [&str; 4] = ["bm25", "vector", "hybrid", "full"];

pub async fn run(args: BenchArgs, state: &mut IndexState) -> Result<()> {
    // Load fixture. serde already rejects a fixture missing `queries`
    // (non-optional field) as a parse error.
    let raw = std::fs::read_to_string(&args.fixture)
        .with_context(|| format!("reading fixture {}", args.fixture))?;
    let fixture: BenchmarkFixture = serde_json::from_str(&raw)
        .with_context(|| format!("parsing fixture {}", args.fixture))?;

    let json = args.json;
    // CLI flag overrides the fixture's default collection.
    let collection = args.collection.clone().or_else(|| fixture.collection.clone());
    let embed_model = resolve_embed_model(None);

    // Borrow order (see vsearch.rs): take the owned `Arc<LlamaCpp>` first, then
    // re-borrow `&Store`. All backend fns take `&Store`, so the immutable borrow
    // is held across the whole loop without conflict.
    let llm = state.llama_cpp()?;
    let store: &Store = state.store_mut()?;
    let llm_dyn: Arc<dyn Llm> = llm;

    let mut results: Vec<QueryResult> = Vec::new();
    for query in &fixture.queries {
        let mut backends: IndexMap<String, BackendResult> = IndexMap::new();
        for backend in BACKENDS {
            if !json {
                eprint!("  {} / {}...", query.id, backend);
            }
            let br = run_query(
                store,
                &llm_dyn,
                backend,
                query,
                collection.as_deref(),
                &embed_model,
            )
            .await;
            if !json {
                eprintln!(" {}ms", br.latency_ms);
            }
            backends.insert(backend.to_string(), br);
        }
        results.push(QueryResult {
            id: query.id.clone(),
            query: query.query.clone(),
            r#type: query.r#type.clone(),
            backends,
        });
    }

    let summary = compute_summary(&results);
    let bench_result = BenchmarkResult {
        timestamp: make_timestamp(),
        fixture: args.fixture.clone(),
        results,
        summary,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&bench_result)?);
    } else {
        println!("\n{}", format_table(&bench_result.results));
        println!("Summary:");
        println!("{}", "-".repeat(70));
        for (name, s) in &bench_result.summary {
            println!(
                "  {} P@k={} R@1={} R@3={} R@5={} MRR={} F1={} Avg={}ms",
                pad(name, 8),
                num3(s.avg_precision),
                num3(s.avg_recall_at_1),
                num3(s.avg_recall_at_3),
                num3(s.avg_recall_at_5),
                num3(s.avg_mrr),
                num3(s.avg_f1),
                s.avg_latency_ms.round() as i64,
            );
        }
    }

    Ok(())
}

/// One query against one backend, with latency + scoring. A backend error
/// (e.g. vector search with no embeddings) degrades to an all-zero result
/// instead of aborting the run — mirrors the TS `runQuery` try/catch.
async fn run_query(
    store: &Store,
    llm: &Arc<dyn Llm>,
    backend: &str,
    query: &BenchmarkQuery,
    collection: Option<&str>,
    embed_model: &str,
) -> BackendResult {
    let limit = query.expected_in_top_k.max(10);
    let total_expected = query.expected_files.len();
    let start = Instant::now();
    let outcome = run_backend(backend, store, llm, query, limit, collection, embed_model).await;
    let latency_ms = start.elapsed().as_millis();

    match outcome {
        Ok(files) => {
            let scores = score_results(&files, &query.expected_files, query.expected_in_top_k);
            let top_files = files.iter().take(10).cloned().collect();
            BackendResult::from_scores(scores, total_expected, latency_ms, top_files)
        }
        Err(_) => BackendResult::zeroed(total_expected, latency_ms, query.expected_files.clone()),
    }
}

/// Run one backend, returning the ordered list of result file paths.
async fn run_backend(
    backend: &str,
    store: &Store,
    llm: &Arc<dyn Llm>,
    query: &BenchmarkQuery,
    limit: usize,
    collection: Option<&str>,
    embed_model: &str,
) -> Result<Vec<String>> {
    let structured = parse_structured_query(&query.query)?;
    match backend {
        // BM25: lex-only when structured, else the raw query.
        "bm25" => {
            if let Some(s) = &structured {
                let mut files = Vec::new();
                for sub in s.searches.iter().filter(|q| q.type_ == ExpandedQueryType::Lex) {
                    let hits = store
                        .with_connection(|c| search_fts(c, &sub.query, Some(limit), collection))?;
                    files.extend(hits.iter().map(|r| r.doc.filepath.clone()));
                }
                Ok(unique_files(files, limit))
            } else {
                let hits = store
                    .with_connection(|c| search_fts(c, &query.query, Some(limit), collection))?;
                Ok(hits.iter().map(|r| r.doc.filepath.clone()).collect())
            }
        }
        // Vector: vec/hyde-only when structured, else the raw query.
        "vector" => {
            if let Some(s) = &structured {
                let mut files = Vec::new();
                for sub in s
                    .searches
                    .iter()
                    .filter(|q| matches!(q.type_, ExpandedQueryType::Vec | ExpandedQueryType::Hyde))
                {
                    let hits =
                        search_vec(store, llm.clone(), &sub.query, embed_model, limit, collection, None)
                            .await?;
                    files.extend(hits.iter().map(|r| r.doc.filepath.clone()));
                }
                Ok(unique_files(files, limit))
            } else {
                let hits = search_vec(
                    store,
                    llm.clone(),
                    &query.query,
                    embed_model,
                    limit,
                    collection,
                    None,
                )
                .await?;
                Ok(hits.iter().map(|r| r.doc.filepath.clone()).collect())
            }
        }
        // Hybrid (RRF, no rerank) / full (with LLM rerank).
        "hybrid" | "full" => {
            let skip_rerank = backend == "hybrid";
            let hits = if let Some(s) = &structured {
                structured_search(
                    store,
                    llm.clone(),
                    &s.searches,
                    StructuredSearchOptions {
                        collections: collection.map(|c| vec![c.to_string()]),
                        limit: Some(limit),
                        intent: s.intent.clone(),
                        skip_rerank,
                        ..Default::default()
                    },
                )
                .await?
            } else {
                hybrid_query(
                    store,
                    llm.clone(),
                    &query.query,
                    HybridQueryOptions {
                        collection: collection.map(|c| c.to_string()),
                        limit: Some(limit),
                        skip_rerank,
                        ..Default::default()
                    },
                )
                .await?
            };
            Ok(hits.iter().map(|r| r.file.clone()).collect())
        }
        other => bail!("unknown backend: {other}"),
    }
}

/// A parsed multi-line structured query.
#[derive(Debug)]
struct ParsedStructuredQuery {
    searches: Vec<ExpandedQuery>,
    intent: Option<String>,
}

/// Parse a structured query (lines prefixed `lex:` / `vec:` / `hyde:` /
/// `intent:`). Returns `None` for a plain single-line query. Port of
/// `parseStructuredQuery` (`bench.ts:46-94`); errors mirror the TS messages.
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

        if lower.starts_with("intent:") {
            if intent.is_some() {
                bail!("Line {number}: only one intent: line is allowed per benchmark query.");
            }
            let text = trimmed["intent:".len()..].trim();
            if text.is_empty() {
                bail!("Line {number}: intent: must include text.");
            }
            intent = Some(text.to_string());
            continue;
        }

        if let Some((type_, prefix)) = match_prefix(&lower) {
            let text = trimmed[prefix.len()..].trim();
            if text.is_empty() {
                let name = &prefix[..prefix.len() - 1]; // drop the ':'
                bail!("Line {number} ({name}:) must include text.");
            }
            searches.push(ExpandedQuery {
                type_,
                query: text.to_string(),
                line: Some(*number),
            });
            continue;
        }

        // A lone line with no prefix is a plain query, not structured.
        if lines.len() == 1 {
            return Ok(None);
        }

        bail!("Line {number} is missing a lex:/vec:/hyde:/intent: prefix.");
    }

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

/// Deduplicate file paths (first occurrence wins) and cap at `limit`.
fn unique_files(files: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for f in files {
        if seen.insert(f.clone()) {
            out.push(f);
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

/// Per-backend averaged metrics, in first-seen backend order.
fn compute_summary(results: &[QueryResult]) -> IndexMap<String, SummaryStats> {
    let mut names: Vec<String> = Vec::new();
    for r in results {
        for name in r.backends.keys() {
            if !names.iter().any(|n| n == name) {
                names.push(name.clone());
            }
        }
    }

    let mut summary: IndexMap<String, SummaryStats> = IndexMap::new();
    for name in names {
        let (mut p, mut rc, mut r1, mut r3, mut r5, mut mrr, mut f1, mut lat) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut count = 0u32;
        for r in results {
            if let Some(br) = r.backends.get(&name) {
                p += br.precision_at_k;
                rc += br.recall;
                r1 += br.recall_at_1;
                r3 += br.recall_at_3;
                r5 += br.recall_at_5;
                mrr += br.mrr;
                f1 += br.f1;
                lat += br.latency_ms as f64;
                count += 1;
            }
        }
        if count > 0 {
            let c = f64::from(count);
            summary.insert(
                name,
                SummaryStats {
                    avg_precision: p / c,
                    avg_recall: rc / c,
                    avg_recall_at_1: r1 / c,
                    avg_recall_at_3: r3 / c,
                    avg_recall_at_5: r5 / c,
                    avg_mrr: mrr / c,
                    avg_f1: f1 / c,
                    avg_latency_ms: lat / c,
                },
            );
        }
    }
    summary
}

/// ASCII results table. Reproduces qmd's `formatTable` (`bench.ts:209-229`)
/// padding exactly: header columns padded, data metrics `{:.2}` right-padded
/// to 5, latency right-padded to 7 + `ms`.
fn format_table(results: &[QueryResult]) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "{} {} {} {} {} {} {} {} {}",
        pad("Query", 25),
        pad("Backend", 8),
        pad("P@k", 6),
        pad("R@1", 6),
        pad("R@3", 6),
        pad("R@5", 6),
        pad("MRR", 6),
        pad("F1", 6),
        pad("ms", 8),
    ));
    lines.push("-".repeat(88));
    for r in results {
        for (backend, br) in &r.backends {
            lines.push(format!(
                "{} {} {} {} {} {} {} {} {:>7}ms",
                pad(&r.id, 25),
                pad(backend, 8),
                num2(br.precision_at_k),
                num2(br.recall_at_1),
                num2(br.recall_at_3),
                num2(br.recall_at_5),
                num2(br.mrr),
                num2(br.f1),
                br.latency_ms,
            ));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

/// `s.slice(0, n).padEnd(n)` — truncate to `n` chars then right-pad with spaces.
fn pad(s: &str, n: usize) -> String {
    let truncated: String = s.chars().take(n).collect();
    let len = truncated.chars().count();
    if len >= n {
        truncated
    } else {
        format!("{truncated}{}", " ".repeat(n - len))
    }
}

/// `x.toFixed(2).padStart(5)`.
fn num2(x: f64) -> String {
    let s = format!("{x:.2}");
    if s.len() >= 5 {
        s
    } else {
        format!("{}{s}", " ".repeat(5 - s.len()))
    }
}

/// `x.toFixed(3).padStart(6)`.
fn num3(x: f64) -> String {
    let s = format!("{x:.3}");
    if s.len() >= 6 {
        s
    } else {
        format!("{}{s}", " ".repeat(6 - s.len()))
    }
}

/// `new Date().toISOString().replace(/[:.]/g, "").slice(0, 15)`.
fn make_timestamp() -> String {
    let iso = now_rfc3339(); // YYYY-MM-DDTHH:MM:SS.sssZ
    iso.chars()
        .filter(|c| *c != ':' && *c != '.')
        .take(15)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_single_line_is_not_structured() {
        assert!(parse_structured_query("API versioning").unwrap().is_none());
    }

    #[test]
    fn blank_query_is_not_structured() {
        assert!(parse_structured_query("   \n  ").unwrap().is_none());
    }

    #[test]
    fn parses_prefixes_and_intent() {
        let parsed = parse_structured_query("lex: auth\nvec: secure sessions\nintent: security")
            .unwrap()
            .expect("structured");
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.searches[0].type_, ExpandedQueryType::Lex);
        assert_eq!(parsed.searches[0].query, "auth");
        assert_eq!(parsed.searches[1].type_, ExpandedQueryType::Vec);
        assert_eq!(parsed.intent.as_deref(), Some("security"));
    }

    #[test]
    fn prefix_is_case_insensitive() {
        let parsed = parse_structured_query("HYDE: expanded text")
            .unwrap()
            .expect("structured");
        assert_eq!(parsed.searches[0].type_, ExpandedQueryType::Hyde);
        assert_eq!(parsed.searches[0].query, "expanded text");
    }

    #[test]
    fn duplicate_intent_errors() {
        let err = parse_structured_query("intent: a\nintent: b").unwrap_err();
        assert!(err.to_string().contains("only one intent:"));
    }

    #[test]
    fn intent_only_errors() {
        let err = parse_structured_query("intent: security").unwrap_err();
        assert!(err.to_string().contains("intent: cannot appear alone"));
    }

    #[test]
    fn multiline_missing_prefix_errors() {
        let err = parse_structured_query("lex: auth\nplain line").unwrap_err();
        assert!(err.to_string().contains("missing a lex:/vec:/hyde:/intent: prefix"));
    }

    #[test]
    fn empty_prefix_text_errors() {
        let err = parse_structured_query("lex:   \nvec: x").unwrap_err();
        assert!(err.to_string().contains("(lex:) must include text"));
    }

    #[test]
    fn unique_files_dedups_and_caps() {
        let files = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
        ];
        assert_eq!(unique_files(files.clone(), 2), vec!["a", "b"]);
        assert_eq!(unique_files(files, 10), vec!["a", "b", "c"]);
    }

    #[test]
    fn pad_truncates_and_fills() {
        assert_eq!(pad("abc", 5), "abc  ");
        assert_eq!(pad("abcdef", 3), "abc");
    }

    #[test]
    fn num_formatting_matches_tofixed_padstart() {
        assert_eq!(num2(1.0), " 1.00");
        assert_eq!(num2(0.5), " 0.50");
        assert_eq!(num3(1.0), " 1.000");
        assert_eq!(num3(0.333_333), " 0.333");
    }
}
