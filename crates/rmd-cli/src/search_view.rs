//! Presentation types for `search` / `vsearch` / `query`.
//!
//! Lives outside `commands/` because future consumers (rmd-mcp, future
//! `csv/md/xml/files` formatters) need the same shape. The three input result
//! types differ in fields and pre-extraction state — this module normalises
//! them into a single owned [`Hit`] with snippet + 1-indexed line.

use serde::Serialize;

use rmd_core::store::rrf::{HybridQueryExplain, QueryType, RRFContributionTrace, RRFExplain};
use rmd_core::store::search::{SearchResult, SearchSource};
use rmd_core::store::snippet::{add_line_numbers, extract_snippet};
use rmd_llm::store_ops::{HybridQueryResult, VectorSearchResult};

use crate::color::Palette;

/// Normalised search hit. Owned because two of three input types have
/// nested / `Option<String>` body shapes and lifetimes get complicated.
#[derive(Debug, Serialize)]
pub struct Hit {
    pub file: String,
    pub display_path: String,
    pub title: String,
    pub score: f64,
    /// 1-indexed line in `body` where the snippet starts.
    pub line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub snippet: String,
    /// Populated only when `--full` was passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    pub docid: String,
}

pub fn search_result_to_hit(
    r: &SearchResult,
    query: &str,
    intent: Option<&str>,
    full: bool,
) -> Hit {
    let body = r.doc.body.as_deref().unwrap_or("");
    let snip = extract_snippet(
        body,
        query,
        None,
        r.chunk_pos.map(|p| p as usize),
        None,
        intent,
    );
    Hit {
        file: r.doc.filepath.clone(),
        display_path: r.doc.display_path.clone(),
        title: r.doc.title.clone(),
        score: r.score,
        line: snip.line,
        context: r.doc.context.clone(),
        snippet: snip.snippet,
        body: full.then(|| body.to_string()),
        docid: r.doc.docid.clone(),
    }
}

pub fn vector_result_to_hit(
    r: &VectorSearchResult,
    query: &str,
    intent: Option<&str>,
    full: bool,
) -> Hit {
    let snip = extract_snippet(&r.body, query, None, None, None, intent);
    Hit {
        file: r.file.clone(),
        display_path: r.display_path.clone(),
        title: r.title.clone(),
        score: r.score,
        line: snip.line,
        context: r.context.clone(),
        snippet: snip.snippet,
        body: full.then(|| r.body.clone()),
        docid: r.docid.clone(),
    }
}

pub fn hybrid_result_to_hit(r: &HybridQueryResult, full: bool) -> Hit {
    Hit {
        file: r.file.clone(),
        display_path: r.display_path.clone(),
        title: r.title.clone(),
        score: r.score,
        line: line_at_byte_offset(&r.body, r.best_chunk_pos),
        context: r.context.clone(),
        snippet: r.best_chunk.clone(),
        body: full.then(|| r.body.clone()),
        docid: r.docid.clone(),
    }
}

/// 1-indexed line at a byte offset in `body`. `'\n'` is ASCII so byte-slice +
/// `matches('\n')` is safe even on multi-byte text. Returns 1 for `byte_pos == 0`.
pub(crate) fn line_at_byte_offset(body: &str, byte_pos: usize) -> usize {
    if byte_pos == 0 {
        return 1;
    }
    let clamp = byte_pos.min(body.len());
    body[..clamp].matches('\n').count() + 1
}

pub fn print_hits_cli(hits: &[Hit], p: &Palette, line_numbers: bool) {
    if hits.is_empty() {
        eprintln!("{}No results.{}", p.dim(), p.reset());
        return;
    }
    for (i, h) in hits.iter().enumerate() {
        println!(
            "{}{}.{} {}{}{}:{} {}({:.3}){}",
            p.bold(),
            i + 1,
            p.reset(),
            p.cyan(),
            h.display_path,
            p.reset(),
            h.line,
            p.dim(),
            h.score,
            p.reset(),
        );
        if !h.title.is_empty() && h.title != h.display_path {
            println!("   {}{}{}", p.dim(), h.title, p.reset());
        }
        if let Some(ctx) = &h.context {
            println!("   {}context:{} {ctx}", p.dim(), p.reset());
        }
        let body_to_show = h.body.as_deref().unwrap_or(&h.snippet);
        let rendered = if line_numbers {
            add_line_numbers(body_to_show, Some(h.line))
        } else {
            body_to_show.to_string()
        };
        for line in rendered.lines() {
            println!("   {line}");
        }
        println!();
    }
}

pub fn print_hits_json(hits: &[Hit]) -> anyhow::Result<()> {
    let s = serde_json::to_string_pretty(hits)?;
    println!("{s}");
    Ok(())
}

// ============================================================================
// Explain JSON wrapper for `query --explain --json`
// ============================================================================

/// CLI-side `Serialize` wrapper for [`HybridQueryExplain`] (rmd-core types
/// don't derive Serialize). Re-defining the shape here means a field rename
/// upstream surfaces as a compile error rather than silently broken JSON.
#[derive(Debug, Serialize)]
pub struct ExplainView<'a> {
    pub file: &'a str,
    pub fts_scores: &'a [f64],
    pub vector_scores: &'a [f64],
    pub rrf: RrfExplainView<'a>,
    pub rerank_score: f64,
    pub blended_score: f64,
}

#[derive(Debug, Serialize)]
pub struct RrfExplainView<'a> {
    pub rank: usize,
    pub position_score: f64,
    pub weight: f64,
    pub base_score: f64,
    pub top_rank_bonus: f64,
    pub total_score: f64,
    pub contributions: Vec<ContributionView<'a>>,
}

#[derive(Debug, Serialize)]
pub struct ContributionView<'a> {
    pub list_index: usize,
    pub source: &'static str,
    pub query_type: &'static str,
    pub query: &'a str,
    pub rank: usize,
    pub weight: f64,
    pub backend_score: f64,
    pub rrf_contribution: f64,
}

impl<'a> ExplainView<'a> {
    pub fn new(file: &'a str, e: &'a HybridQueryExplain) -> Self {
        Self {
            file,
            fts_scores: &e.fts_scores,
            vector_scores: &e.vector_scores,
            rrf: RrfExplainView::from_ref(&e.rrf),
            rerank_score: e.rerank_score,
            blended_score: e.blended_score,
        }
    }
}

impl<'a> RrfExplainView<'a> {
    fn from_ref(e: &'a RRFExplain) -> Self {
        Self {
            rank: e.rank,
            position_score: e.position_score,
            weight: e.weight,
            base_score: e.base_score,
            top_rank_bonus: e.top_rank_bonus,
            total_score: e.total_score,
            contributions: e
                .contributions
                .iter()
                .map(ContributionView::from_ref)
                .collect(),
        }
    }
}

impl<'a> ContributionView<'a> {
    fn from_ref(c: &'a RRFContributionTrace) -> Self {
        Self {
            list_index: c.list_index,
            source: source_str(c.source),
            query_type: query_type_str(c.query_type),
            query: &c.query,
            rank: c.rank,
            weight: c.weight,
            backend_score: c.backend_score,
            rrf_contribution: c.rrf_contribution,
        }
    }
}

fn source_str(s: SearchSource) -> &'static str {
    match s {
        SearchSource::Fts => "fts",
        SearchSource::Vec => "vec",
    }
}

fn query_type_str(t: QueryType) -> &'static str {
    match t {
        QueryType::Original => "original",
        QueryType::Lex => "lex",
        QueryType::Vec => "vec",
        QueryType::Hyde => "hyde",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_at_byte_offset_basics() {
        assert_eq!(line_at_byte_offset("", 0), 1);
        assert_eq!(line_at_byte_offset("abc\ndef", 0), 1);
        assert_eq!(line_at_byte_offset("abc\ndef", 4), 2);
        assert_eq!(line_at_byte_offset("a\nb\nc", 4), 3);
        // Multi-byte safe: pos points just after a multi-byte char (3-byte CJK).
        // "日本\nabc" — byte length of "日本" is 6, '\n' at byte 6.
        assert_eq!(line_at_byte_offset("日本\nabc", 7), 2);
        // Out-of-bounds is clamped.
        assert_eq!(line_at_byte_offset("a\nb", 100), 2);
    }
}
