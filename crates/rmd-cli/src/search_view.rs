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
use rmd_core::store_ops::{HybridQueryResult, VectorSearchResult};

use crate::color::Palette;
use crate::output::{escape_csv, escape_xml, OutputFormat};

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

/// Selects body (with `--full`) or snippet, applying `add_line_numbers` if
/// `line_numbers` is set. Full bodies number from 1; snippets number from
/// `h.line` so the printed numbers match the source file.
fn render_content(h: &Hit, line_numbers: bool) -> String {
    let (text, start) = match &h.body {
        Some(b) => (b.as_str(), 1),
        None => (h.snippet.as_str(), h.line),
    };
    if line_numbers {
        add_line_numbers(text, Some(start))
    } else {
        text.to_string()
    }
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
        for line in render_content(h, line_numbers).lines() {
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

pub fn fmt_hits_csv(hits: &[Hit], line_numbers: bool) -> String {
    let mut out = String::from("docid,score,file,title,context,line,snippet");
    for h in hits {
        let content = render_content(h, line_numbers);
        out.push('\n');
        out.push_str(&format!(
            "#{},{:.4},{},{},{},{},{}",
            h.docid,
            h.score,
            escape_csv(&h.display_path),
            escape_csv(&h.title),
            escape_csv(h.context.as_deref().unwrap_or("")),
            h.line,
            escape_csv(&content),
        ));
    }
    out
}

pub fn print_hits_csv(hits: &[Hit], line_numbers: bool) {
    println!("{}", fmt_hits_csv(hits, line_numbers));
}

pub fn fmt_hits_files(hits: &[Hit]) -> String {
    hits.iter()
        .map(|h| {
            // `display_path` is intentionally NOT CSV-escaped — matches qmd
            // formatter.ts:164. Paths with commas/quotes pass through raw.
            if let Some(ctx) = &h.context {
                let esc = ctx.replace('"', "\"\"");
                format!("#{},{:.2},{},\"{esc}\"", h.docid, h.score, h.display_path)
            } else {
                format!("#{},{:.2},{}", h.docid, h.score, h.display_path)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn print_hits_files(hits: &[Hit]) {
    let s = fmt_hits_files(hits);
    if !s.is_empty() {
        println!("{s}");
    }
}

pub fn fmt_hits_md(hits: &[Hit], line_numbers: bool) -> String {
    hits.iter()
        .map(|h| {
            let heading = if !h.title.is_empty() {
                h.title.as_str()
            } else {
                h.display_path.as_str()
            };
            // contextLine is "" when no context — but the literal `\n` after it
            // in the template still fires, so the blank line above content is
            // always present. Matches qmd formatter.ts:188-189.
            let context_line = match &h.context {
                Some(ctx) => format!("**context:** {ctx}\n"),
                None => String::new(),
            };
            let content = render_content(h, line_numbers);
            format!(
                "---\n# {heading}\n\n**docid:** `#{docid}`\n{context_line}\n{content}\n",
                docid = h.docid,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn print_hits_md(hits: &[Hit], line_numbers: bool) {
    let s = fmt_hits_md(hits, line_numbers);
    if !s.is_empty() {
        // `print!` + explicit trailing newline matches TS `console.log(...)`
        // — sections already end with `\n`, and `console.log` adds one more.
        print!("{s}");
        println!();
    }
}

pub fn fmt_hits_xml(hits: &[Hit], line_numbers: bool) -> String {
    hits.iter()
        .map(|h| {
            let title_attr = if !h.title.is_empty() {
                format!(" title=\"{}\"", escape_xml(&h.title))
            } else {
                String::new()
            };
            let context_attr = match &h.context {
                Some(ctx) => format!(" context=\"{}\"", escape_xml(ctx)),
                None => String::new(),
            };
            let content = render_content(h, line_numbers);
            format!(
                "<file docid=\"#{docid}\" name=\"{name}\"{title_attr}{context_attr}>\n{body}\n</file>",
                docid = h.docid,
                name = escape_xml(&h.display_path),
                body = escape_xml(&content),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn print_hits_xml(hits: &[Hit], line_numbers: bool) {
    let s = fmt_hits_xml(hits, line_numbers);
    if !s.is_empty() {
        print!("{s}");
        println!();
    }
}

/// Dispatch to the right `print_hits_*` for the resolved [`OutputFormat`].
pub fn print_hits(
    hits: &[Hit],
    format: OutputFormat,
    p: &Palette,
    line_numbers: bool,
) -> anyhow::Result<()> {
    match format {
        OutputFormat::Cli => print_hits_cli(hits, p, line_numbers),
        OutputFormat::Json => print_hits_json(hits)?,
        OutputFormat::Csv => print_hits_csv(hits, line_numbers),
        OutputFormat::Md => print_hits_md(hits, line_numbers),
        OutputFormat::Xml => print_hits_xml(hits, line_numbers),
        OutputFormat::Files => print_hits_files(hits),
    }
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

    #[allow(clippy::too_many_arguments)]
    fn hit(
        docid: &str,
        display_path: &str,
        title: &str,
        context: Option<&str>,
        score: f64,
        line: usize,
        snippet: &str,
        body: Option<&str>,
    ) -> Hit {
        Hit {
            file: display_path.to_string(),
            display_path: display_path.to_string(),
            title: title.to_string(),
            score,
            line,
            context: context.map(String::from),
            snippet: snippet.to_string(),
            body: body.map(String::from),
            docid: docid.to_string(),
        }
    }

    #[test]
    fn render_content_picks_body_with_start_1_when_full() {
        let h = hit("a", "f", "", None, 0.0, 5, "snip", Some("L1\nL2"));
        assert_eq!(render_content(&h, false), "L1\nL2");
        assert_eq!(render_content(&h, true), "1: L1\n2: L2");
    }

    #[test]
    fn render_content_picks_snippet_with_start_at_hit_line() {
        let h = hit("a", "f", "", None, 0.0, 7, "X\nY", None);
        assert_eq!(render_content(&h, false), "X\nY");
        // snippet numbering anchors at h.line so printed numbers line up with
        // the source file
        assert_eq!(render_content(&h, true), "7: X\n8: Y");
    }

    #[test]
    fn csv_header_and_basic_row() {
        let hits = vec![hit(
            "abc123",
            "docs/x.md",
            "Title",
            Some("ctx"),
            0.125,
            3,
            "snip",
            None,
        )];
        let s = fmt_hits_csv(&hits, false);
        let mut lines = s.lines();
        assert_eq!(
            lines.next(),
            Some("docid,score,file,title,context,line,snippet")
        );
        assert_eq!(
            lines.next(),
            Some("#abc123,0.1250,docs/x.md,Title,ctx,3,snip")
        );
        assert_eq!(lines.next(), None);
    }

    #[test]
    fn csv_empty_emits_header_only() {
        let s = fmt_hits_csv(&[], false);
        assert_eq!(s, "docid,score,file,title,context,line,snippet");
    }

    #[test]
    fn csv_escapes_commas_quotes_newlines() {
        let hits = vec![hit(
            "x",
            "a,b.md",
            "T\"itle",
            Some("c\nd"),
            0.5,
            1,
            "s,n",
            None,
        )];
        let s = fmt_hits_csv(&hits, false);
        // Embedded newlines in an escaped cell span multiple `\n`-split lines,
        // so assert on the whole string. Commas, doubled quotes, and embedded
        // newlines are all wrapped in quotes per RFC 4180.
        assert!(s.contains("\"a,b.md\""));
        assert!(s.contains("\"T\"\"itle\""));
        assert!(s.contains("\"c\nd\""));
        assert!(s.contains("\"s,n\""));
    }

    #[test]
    fn files_with_and_without_context() {
        let hits = vec![
            hit("a", "x.md", "T", None, 0.12, 1, "s", None),
            hit("b", "y.md", "T", Some("ctx\"q"), 0.34, 1, "s", None),
        ];
        let s = fmt_hits_files(&hits);
        assert_eq!(s, "#a,0.12,x.md\n#b,0.34,y.md,\"ctx\"\"q\"");
    }

    #[test]
    fn md_with_context_has_blank_line_before_content() {
        let hits = vec![hit("a", "x.md", "Title", Some("ctx"), 0.0, 1, "BODY", None)];
        let s = fmt_hits_md(&hits, false);
        assert_eq!(
            s,
            "---\n# Title\n\n**docid:** `#a`\n**context:** ctx\n\nBODY\n"
        );
    }

    #[test]
    fn md_without_context_keeps_same_blank_line_above_content() {
        let hits = vec![hit("a", "x.md", "Title", None, 0.0, 1, "BODY", None)];
        let s = fmt_hits_md(&hits, false);
        // contextLine is "" but the literal \n after it still fires, so a
        // single blank line separates the docid line from BODY.
        assert_eq!(s, "---\n# Title\n\n**docid:** `#a`\n\nBODY\n");
    }

    #[test]
    fn md_falls_back_to_display_path_when_title_empty() {
        let hits = vec![hit("a", "docs/x.md", "", None, 0.0, 1, "B", None)];
        let s = fmt_hits_md(&hits, false);
        assert!(s.starts_with("---\n# docs/x.md\n"));
    }

    #[test]
    fn xml_omits_title_attr_when_title_empty() {
        let hits = vec![hit("a", "x.md", "", None, 0.0, 1, "B", None)];
        let s = fmt_hits_xml(&hits, false);
        assert_eq!(s, "<file docid=\"#a\" name=\"x.md\">\nB\n</file>");
        assert!(!s.contains("title="));
    }

    #[test]
    fn xml_includes_title_and_context_when_present() {
        let hits = vec![hit("a", "x.md", "T<", Some("c&d"), 0.0, 1, "B>", None)];
        let s = fmt_hits_xml(&hits, false);
        assert_eq!(
            s,
            "<file docid=\"#a\" name=\"x.md\" title=\"T&lt;\" context=\"c&amp;d\">\nB&gt;\n</file>"
        );
    }

    #[test]
    fn xml_two_hits_joined_with_blank_line() {
        let hits = vec![
            hit("a", "x.md", "", None, 0.0, 1, "B1", None),
            hit("b", "y.md", "", None, 0.0, 1, "B2", None),
        ];
        let s = fmt_hits_xml(&hits, false);
        assert_eq!(
            s,
            "<file docid=\"#a\" name=\"x.md\">\nB1\n</file>\n\n<file docid=\"#b\" name=\"y.md\">\nB2\n</file>"
        );
    }

    #[test]
    fn full_plus_line_numbers_numbers_from_one() {
        // Regression for the latent bug: with --full we want numbering to
        // start at 1, not at the snippet's offset.
        let hits = vec![hit("a", "x.md", "", None, 0.0, 42, "snip", Some("L1\nL2"))];
        let csv_row = fmt_hits_csv(&hits, true);
        assert!(
            csv_row.contains("1: L1\n2: L2"),
            "expected body to be numbered from 1 with --full --line-numbers, got: {csv_row}"
        );
    }

    #[test]
    fn snippet_plus_line_numbers_numbers_from_hit_line() {
        let hits = vec![hit("a", "x.md", "", None, 0.0, 42, "S1\nS2", None)];
        let csv_row = fmt_hits_csv(&hits, true);
        assert!(
            csv_row.contains("42: S1\n43: S2"),
            "expected snippet numbering anchored at line 42, got: {csv_row}"
        );
    }

    #[test]
    fn empty_inputs_produce_minimal_output() {
        assert_eq!(fmt_hits_files(&[]), "");
        assert_eq!(fmt_hits_md(&[], false), "");
        assert_eq!(fmt_hits_xml(&[], false), "");
    }
}
