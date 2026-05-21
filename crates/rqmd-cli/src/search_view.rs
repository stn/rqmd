//! Presentation types for `search` / `vsearch` / `query`.
//!
//! Lives outside `commands/` because future consumers (rqmd-mcp, future
//! `csv/md/xml/files` formatters) need the same shape. The three input result
//! types differ in fields and pre-extraction state — this module normalises
//! them into a single owned [`Hit`] with snippet + 1-indexed line.

use serde::Serialize;

use rqmd_core::store::rrf::{HybridQueryExplain, QueryType, RRFContributionTrace, RRFExplain};
use rqmd_core::store::search::{SearchResult, SearchSource};
use rqmd_core::store::snippet::{add_line_numbers, extract_snippet};
use rqmd_core::store::virtual_path::{build_virtual_path, parse_virtual_path};
use rqmd_core::store_ops::{HybridQueryResult, VectorSearchResult};

use crate::color::Palette;
use crate::output::{escape_csv, escape_xml, OutputFormat};

/// Build a `qmd://` URI from a bare `collection/path` display path, appending
/// `?index=<name>` only for a non-default index. Mirrors qmd's `toQmdPath`
/// helper inside `outputResults` (`qmd.ts:2106-2117`): a display path with no
/// `/` (or an empty trailing segment) is returned as `qmd://<display_path>`
/// unchanged. `index_name` is `Some` only when the active index != `"index"`.
pub fn to_qmd_path(display_path: &str, index_name: Option<&str>) -> String {
    match display_path.split_once('/') {
        Some((coll, rest)) if !coll.is_empty() && !rest.is_empty() => {
            build_virtual_path(coll, rest, index_name)
        }
        _ => format!("qmd://{display_path}"),
    }
}

// ============================================================================
// CLI ("cli") human-format helpers — port of qmd `outputResults`'s cli branch
// and its helpers (`qmd.ts:1972-2095`, 2150-2228).
// ============================================================================

/// Default editor URI template when no env var / config override is set
/// (`qmd.ts:2046`).
pub const DEFAULT_EDITOR_URI_TEMPLATE: &str = "vscode://file/{path}:{line}:{col}";

/// Resolve the editor-URI template for clickable CLI links. Precedence
/// (mirrors `getEditorUriTemplate`, `qmd.ts:2054-2078`): env `RQMD_EDITOR_URI`
/// (rqmd-native) → env `QMD_EDITOR_URI` → config `editor_uri`
/// (`editor_uri_template` alias handled on deserialize) → built-in default.
pub fn editor_uri_template(config_editor_uri: Option<&str>) -> String {
    for var in ["RQMD_EDITOR_URI", "QMD_EDITOR_URI"] {
        if let Ok(t) = std::env::var(var) {
            let t = t.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    if let Some(t) = config_editor_uri {
        let t = t.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    DEFAULT_EDITOR_URI_TEMPLATE.to_string()
}

/// Percent-encode an absolute path for an editor URI. Mirrors qmd's
/// `encodePathForEditorUri` (`qmd.ts:2048-2052`): JS `encodeURI` (leaves the
/// unreserved + reserved set and `#`), then additionally encodes `?` and `#`.
fn encode_path_for_editor_uri(abs_path: &str) -> String {
    // Bytes JS `encodeURI` leaves unescaped, besides ASCII alphanumerics.
    const KEEP: &[u8] = b"-_.!~*'();,/?:@&=+$#";
    let mut out = String::with_capacity(abs_path.len());
    for &b in abs_path.as_bytes() {
        if b.is_ascii_alphanumeric() || KEEP.contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out.replace('?', "%3F").replace('#', "%23")
}

/// Substitute `{path}`/`{line}`/`{col}`/`{column}` in an editor-URI template.
/// Mirrors `buildEditorUri` (`qmd.ts:2080-2090`). `line`/`col` are clamped to
/// `>= 1`.
pub fn build_editor_uri(template: &str, abs_path: &str, line: usize, col: usize) -> String {
    let safe_line = line.max(1);
    let safe_col = col.max(1);
    let encoded = encode_path_for_editor_uri(abs_path);
    template
        .replace("{path}", &encoded)
        .replace("{line}", &safe_line.to_string())
        .replace("{col}", &safe_col.to_string())
        .replace("{column}", &safe_col.to_string())
}

/// Wrap `text` in an OSC-8 hyperlink to `url` when `is_tty`; otherwise return
/// `text` unchanged. Mirrors `termLink` (`qmd.ts:2092-2095`).
pub fn term_link(text: &str, url: &str, is_tty: bool) -> String {
    if !is_tty {
        return text.to_string();
    }
    format!("\x1b]8;;{url}\x07{text}\x1b]8;;\x07")
}

/// Highlight query terms (length >= 3, ASCII case-insensitive) in `text` with
/// yellow+bold. No-op when colour is disabled. Mirrors `highlightTerms`
/// (`qmd.ts:1972-1981`).
pub fn highlight_terms(text: &str, query: &str, p: &Palette) -> String {
    if !p.enabled {
        return text.to_string();
    }
    let pre = format!("{}{}", p.yellow(), p.bold());
    let post = p.reset();
    let mut result = text.to_string();
    for term in query.split_whitespace().filter(|t| t.chars().count() >= 3) {
        result = wrap_ascii_ci(&result, term, &pre, post);
    }
    result
}

/// Wrap every ASCII-case-insensitive occurrence of `needle` in `haystack` with
/// `pre`/`post`. `to_ascii_lowercase` preserves byte length, so match offsets
/// align with the original and slicing always lands on char boundaries.
fn wrap_ascii_ci(haystack: &str, needle: &str, pre: &str, post: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let hay_lc = haystack.to_ascii_lowercase();
    let needle_lc = needle.to_ascii_lowercase();
    let n = needle_lc.len();
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0;
    while i < haystack.len() {
        if hay_lc.as_bytes()[i..].starts_with(needle_lc.as_bytes()) {
            out.push_str(pre);
            out.push_str(&haystack[i..i + n]);
            out.push_str(post);
            i += n;
        } else {
            let ch = haystack[i..].chars().next().expect("char boundary");
            let clen = ch.len_utf8();
            out.push_str(&haystack[i..i + clen]);
            i += clen;
        }
    }
    out
}

/// Format a score as a colour-coded, 3-wide percentage. Mirrors `formatScore`
/// (`qmd.ts:1984-1990`): green >= 70%, yellow >= 40%, dim below.
pub fn format_score(score: f64, p: &Palette) -> String {
    let pct = format!("{:>3}", (score * 100.0).round() as i64);
    if !p.enabled {
        return format!("{pct}%");
    }
    let color = if score >= 0.7 {
        p.green()
    } else if score >= 0.4 {
        p.yellow()
    } else {
        p.dim()
    };
    format!("{color}{pct}%{}", p.reset())
}

/// `value.toFixed(4)` (`qmd.ts:1992-1994`).
fn format_explain_number(value: f64) -> String {
    format!("{value:.4}")
}

/// Per-invocation context for the CLI format's clickable links (qmd resolves
/// these once at the top of the cli branch, `qmd.ts:2151-2152`).
pub struct CliLinkCtx {
    /// Editor-URI template (see [`editor_uri_template`]).
    pub editor_template: String,
    /// Whether stdout is a TTY — gates OSC-8 link emission (qmd uses
    /// `process.stdout.isTTY`; note rqmd's [`Palette`] gates *colour* on stderr,
    /// so colour and link-ability deliberately use different streams).
    pub stdout_tty: bool,
}

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
    /// Absolute filesystem path, resolved by the command while the store
    /// connection is open, for the CLI format's OSC-8 clickable links. Never
    /// serialized (qmd's JSON has no such field).
    #[serde(skip)]
    pub abs_path: Option<String>,
    /// RRF/rerank trace, present only for `query --explain`. Rendered inline by
    /// the CLI format (qmd `outputResults` cli branch); the JSON `--explain`
    /// path uses [`ExplainView`] separately. Never serialized here.
    #[serde(skip)]
    pub explain: Option<HybridQueryExplain>,
}

pub fn search_result_to_hit(
    r: &SearchResult,
    query: &str,
    intent: Option<&str>,
    full: bool,
    index_name: Option<&str>,
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
        file: to_qmd_path(&r.doc.display_path, index_name),
        display_path: r.doc.display_path.clone(),
        title: r.doc.title.clone(),
        score: r.score,
        line: snip.line,
        context: r.doc.context.clone(),
        snippet: snip.snippet,
        body: full.then(|| body.to_string()),
        docid: r.doc.docid.clone(),
        abs_path: None,
        explain: None,
    }
}

pub fn vector_result_to_hit(
    r: &VectorSearchResult,
    query: &str,
    intent: Option<&str>,
    full: bool,
    index_name: Option<&str>,
) -> Hit {
    let snip = extract_snippet(&r.body, query, None, None, None, intent);
    Hit {
        file: to_qmd_path(&r.display_path, index_name),
        display_path: r.display_path.clone(),
        title: r.title.clone(),
        score: r.score,
        line: snip.line,
        context: r.context.clone(),
        snippet: snip.snippet,
        body: full.then(|| r.body.clone()),
        docid: r.docid.clone(),
        abs_path: None,
        explain: None,
    }
}

pub fn hybrid_result_to_hit(r: &HybridQueryResult, full: bool, index_name: Option<&str>) -> Hit {
    Hit {
        file: to_qmd_path(&r.display_path, index_name),
        display_path: r.display_path.clone(),
        title: r.title.clone(),
        score: r.score,
        line: line_at_byte_offset(&r.body, r.best_chunk_pos),
        context: r.context.clone(),
        snippet: r.best_chunk.clone(),
        body: full.then(|| r.body.clone()),
        docid: r.docid.clone(),
        abs_path: None,
        explain: r.explain.clone(),
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

/// Human "cli" search output. Full port of qmd `outputResults`'s cli branch
/// (`qmd.ts:2150-2228`): a `qmd://` (or OSC-8-clickable, on a TTY) path line
/// with optional `:line` and docid, then `Title:`/`Context:`/`Score:` lines,
/// an optional `--explain` block, a blank line, and the term-highlighted
/// snippet (or full body). Results are separated by a blank line.
pub fn print_hits_cli(
    hits: &[Hit],
    p: &Palette,
    line_numbers: bool,
    query: &str,
    link: &CliLinkCtx,
) {
    if hits.is_empty() {
        // qmd `printEmptySearchResults` cli default (`qmd.ts:2028`).
        println!("No results found.");
        return;
    }
    let query_lc = query.to_lowercase();
    let n = hits.len();
    for (i, h) in hits.iter().enumerate() {
        // `:line` is shown only when a query term actually matched in the
        // snippet body — excluding the `@@ … @@` diff header line that
        // `extract_snippet` prepends (qmd.ts:2169-2170).
        let snippet_body = h
            .snippet
            .split('\n')
            .skip(1)
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase();
        let has_match = query_lc
            .split_whitespace()
            .any(|t| !t.is_empty() && snippet_body.contains(t));
        let line_info = if has_match {
            format!(":{}", h.line)
        } else {
            String::new()
        };
        let docid_str = if h.docid.is_empty() {
            String::new()
        } else {
            format!(" {}#{}{}", p.dim(), h.docid, p.reset())
        };

        // Line 1: on a TTY (with a resolved abs path), an OSC-8 link whose text
        // is the collection-relative path; otherwise the full `qmd://…[?index=]`
        // link (qmd.ts:2160-2186).
        let parsed_rel = parse_virtual_path(&h.file)
            .ok()
            .map(|vp| vp.path)
            .filter(|s| !s.is_empty());
        let line1 = match (
            link.stdout_tty,
            h.abs_path.as_deref(),
            parsed_rel.as_deref(),
        ) {
            (true, Some(abs), Some(rel)) => {
                let link_line = if has_match { h.line } else { 1 };
                let target = build_editor_uri(&link.editor_template, abs, link_line, 1);
                let clickable = term_link(&format!("{rel}{line_info}"), &target, true);
                format!("{}{clickable}{}{docid_str}", p.cyan(), p.reset())
            }
            _ => format!(
                "{}{}{}{line_info}{}{docid_str}",
                p.cyan(),
                h.file,
                p.dim(),
                p.reset()
            ),
        };
        println!("{line1}");

        if !h.title.is_empty() {
            println!("{}Title: {}{}", p.bold(), h.title, p.reset());
        }
        if let Some(ctx) = &h.context {
            println!("{}Context: {}{}", p.dim(), ctx, p.reset());
        }
        println!(
            "Score: {}{}{}",
            p.bold(),
            format_score(h.score, p),
            p.reset()
        );

        if let Some(e) = &h.explain {
            print_explain_block(e, p);
        }

        println!();
        let content = render_content(h, line_numbers);
        println!("{}", highlight_terms(&content, query, p));

        // Double blank between results (qmd `console.log('\n')`, qmd.ts:2227).
        if i < n - 1 {
            println!("\n");
        }
    }
}

/// The `--explain` block in the cli format (qmd.ts:2195-2218).
fn print_explain_block(e: &HybridQueryExplain, p: &Palette) {
    let join_scores = |xs: &[f64]| -> String {
        if xs.is_empty() {
            "none".to_string()
        } else {
            xs.iter()
                .map(|v| format_explain_number(*v))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };
    println!(
        "{}Explain: fts=[{}] vec=[{}]{}",
        p.dim(),
        join_scores(&e.fts_scores),
        join_scores(&e.vector_scores),
        p.reset()
    );
    println!(
        "{}  RRF: total={} base={} bonus={} rank={}{}",
        p.dim(),
        format_explain_number(e.rrf.total_score),
        format_explain_number(e.rrf.base_score),
        format_explain_number(e.rrf.top_rank_bonus),
        e.rrf.rank,
        p.reset(),
    );
    println!(
        "{}  Blend: {}%*{} + {}%*{} = {}{}",
        p.dim(),
        (e.rrf.weight * 100.0).round() as i64,
        format_explain_number(e.rrf.position_score),
        ((1.0 - e.rrf.weight) * 100.0).round() as i64,
        format_explain_number(e.rerank_score),
        format_explain_number(e.blended_score),
        p.reset(),
    );
    let mut contribs: Vec<&RRFContributionTrace> = e.rrf.contributions.iter().collect();
    contribs.sort_by(|a, b| {
        b.rrf_contribution
            .partial_cmp(&a.rrf_contribution)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let summary = contribs
        .iter()
        .take(3)
        .map(|c| {
            format!(
                "{}/{}#{}:{}",
                source_str(c.source),
                query_type_str(c.query_type),
                c.rank,
                format_explain_number(c.rrf_contribution)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    if !summary.is_empty() {
        println!("{}  Top RRF contributions: {summary}{}", p.dim(), p.reset());
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
            escape_csv(&h.file),
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
            // `file` is the `qmd://...[?index=]` URI (qmd `outputResults` files
            // branch, `qmd.ts:2179`). Intentionally NOT CSV-escaped — matches
            // qmd. Paths with commas/quotes pass through raw.
            if let Some(ctx) = &h.context {
                let esc = ctx.replace('"', "\"\"");
                format!("#{},{:.2},{},\"{esc}\"", h.docid, h.score, h.file)
            } else {
                format!("#{},{:.2},{}", h.docid, h.score, h.file)
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
                name = escape_xml(&h.file),
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
    query: &str,
    link: &CliLinkCtx,
) -> anyhow::Result<()> {
    match format {
        OutputFormat::Cli => print_hits_cli(hits, p, line_numbers, query, link),
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

/// CLI-side `Serialize` wrapper for [`HybridQueryExplain`] (rqmd-core types
/// don't derive Serialize). Re-defining the shape here means a field rename
/// upstream surfaces as a compile error rather than silently broken JSON.
#[derive(Debug, Serialize)]
pub struct ExplainView<'a> {
    /// Owned because it's the `?index=`-annotated `qmd://` link built per row,
    /// not a field borrowed from the core result.
    pub file: String,
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
    pub fn new(file: String, e: &'a HybridQueryExplain) -> Self {
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

// ============================================================================
// MCP CSV
//
// Maps to qmd `searchResultsToMcpCsv` (`formatter.ts:217–225`). Lives here
// alongside the other search formatters (qmd keeps it in the CLI formatter
// module too). The MCP server (rqmd-mcp) is still a stub, so this is
// forward-looking and currently unused outside tests.
// ============================================================================

/// A pre-extracted MCP search hit (caller has already computed the snippet).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct McpHit {
    pub docid: String,
    pub file: String,
    pub title: String,
    pub score: f64,
    pub context: Option<String>,
    pub snippet: String,
}

/// Format MCP search hits as the simple CSV qmd's MCP server emits.
#[allow(dead_code)]
pub fn mcp_results_to_csv(rows: &[McpHit]) -> String {
    let mut out = String::from("docid,file,title,score,context,snippet");
    for r in rows {
        out.push('\n');
        out.push_str(
            &[
                escape_csv(&format!("#{}", r.docid)),
                escape_csv(&r.file),
                escape_csv(&r.title),
                // qmd passes the raw number through `String(...)`; Rust's f64
                // Display matches typical scores but can diverge from JS number
                // formatting at extreme magnitudes (and applies no rounding).
                escape_csv(&r.score.to_string()),
                escape_csv(r.context.as_deref().unwrap_or("")),
                escape_csv(&r.snippet),
            ]
            .join(","),
        );
    }
    out
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
            abs_path: None,
            explain: None,
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

    // ========================================================================
    // qmd formatter.test.ts parity: search JSON / dispatch / MCP CSV
    // ========================================================================

    const TEST_CONTEXT: &str = "Internal engineering keynotes from company summit events";

    fn ctx_hits() -> Vec<Hit> {
        vec![hit(
            "dc5590",
            "qmd://archive/summit/keynote.md",
            "Summit Keynote",
            Some(TEST_CONTEXT),
            0.84,
            3,
            "This is the keynote content.",
            None,
        )]
    }

    // ---- search JSON: context present / line (qmd A1/A2/A3) ----

    #[test]
    fn json_includes_context() {
        let v = serde_json::to_value(ctx_hits()).unwrap();
        assert_eq!(v[0]["context"], TEST_CONTEXT);
    }

    #[test]
    fn json_omits_context_when_none() {
        let hits = vec![hit("a", "x.md", "T", None, 0.5, 1, "s", None)];
        let v = serde_json::to_value(hits).unwrap();
        assert!(v[0].get("context").is_none());
    }

    #[test]
    fn json_includes_line() {
        let v = serde_json::to_value(ctx_hits()).unwrap();
        assert!(v[0]["line"].as_u64().unwrap() > 0);
    }

    #[test]
    fn json_includes_line_with_full() {
        let hits = vec![hit("a", "x.md", "T", None, 0.5, 5, "snip", Some("L1\nL2"))];
        let v = serde_json::to_value(hits).unwrap();
        assert!(v[0]["line"].as_u64().unwrap() > 0);
    }

    // ---- formatSearchResults dispatch targets (qmd A9–A13) ----

    #[test]
    fn format_search_results_json_includes_context() {
        assert!(serde_json::to_string(&ctx_hits())
            .unwrap()
            .contains(TEST_CONTEXT));
    }

    #[test]
    fn format_search_results_csv_includes_context() {
        assert!(fmt_hits_csv(&ctx_hits(), false).contains(TEST_CONTEXT));
    }

    #[test]
    fn format_search_results_files_includes_context() {
        assert!(fmt_hits_files(&ctx_hits()).contains(TEST_CONTEXT));
    }

    #[test]
    fn format_search_results_md_includes_context() {
        assert!(fmt_hits_md(&ctx_hits(), false).contains(TEST_CONTEXT));
    }

    #[test]
    fn format_search_results_xml_includes_context() {
        assert!(fmt_hits_xml(&ctx_hits(), false).contains(TEST_CONTEXT));
    }

    // ---- MCP CSV (qmd A8) ----

    #[test]
    fn mcp_csv_includes_context() {
        let rows = vec![McpHit {
            docid: "dc5590".to_string(),
            file: "qmd://archive/summit/keynote.md".to_string(),
            title: "Summit Keynote".to_string(),
            score: 0.84,
            context: Some(TEST_CONTEXT.to_string()),
            snippet: "This is the keynote content.".to_string(),
        }];
        let out = mcp_results_to_csv(&rows);
        assert!(out.lines().next().unwrap().contains("context"));
        assert!(out.contains(TEST_CONTEXT));
    }

    // ========================================================================
    // ?index= link annotation + cli-format helpers (qmd parity)
    // ========================================================================

    #[test]
    fn to_qmd_path_default_index_has_no_query() {
        assert_eq!(
            to_qmd_path("fixtures/test1.md", None),
            "qmd://fixtures/test1.md"
        );
    }

    #[test]
    fn to_qmd_path_custom_index_appends_query() {
        assert_eq!(
            to_qmd_path("fixtures/a/b.md", Some("release-notes")),
            "qmd://fixtures/a/b.md?index=release-notes"
        );
    }

    #[test]
    fn to_qmd_path_no_slash_is_left_bare() {
        // No `/` → returned as `qmd://<display_path>`, no `?index=` (qmd parity).
        assert_eq!(to_qmd_path("fixtures", Some("x")), "qmd://fixtures");
    }

    #[test]
    fn to_qmd_path_trailing_slash_keeps_bare() {
        // Empty trailing segment → bare, matching qmd's `segments.length === 0`.
        assert_eq!(to_qmd_path("fixtures/", Some("x")), "qmd://fixtures/");
    }

    #[test]
    fn to_qmd_path_encodes_index_name() {
        assert_eq!(
            to_qmd_path("c/p.md", Some("a b")),
            "qmd://c/p.md?index=a%20b"
        );
    }

    #[test]
    fn term_link_wraps_only_on_tty() {
        assert_eq!(term_link("text", "url", false), "text");
        assert_eq!(
            term_link("text", "url", true),
            "\x1b]8;;url\x07text\x1b]8;;\x07"
        );
    }

    #[test]
    fn build_editor_uri_substitutes_placeholders() {
        let s = build_editor_uri("vscode://file/{path}:{line}:{col}", "/a/b.md", 12, 1);
        assert_eq!(s, "vscode://file//a/b.md:12:1");
    }

    #[test]
    fn build_editor_uri_column_alias_and_clamp_and_encode() {
        // {column} alias, line/col clamped to >= 1, space percent-encoded.
        let s = build_editor_uri("e://{path}#{line},{column}", "/x y.md", 0, 0);
        assert_eq!(s, "e:///x%20y.md#1,1");
    }

    #[test]
    fn highlight_terms_noop_when_color_disabled() {
        let p = Palette { enabled: false };
        assert_eq!(highlight_terms("hello world", "world", &p), "hello world");
    }

    #[test]
    fn highlight_terms_wraps_long_terms_case_insensitively() {
        let p = Palette { enabled: true };
        let out = highlight_terms("Hello WORLD", "world", &p);
        assert_eq!(
            out,
            format!("Hello {}{}WORLD{}", p.yellow(), p.bold(), p.reset())
        );
    }

    #[test]
    fn highlight_terms_skips_short_terms() {
        let p = Palette { enabled: true };
        assert_eq!(highlight_terms("a of b", "of", &p), "a of b");
    }

    #[test]
    fn format_score_plain_when_disabled() {
        let p = Palette { enabled: false };
        assert_eq!(format_score(0.842, &p), " 84%");
        assert_eq!(format_score(0.05, &p), "  5%");
    }

    #[test]
    fn format_score_colors_by_threshold() {
        let p = Palette { enabled: true };
        assert!(format_score(0.8, &p).contains(p.green()));
        assert!(format_score(0.5, &p).contains(p.yellow()));
        assert!(format_score(0.1, &p).contains(p.dim()));
    }
}
