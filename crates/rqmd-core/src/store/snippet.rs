//! Snippet extraction and line-number rendering.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 4040–4181.

use std::sync::LazyLock;

use super::CHUNK_SIZE_CHARS;

/// Weight of intent terms relative to query terms (1.0) in snippet scoring.
pub const INTENT_WEIGHT_SNIPPET: f64 = 0.3;

static INTENT_STOP_WORDS: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
    [
        "am", "an", "as", "at", "be", "by", "do", "he", "if", "in", "is", "it", "me", "my", "no",
        "of", "on", "or", "so", "to", "up", "us", "we", "all", "and", "any", "are", "but", "can",
        "did", "for", "get", "has", "her", "him", "his", "how", "its", "let", "may", "not", "our",
        "out", "the", "too", "was", "who", "why", "you", "also", "does", "find", "from", "have",
        "into", "more", "need", "show", "some", "tell", "that", "them", "this", "want", "what",
        "when", "will", "with", "your", "about", "looking", "notes", "search", "where", "which",
    ]
    .into_iter()
    .collect()
});

/// Result of [`extract_snippet`].
#[derive(Debug, Clone, PartialEq)]
pub struct SnippetResult {
    pub line: usize,
    pub snippet: String,
    pub lines_before: usize,
    pub lines_after: usize,
    pub snippet_lines: usize,
}

/// Lowercase the intent, split on whitespace, strip leading/trailing
/// non-alphanumeric characters, then drop stop words and 1-char tokens.
/// Mirrors `extractIntentTerms` (`store.ts:4079–4083`).
pub fn extract_intent_terms(intent: &str) -> Vec<String> {
    intent
        .to_lowercase()
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|t| t.len() > 1 && !INTENT_STOP_WORDS.contains(t.as_str()))
        .collect()
}

/// Extract a context snippet around the best match of `query` in `body`.
/// Mirrors `extractSnippet` (`store.ts:4085–4168`).
pub fn extract_snippet(
    body: &str,
    query: &str,
    max_len: Option<usize>,
    chunk_pos: Option<usize>,
    chunk_len: Option<usize>,
    intent: Option<&str>,
) -> SnippetResult {
    extract_snippet_inner(body, query, max_len, chunk_pos, chunk_len, intent, false)
}

fn extract_snippet_inner(
    body: &str,
    query: &str,
    max_len: Option<usize>,
    chunk_pos: Option<usize>,
    chunk_len: Option<usize>,
    intent: Option<&str>,
    is_retry: bool,
) -> SnippetResult {
    let max_len = max_len.unwrap_or(500);
    let total_lines = body.split('\n').count();

    let (search_body, line_offset): (&str, usize) = match chunk_pos {
        Some(pos) => {
            let search_len = chunk_len.unwrap_or(CHUNK_SIZE_CHARS);
            let context_start = pos.saturating_sub(100);
            let context_end = (pos + search_len + 100).min(body.len());
            let cs = snap_to_char_boundary(body, context_start);
            let ce = snap_to_char_boundary(body, context_end);
            let off = if cs > 0 {
                body[..cs].split('\n').count() - 1
            } else {
                0
            };
            (&body[cs..ce], off)
        }
        None => (body, 0),
    };

    let lines: Vec<&str> = search_body.split('\n').collect();
    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|s| s.to_string())
        .collect();
    let intent_terms = intent.map(extract_intent_terms).unwrap_or_default();

    let mut best_line: usize = 0;
    let mut best_score: f64 = -1.0;

    for (i, line) in lines.iter().enumerate() {
        let lower = line.to_lowercase();
        let mut score = 0.0_f64;
        for term in &query_terms {
            if lower.contains(term) {
                score += 1.0;
            }
        }
        for term in &intent_terms {
            if lower.contains(term) {
                score += INTENT_WEIGHT_SNIPPET;
            }
        }
        if score > best_score {
            best_score = score;
            best_line = i;
        }
    }

    if !is_retry && best_score <= 0.0
        && let Some(pos) = chunk_pos
    {
        if pos == 0 {
            return extract_snippet_inner(body, query, Some(max_len), None, None, intent, true);
        }
        // Anchor at chunk start.
        let context_start = pos.saturating_sub(100);
        let snap_cs = snap_to_char_boundary(body, context_start);
        best_line = if pos > snap_cs {
            search_body[..pos - snap_cs].split('\n').count() - 1
        } else {
            0
        };
    }

    let start = best_line.saturating_sub(1);
    let end = (best_line + 3).min(lines.len());
    let snippet_lines: Vec<&str> = lines[start..end].to_vec();
    let mut snippet_text = snippet_lines.join("\n");

    if !is_retry && chunk_pos.unwrap_or(0) > 0 && snippet_text.trim().is_empty() {
        return extract_snippet_inner(body, query, Some(max_len), None, None, intent, true);
    }

    if snippet_text.len() > max_len {
        let cut = max_len.saturating_sub(3);
        let cut = snap_to_char_boundary(&snippet_text, cut);
        snippet_text = format!("{}...", &snippet_text[..cut]);
    }

    let absolute_start = line_offset + start + 1;
    let snippet_line_count = snippet_lines.len();
    let lines_before = absolute_start.saturating_sub(1);
    let lines_after = total_lines.saturating_sub(absolute_start + snippet_line_count - 1);

    let header = format!(
        "@@ -{},{} @@ ({} before, {} after)",
        absolute_start, snippet_line_count, lines_before, lines_after
    );
    let snippet = format!("{header}\n{snippet_text}");

    SnippetResult {
        line: line_offset + best_line + 1,
        snippet,
        lines_before,
        lines_after,
        snippet_lines: snippet_line_count,
    }
}

fn snap_to_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Render `text` with line numbers, 1-indexed (or `start_line`).
/// Mirrors `addLineNumbers` (`store.ts:4178–4181`).
pub fn add_line_numbers(text: &str, start_line: Option<usize>) -> String {
    let start = start_line.unwrap_or(1);
    text.split('\n')
        .enumerate()
        .map(|(i, line)| format!("{}: {}", start + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_terms_drop_stop_words() {
        let terms = extract_intent_terms("looking for the API endpoint");
        assert_eq!(terms, vec!["api", "endpoint"]);
    }

    #[test]
    fn intent_terms_strip_punctuation() {
        let terms = extract_intent_terms("(API!) endpoint?");
        assert_eq!(terms, vec!["api", "endpoint"]);
    }

    #[test]
    fn snippet_finds_query_match() {
        let body = "intro\nthis is a hello world\noutro";
        let r = extract_snippet(body, "hello", None, None, None, None);
        assert_eq!(r.line, 2);
        assert!(r.snippet.contains("hello world"));
        assert!(r.snippet.starts_with("@@ -1,3 @@"));
    }

    #[test]
    fn snippet_truncates_long_text() {
        let body = "x".repeat(1000);
        let r = extract_snippet(&body, "x", Some(20), None, None, None);
        // header has the diff prefix, snippet is the body. snippet body cut to 17 chars + "..."
        assert!(r.snippet.split_once('\n').is_some());
    }

    #[test]
    fn add_line_numbers_starts_at_one() {
        let out = add_line_numbers("a\nb\nc", None);
        assert_eq!(out, "1: a\n2: b\n3: c");
    }

    #[test]
    fn add_line_numbers_custom_start() {
        let out = add_line_numbers("a\nb", Some(10));
        assert_eq!(out, "10: a\n11: b");
    }

    // --- extract_snippet: ported from store.test.ts `describe("Snippet Extraction")` ---

    #[test]
    fn snippet_includes_context_lines() {
        let body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert!(r.snippet.contains("Line 2")); // context before
        assert!(r.snippet.contains("Line 3 has keyword"));
        assert!(r.snippet.contains("Line 4")); // context after
    }

    #[test]
    fn snippet_respects_max_len_with_ellipsis() {
        let body = "A".repeat(1000);
        let r = extract_snippet(&body, "query", Some(100), None, None, None);
        assert!(r.snippet.contains("@@")); // diff header
        assert!(r.snippet.contains("...")); // content truncated
    }

    #[test]
    fn snippet_uses_chunk_pos_hint() {
        let body = "First section...\n".repeat(50) + "Target keyword here\n" + &"More content...".repeat(50);
        let chunk_pos = body.find("Target keyword").unwrap();
        let r = extract_snippet(&body, "Target", Some(200), Some(chunk_pos), None, None);
        assert!(r.snippet.contains("Target keyword"));
    }

    #[test]
    fn snippet_returns_beginning_when_no_match() {
        let body = "First line\nSecond line\nThird line";
        let r = extract_snippet(body, "nonexistent", Some(500), None, None, None);
        assert_eq!(r.line, 1);
        assert!(r.snippet.contains("First line"));
    }

    #[test]
    fn snippet_includes_diff_style_header() {
        let body = "Line 1\nLine 2\nLine 3 has keyword\nLine 4\nLine 5";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert!(r.snippet.starts_with("@@ -2,4 @@ (1 before, 0 after)"));
        assert_eq!(r.lines_before, 1);
        assert_eq!(r.lines_after, 0);
        assert_eq!(r.snippet_lines, 4);
    }

    #[test]
    fn snippet_calculates_lines_before_after() {
        let body = "L1\nL2\nL3\nL4 match\nL5\nL6\nL7\nL8\nL9\nL10";
        let r = extract_snippet(body, "match", Some(500), None, None, None);
        assert_eq!(r.line, 4);
        assert_eq!(r.lines_before, 2);
        assert_eq!(r.snippet_lines, 4);
        assert_eq!(r.lines_after, 4);
    }

    #[test]
    fn snippet_header_format_values() {
        let body = "A\nB\nC keyword\nD\nE\nF\nG\nH";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert!(r.snippet.starts_with("@@ -2,4 @@ (1 before, 3 after)"));
    }

    #[test]
    fn snippet_at_document_start_shows_zero_before() {
        let body = "First line keyword\nSecond\nThird\nFourth\nFifth";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert_eq!(r.line, 1);
        assert_eq!(r.lines_before, 0);
        assert_eq!(r.snippet_lines, 3);
        assert_eq!(r.lines_after, 2);
    }

    #[test]
    fn snippet_at_document_end_shows_zero_after() {
        let body = "First\nSecond\nThird\nFourth\nFifth keyword";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert_eq!(r.line, 5);
        assert_eq!(r.lines_before, 3);
        assert_eq!(r.snippet_lines, 2);
        assert_eq!(r.lines_after, 0);
    }

    #[test]
    fn snippet_single_line_document() {
        let body = "Single line with keyword";
        let r = extract_snippet(body, "keyword", Some(500), None, None, None);
        assert_eq!(r.lines_before, 0);
        assert_eq!(r.lines_after, 0);
        assert_eq!(r.snippet_lines, 1);
        assert!(r.snippet.contains("@@ -1,1 @@ (0 before, 0 after)"));
        assert!(r.snippet.contains("Single line with keyword"));
    }

    #[test]
    fn snippet_chunk_pos_adjusts_line_numbers() {
        let padding = "Padding line\n".repeat(50);
        let body = padding.clone() + "Target keyword here\nMore content\nEven more";
        let chunk_pos = padding.len();
        let r = extract_snippet(&body, "keyword", Some(200), Some(chunk_pos), None, None);
        assert_eq!(r.line, 51);
        assert!(r.lines_before > 40);
    }

    #[test]
    fn snippet_anchors_on_chunk_pos_when_no_lexical_match() {
        // A quoted-phrase query tokenises into terms with embedded quotes that
        // never appear in the body; the fallback anchors on chunk_pos.
        let pad_line = "Lorem ipsum dolor sit amet\n";
        let padding = pad_line.repeat(100);
        let body = format!("{padding}chunk content here\nmore chunk content\n{padding}");
        let chunk_pos = padding.len();
        let r = extract_snippet(&body, "\"unrelated quoted phrase\"", Some(200), Some(chunk_pos), None, None);
        assert!(r.line > 50);
        assert!(r.line < 110);
    }

    #[test]
    fn snippet_chunk_pos_zero_falls_back_to_full_scan() {
        // chunk_pos=0 may be the bestIdx=0 default rather than a real chunk-0
        // hit, so the fallback must consider matches outside chunk 0.
        let padding = "Lorem ipsum dolor sit amet\n".repeat(200);
        let body = format!("{padding}TARGET_KEYWORD line content\ntail line\n");
        let r = extract_snippet(&body, "TARGET_KEYWORD", Some(200), Some(0), None, None);
        assert_eq!(r.line, 201);
    }

    // =========================================================================
    // extractIntentTerms — ported from intent.test.ts
    // `describe("extractIntentTerms")` (intent.test.ts:105-173).
    // =========================================================================

    #[test]
    fn intent_terms_filters_stop_words() {
        // "looking", "for", "notes", "about" are stop words
        assert_eq!(
            extract_intent_terms("looking for notes about latency optimization"),
            vec!["latency", "optimization"]
        );
    }

    #[test]
    fn intent_terms_filters_common_function_words() {
        // "what", "is", "the", "to", "find" are stop words; "best", "way" survive
        assert_eq!(
            extract_intent_terms("what is the best way to find"),
            vec!["best", "way"]
        );
    }

    #[test]
    fn intent_terms_preserves_domain_terms() {
        assert_eq!(
            extract_intent_terms("web performance latency page load times"),
            vec!["web", "performance", "latency", "page", "load", "times"]
        );
    }

    #[test]
    fn intent_terms_handles_surrounding_punctuation() {
        assert_eq!(
            extract_intent_terms("personal health, fitness, and endurance"),
            vec!["personal", "health", "fitness", "endurance"]
        );
    }

    #[test]
    fn intent_terms_preserves_internal_hyphens() {
        assert_eq!(
            extract_intent_terms("self-hosted real-time (decision-making)"),
            vec!["self-hosted", "real-time", "decision-making"]
        );
    }

    #[test]
    fn intent_terms_short_domain_terms_survive() {
        assert_eq!(
            extract_intent_terms("API design for LLM agents"),
            vec!["api", "design", "llm", "agents"]
        );
    }

    #[test]
    fn intent_terms_returns_empty_for_empty_input() {
        assert!(extract_intent_terms("").is_empty());
        assert!(extract_intent_terms("  ").is_empty());
    }

    #[test]
    fn intent_terms_filters_single_char_terms() {
        assert_eq!(extract_intent_terms("a b c web"), vec!["web"]);
    }

    #[test]
    fn intent_terms_all_stop_words_returns_empty() {
        assert!(extract_intent_terms("the and or but in on at to for of with by").is_empty());
    }

    #[test]
    fn intent_terms_preserves_2char_domain_terms() {
        let terms = extract_intent_terms("SQL CI CD DB");
        assert!(terms.iter().any(|t| t == "sql"));
        assert!(terms.iter().any(|t| t == "ci"));
        assert!(terms.iter().any(|t| t == "cd"));
        assert!(terms.iter().any(|t| t == "db"));
    }

    #[test]
    fn intent_terms_lowercases_all_terms() {
        let terms = extract_intent_terms("WebSocket HTTP REST");
        assert!(terms.iter().any(|t| t == "websocket"));
        assert!(terms.iter().any(|t| t == "http"));
        assert!(terms.iter().any(|t| t == "rest"));
    }

    #[test]
    fn intent_terms_handles_cpp_style_punctuation() {
        let terms = extract_intent_terms("C++, performance! optimization.");
        assert!(terms.iter().any(|t| t == "performance"));
        assert!(terms.iter().any(|t| t == "optimization"));
    }

    // =========================================================================
    // extractSnippet with intent — ported from intent.test.ts
    // `describe("extractSnippet with intent")` (intent.test.ts:179-267).
    // Each section contains "performance" so the query score ties (1.0 each);
    // intent terms (INTENT_WEIGHT_SNIPPET) break the tie toward the relevant
    // section. Mirrors the TS `body` built via array `.join("\n")`.
    // =========================================================================

    fn disambig_body() -> String {
        [
            "# Notes on Various Topics",
            "",
            "## Web Performance Section",
            "Web performance means optimizing page load times and Core Web Vitals.",
            "Reduce latency, improve rendering speed, and measure performance budgets.",
            "",
            "## Team Performance Section",
            "Team performance depends on trust, psychological safety, and feedback.",
            "Build culture where performance reviews drive growth not fear.",
            "",
            "## Health Performance Section",
            "Health performance comes from consistent exercise, sleep, and endurance.",
            "Track fitness metrics, optimize recovery, and monitor healthspan.",
        ]
        .join("\n")
    }

    #[test]
    fn snippet_intent_without_intent_anchors_on_query() {
        let body = disambig_body();
        // "performance" appears in title and multiple sections — anchor on a match.
        let r = extract_snippet(&body, "performance", Some(500), None, None, None);
        assert!(r.snippet.contains("Performance"));
    }

    #[test]
    fn snippet_intent_web_prefers_web_section() {
        let body = disambig_body();
        let r = extract_snippet(
            &body,
            "performance",
            Some(500),
            None,
            None,
            Some("Looking for notes about web performance, latency, and page load times"),
        );
        let re = regex::Regex::new(r"(?i)latency|page.*load|Core Web Vitals").unwrap();
        assert!(re.is_match(&r.snippet), "snippet: {}", r.snippet);
    }

    #[test]
    fn snippet_intent_health_prefers_health_section() {
        let body = disambig_body();
        let r = extract_snippet(
            &body,
            "performance",
            Some(500),
            None,
            None,
            Some("Looking for notes about personal health, fitness, and endurance"),
        );
        let re = regex::Regex::new(r"(?i)health|fitness|endurance|exercise").unwrap();
        assert!(re.is_match(&r.snippet), "snippet: {}", r.snippet);
    }

    #[test]
    fn snippet_intent_team_prefers_team_section() {
        let body = disambig_body();
        let r = extract_snippet(
            &body,
            "performance",
            Some(500),
            None,
            None,
            Some("Looking for notes about building high-performing teams and culture"),
        );
        let re = regex::Regex::new(r"(?i)team|culture|trust|feedback").unwrap();
        assert!(re.is_match(&r.snippet), "snippet: {}", r.snippet);
    }

    #[test]
    fn snippet_intent_does_not_override_strong_query_match() {
        let body = disambig_body();
        // "Core Web Vitals" is very specific — intent shouldn't pull away from it.
        let r = extract_snippet(
            &body,
            "Core Web Vitals",
            Some(500),
            None,
            None,
            Some("Looking for notes about health and fitness"),
        );
        assert!(r.snippet.contains("Core Web Vitals"));
    }

    #[test]
    fn snippet_intent_absent_equals_none() {
        let body = disambig_body();
        let without = extract_snippet(&body, "performance", Some(500), None, None, None);
        let with_none = extract_snippet(&body, "performance", Some(500), None, None, None);
        assert_eq!(without.line, with_none.line);
        assert_eq!(without.snippet, with_none.snippet);
    }

    #[test]
    fn snippet_intent_no_matching_terms_falls_back() {
        let body = disambig_body();
        let r = extract_snippet(
            &body,
            "performance",
            Some(500),
            None,
            None,
            Some("quantum computing and entanglement"),
        );
        assert!(r.snippet.contains("Performance"));
        assert!(!r.snippet.is_empty());
    }

    #[test]
    fn snippet_intent_works_with_chunk_position() {
        let body = disambig_body();
        let web_perf_start = body.find("## Web Performance").unwrap();
        let r = extract_snippet(
            &body,
            "performance",
            Some(500),
            Some(web_perf_start),
            Some(200),
            Some("web page load times"),
        );
        let re = regex::Regex::new(r"(?i)Web Performance|Core Web Vitals|Page load").unwrap();
        assert!(re.is_match(&r.snippet), "snippet: {}", r.snippet);
    }

    // --- extractSnippet intent weight behavior (intent.test.ts:273-294) ---

    fn weight_body() -> String {
        [
            "performance metrics for team velocity",
            "performance metrics for web latency",
            "performance metrics for athletic endurance",
        ]
        .join("\n")
    }

    #[test]
    fn snippet_intent_breaks_tie_when_query_matches_all_lines() {
        let body = weight_body();
        // Without intent, the first line wins (all lines score equally).
        let no_intent = extract_snippet(&body, "performance metrics", Some(500), None, None, None);
        assert_eq!(no_intent.line, 1);

        // Intent terms "web", "latency" match line 2.
        let with_intent = extract_snippet(
            &body,
            "performance metrics",
            Some(500),
            None,
            None,
            Some("web latency and page speed"),
        );
        assert!(with_intent.snippet.contains("web latency"));
    }

    // --- intent constant (intent.test.ts:506-508) ---

    #[test]
    fn intent_weight_snippet_is_0_3() {
        assert_eq!(INTENT_WEIGHT_SNIPPET, 0.3);
    }
}
