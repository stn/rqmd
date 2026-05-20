//! Smart markdown chunking with break-point detection.
//!
//! Port of the chunking portion of `tobi/qmd`'s `src/store.ts`
//! (lines 79–310, 2363–2380). The token-based variant
//! `chunkDocumentByTokens` (lines 2412–2530) is LLM-dependent and
//! deliberately out of scope.
//!
//! ## Algorithm
//!
//! The TS implementation used a regex table with negative lookahead
//! (`/\n#{1}(?!#)/g`). Rust's `regex` crate has no lookahead, so we encode
//! the break table as an [`BreakKind`] enum and walk newlines manually,
//! classifying each by inspecting the next few bytes. This matches the TS
//! `match.index` semantics exactly and avoids dragging in a regex engine
//! variant.

use std::collections::BTreeMap;

use super::{CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, CHUNK_WINDOW_CHARS};

// ============================================================================
// Types
// ============================================================================

/// A potential break point in a document with its base score.
#[derive(Debug, Clone, PartialEq)]
pub struct BreakPoint {
    pub pos: usize,
    pub score: i32,
    pub kind: BreakKind,
}

/// Classification of a break point — score is intrinsic to the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakKind {
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
    CodeBlock,
    Hr,
    Blank,
    List,
    NumList,
    Newline,
    /// AST: class / interface / struct / trait / impl / mod boundary.
    AstClass,
    /// AST: function / method / export / decorated definition boundary.
    AstFunc,
    /// AST: type alias / enum boundary.
    AstType,
    /// AST: import / use declaration.
    AstImport,
    /// AST: capture name not in the known set — score-20 fallback so future
    /// query edits don't silently get promoted to function priority.
    /// Mirrors TS `SCORE_MAP[name] ?? 20` (`ast.ts:306`).
    AstUnknown,
}

impl BreakKind {
    pub fn score(self) -> i32 {
        match self {
            BreakKind::H1 => 100,
            BreakKind::H2 => 90,
            BreakKind::H3 => 80,
            BreakKind::H4 => 70,
            BreakKind::H5 => 60,
            BreakKind::H6 => 50,
            BreakKind::CodeBlock => 80,
            BreakKind::Hr => 60,
            BreakKind::Blank => 20,
            BreakKind::List | BreakKind::NumList => 5,
            BreakKind::Newline => 1,
            BreakKind::AstClass => 100,
            BreakKind::AstFunc => 90,
            BreakKind::AstType => 80,
            BreakKind::AstImport => 60,
            BreakKind::AstUnknown => 20,
        }
    }
}

/// A region between matching ```` ``` ```` markers where chunk boundaries
/// must not fall.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeFenceRegion {
    pub start: usize,
    pub end: usize,
}

/// Chunking strategy. The token-based variant is out of scope this pass.
///
/// `Auto` augments the regex (manual-scan) break points with AST break points
/// when a supported code `filepath` is supplied; `Regex` always uses the regex
/// algorithm alone and never invokes the AST parser, even for code files.
/// Mirrors TS `chunkDocumentAsync` strategy handling (`store.ts:2394`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkStrategy {
    #[default]
    Auto,
    Regex,
}

/// A single chunk: text slice + character offset within the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub pos: usize,
}

// ============================================================================
// Break-point scanning
// ============================================================================

/// Scan `text` for every potential break point.
///
/// Mirrors `scanBreakPoints` (`store.ts:120–141`). At each `\n` we classify
/// the position and feed it into a position-keyed map keeping the highest
/// score per position.
pub fn scan_break_points(text: &str) -> Vec<BreakPoint> {
    let mut seen: BTreeMap<usize, BreakPoint> = BTreeMap::new();
    let bytes = text.as_bytes();

    for (i, _) in text.match_indices('\n') {
        for kind in classify_newline(bytes, i) {
            let bp = BreakPoint {
                pos: i,
                score: kind.score(),
                kind,
            };
            seen.entry(i)
                .and_modify(|existing| {
                    if bp.score > existing.score {
                        *existing = bp.clone();
                    }
                })
                .or_insert(bp);
        }
    }

    seen.into_values().collect()
}

/// Return every [`BreakKind`] that the newline at byte position `i` matches.
/// The list mirrors the TS regex table at `store.ts:100–113`.
fn classify_newline(bytes: &[u8], i: usize) -> Vec<BreakKind> {
    let mut kinds = Vec::with_capacity(2);
    let after = i + 1;

    // Headings: `\n` + 1–6 `#` + non-`#`.
    let mut n = 0;
    while n < 6 && after + n < bytes.len() && bytes[after + n] == b'#' {
        n += 1;
    }
    if n >= 1 && after + n < bytes.len() && bytes[after + n] != b'#' {
        kinds.push(match n {
            1 => BreakKind::H1,
            2 => BreakKind::H2,
            3 => BreakKind::H3,
            4 => BreakKind::H4,
            5 => BreakKind::H5,
            6 => BreakKind::H6,
            _ => unreachable!(),
        });
    }

    // Code fence: `\n````.
    if after + 2 < bytes.len() && &bytes[after..after + 3] == b"```" {
        kinds.push(BreakKind::CodeBlock);
    }

    // Horizontal rule: `\n` + (`---` | `***` | `___`) + optional ws + `\n`.
    if after + 2 < bytes.len() {
        let triplet = &bytes[after..after + 3];
        if triplet == b"---" || triplet == b"***" || triplet == b"___" {
            // Walk past trailing whitespace until we find a newline within ~32 bytes.
            let scan_end = (after + 3 + 32).min(bytes.len());
            let mut j = after + 3;
            while j < scan_end && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < scan_end && bytes[j] == b'\n' {
                kinds.push(BreakKind::Hr);
            }
        }
    }

    // Blank line: `\n\n+`.
    if after < bytes.len() && bytes[after] == b'\n' {
        kinds.push(BreakKind::Blank);
    }

    // List markers: `\n- ` or `\n* `.
    if after + 1 < bytes.len() && matches!(bytes[after], b'-' | b'*') && bytes[after + 1] == b' ' {
        kinds.push(BreakKind::List);
    }

    // Numbered list: `\n\d+. `.
    if after < bytes.len() && bytes[after].is_ascii_digit() {
        let mut j = after;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j + 1 < bytes.len() && bytes[j] == b'.' && bytes[j + 1] == b' ' {
            kinds.push(BreakKind::NumList);
        }
    }

    // Newline fallback — always present.
    kinds.push(BreakKind::Newline);

    kinds
}

// ============================================================================
// Code fences
// ============================================================================

/// Find every code fence region in `text`. Unclosed fences extend to EOF.
/// Mirrors `findCodeFences` (`store.ts:147–169`).
pub fn find_code_fences(text: &str) -> Vec<CodeFenceRegion> {
    let mut regions = Vec::new();
    let mut in_fence = false;
    let mut fence_start = 0;

    for (idx, m) in text.match_indices("\n```") {
        if !in_fence {
            fence_start = idx;
            in_fence = true;
        } else {
            regions.push(CodeFenceRegion {
                start: fence_start,
                end: idx + m.len(),
            });
            in_fence = false;
        }
    }

    if in_fence {
        regions.push(CodeFenceRegion {
            start: fence_start,
            end: text.len(),
        });
    }

    regions
}

pub fn is_inside_code_fence(pos: usize, fences: &[CodeFenceRegion]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

// ============================================================================
// Cutoff selection
// ============================================================================

/// Find the best cut position using scored break points with squared-distance decay.
///
/// Mirrors `findBestCutoff` (`store.ts:191–227`).
pub fn find_best_cutoff(
    break_points: &[BreakPoint],
    target_char_pos: usize,
    window_chars: usize,
    decay_factor: f64,
    code_fences: &[CodeFenceRegion],
) -> usize {
    // Sorted-by-pos invariant: the `break` below is only safe if the input
    // is non-decreasing in `pos`. `merge_break_points` produces this via
    // BTreeMap iteration; this assertion fails fast in dev if a future
    // change provides an unsorted slice.
    debug_assert!(
        break_points.windows(2).all(|w| w[0].pos <= w[1].pos),
        "break_points must be sorted by pos"
    );

    let window_start = target_char_pos.saturating_sub(window_chars);
    let mut best_score = -1.0_f64;
    let mut best_pos = target_char_pos;

    for bp in break_points {
        if bp.pos < window_start {
            continue;
        }
        if bp.pos > target_char_pos {
            break;
        }
        if is_inside_code_fence(bp.pos, code_fences) {
            continue;
        }
        let distance = (target_char_pos - bp.pos) as f64;
        let normalised = distance / window_chars as f64;
        let multiplier = 1.0 - (normalised * normalised) * decay_factor;
        let final_score = bp.score as f64 * multiplier;
        if final_score > best_score {
            best_score = final_score;
            best_pos = bp.pos;
        }
    }

    best_pos
}

// ============================================================================
// Chunking
// ============================================================================

/// Merge two break-point lists keeping the highest score at each position.
/// Mirrors `mergeBreakPoints` (`store.ts:239–254`).
pub fn merge_break_points(a: &[BreakPoint], b: &[BreakPoint]) -> Vec<BreakPoint> {
    let mut seen: BTreeMap<usize, BreakPoint> = BTreeMap::new();
    for bp in a.iter().chain(b.iter()) {
        seen.entry(bp.pos)
            .and_modify(|existing| {
                if bp.score > existing.score {
                    *existing = bp.clone();
                }
            })
            .or_insert_with(|| bp.clone());
    }
    seen.into_values().collect()
}

/// Core chunk algorithm over precomputed break points + code fences.
/// Mirrors `chunkDocumentWithBreakPoints` (`store.ts:260–310`).
pub fn chunk_document_with_break_points(
    content: &str,
    break_points: &[BreakPoint],
    code_fences: &[CodeFenceRegion],
    max_chars: usize,
    overlap_chars: usize,
    window_chars: usize,
) -> Vec<Chunk> {
    if content.len() <= max_chars {
        return vec![Chunk {
            text: content.to_string(),
            pos: 0,
        }];
    }

    let mut chunks = Vec::new();
    let mut char_pos = 0;

    while char_pos < content.len() {
        let target_end_pos = (char_pos + max_chars).min(content.len());
        let mut end_pos = target_end_pos;

        if end_pos < content.len() {
            let best =
                find_best_cutoff(break_points, target_end_pos, window_chars, 0.7, code_fences);
            if best > char_pos && best <= target_end_pos {
                end_pos = best;
            }
        }

        if end_pos <= char_pos {
            end_pos = (char_pos + max_chars).min(content.len());
        }

        // Snap to a char boundary so we never split mid-UTF-8.
        let end_pos = snap_to_char_boundary(content, end_pos);
        let safe_start = snap_to_char_boundary(content, char_pos);

        chunks.push(Chunk {
            text: content[safe_start..end_pos].to_string(),
            pos: safe_start,
        });

        if end_pos >= content.len() {
            break;
        }

        let new_pos = end_pos.saturating_sub(overlap_chars);
        let last_pos = chunks.last().map(|c| c.pos).unwrap_or(0);
        char_pos = if new_pos <= last_pos {
            end_pos
        } else {
            new_pos
        };
    }

    chunks
}

fn snap_to_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Synchronous chunk entry point. Mirrors `chunkDocument` (`store.ts:2363–2380`).
///
/// In `Auto` mode, `filepath` enables AST-aware chunking for supported code
/// files (see [`super::ast::detect_language`]). Pass `None` for content with no
/// associated path or when AST chunking is undesired; passing `Some` for a
/// markdown or other unsupported extension is harmless — the AST step
/// becomes a no-op. The `Regex` strategy bypasses the AST parser entirely
/// regardless of `filepath`.
pub fn chunk_document(
    content: &str,
    strategy: ChunkStrategy,
    filepath: Option<&str>,
    max_chars: Option<usize>,
    overlap_chars: Option<usize>,
    window_chars: Option<usize>,
) -> Vec<Chunk> {
    let regex_bp = scan_break_points(content);
    // AST break points are only used in `Auto` mode. The `Regex` strategy
    // bypasses the AST parser entirely, even when a code `filepath` is given —
    // mirrors TS `chunkDocumentAsync` (`store.ts:2394`, AST only when
    // `strategy === "auto" && filepath`).
    let ast_bp = if strategy == ChunkStrategy::Regex {
        Vec::new()
    } else {
        filepath
            .map(|p| super::ast::get_ast_break_points(content, p))
            .unwrap_or_default()
    };
    let break_points = if ast_bp.is_empty() {
        regex_bp
    } else {
        merge_break_points(&regex_bp, &ast_bp)
    };
    let fences = find_code_fences(content);
    chunk_document_with_break_points(
        content,
        &break_points,
        &fences,
        max_chars.unwrap_or(CHUNK_SIZE_CHARS),
        overlap_chars.unwrap_or(CHUNK_OVERLAP_CHARS),
        window_chars.unwrap_or(CHUNK_WINDOW_CHARS),
    )
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn break_points_classify_headings() {
        let bps = scan_break_points("\n# H1\n## H2\n### H3");
        let kinds: Vec<BreakKind> = bps.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BreakKind::H1));
        assert!(kinds.contains(&BreakKind::H2));
        assert!(kinds.contains(&BreakKind::H3));
    }

    #[test]
    fn break_points_reject_too_many_hashes() {
        // `####### H7` should not classify as any heading (more than 6 `#`).
        let bps = scan_break_points("\n####### too deep");
        for bp in &bps {
            assert!(
                !matches!(
                    bp.kind,
                    BreakKind::H1
                        | BreakKind::H2
                        | BreakKind::H3
                        | BreakKind::H4
                        | BreakKind::H5
                        | BreakKind::H6
                ),
                "unexpected heading classification: {:?}",
                bp
            );
        }
    }

    #[test]
    fn break_points_lookahead_blocks_h3_without_space() {
        // `###heading-without-space` — TS `(?!#)` requires the next char to
        // not be `#`. Here the 4th char is `h`, so this SHOULD classify as H3.
        let bps = scan_break_points("\n###heading");
        assert!(bps.iter().any(|b| b.kind == BreakKind::H3));
    }

    #[test]
    fn break_points_h3_requires_non_hash_follow() {
        // `####heading` — 4 hashes, next byte is `h` -> H4, never H3.
        let bps = scan_break_points("\n####heading");
        let kinds: Vec<BreakKind> = bps.iter().map(|b| b.kind).collect();
        assert!(kinds.contains(&BreakKind::H4));
        assert!(!kinds.contains(&BreakKind::H3));
    }

    #[test]
    fn code_fences_pair_up() {
        let text = "before\n```\ncode\n```\nafter";
        let fences = find_code_fences(text);
        assert_eq!(fences.len(), 1);
        assert!(text[fences[0].start..fences[0].end].contains("code"));
    }

    #[test]
    fn unclosed_fence_extends_to_eof() {
        let text = "before\n```\nopen forever";
        let fences = find_code_fences(text);
        assert_eq!(fences.len(), 1);
        assert_eq!(fences[0].end, text.len());
    }

    #[test]
    fn chunk_short_doc_returns_one_chunk() {
        let chunks = chunk_document(
            "hello",
            ChunkStrategy::Auto,
            None,
            Some(100),
            Some(10),
            Some(20),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello");
    }

    #[test]
    fn chunk_long_doc_overlaps() {
        let text = "a".repeat(50) + "\n\n" + &"b".repeat(50);
        let chunks = chunk_document(
            &text,
            ChunkStrategy::Auto,
            None,
            Some(40),
            Some(8),
            Some(16),
        );
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.len() <= 40 || c.pos == 0);
        }
    }

    // Helper: build a BreakPoint with an explicit score (find_best_cutoff /
    // merge_break_points compare the `score` field, not `kind.score()`).
    fn mk(pos: usize, score: i32, kind: BreakKind) -> BreakPoint {
        BreakPoint { pos, score, kind }
    }

    // --- scanBreakPoints: ported from store.test.ts `describe("scanBreakPoints")` ---

    #[test]
    fn break_points_detect_code_blocks() {
        let bps = scan_break_points("Before\n```js\ncode\n```\nAfter");
        let code: Vec<_> = bps
            .iter()
            .filter(|b| b.kind == BreakKind::CodeBlock)
            .collect();
        assert_eq!(code.len(), 2); // opening and closing
        assert_eq!(code[0].score, 80);
    }

    #[test]
    fn break_points_detect_horizontal_rule() {
        let bps = scan_break_points("Text\n---\nMore text");
        let hr = bps.iter().find(|b| b.kind == BreakKind::Hr).unwrap();
        assert_eq!(hr.score, 60);
    }

    #[test]
    fn break_points_detect_blank_lines() {
        let bps = scan_break_points("First paragraph.\n\nSecond paragraph.");
        let blank = bps.iter().find(|b| b.kind == BreakKind::Blank).unwrap();
        assert_eq!(blank.score, 20);
    }

    #[test]
    fn break_points_detect_list_items() {
        let bps = scan_break_points("Intro\n- Item 1\n- Item 2\n1. Numbered");
        let lists: Vec<_> = bps.iter().filter(|b| b.kind == BreakKind::List).collect();
        let num: Vec<_> = bps
            .iter()
            .filter(|b| b.kind == BreakKind::NumList)
            .collect();
        assert_eq!(lists.len(), 2);
        assert_eq!(num.len(), 1);
        assert_eq!(lists[0].score, 5);
        assert_eq!(num[0].score, 5);
    }

    #[test]
    fn break_points_detect_newlines_fallback() {
        let bps = scan_break_points("Line 1\nLine 2\nLine 3");
        let nl: Vec<_> = bps
            .iter()
            .filter(|b| b.kind == BreakKind::Newline)
            .collect();
        assert_eq!(nl.len(), 2);
        assert_eq!(nl[0].score, 1);
    }

    #[test]
    fn break_points_sorted_by_position() {
        let bps = scan_break_points("A\n# B\n\nC\n## D");
        for w in bps.windows(2) {
            assert!(w[1].pos > w[0].pos);
        }
    }

    #[test]
    fn break_points_higher_score_wins_same_position() {
        // `\n#` matches both newline (1) and h1 (100) — only h1 is kept.
        let bps = scan_break_points("Text\n# Heading");
        let at4: Vec<_> = bps.iter().filter(|b| b.pos == 4).collect();
        assert_eq!(at4.len(), 1);
        assert_eq!(at4[0].kind, BreakKind::H1);
        assert_eq!(at4[0].score, 100);
    }

    // --- findCodeFences: ported from store.test.ts `describe("findCodeFences")` ---

    #[test]
    fn find_single_code_fence_exact_bounds() {
        let text = "Before\n```js\ncode here\n```\nAfter";
        let fences = find_code_fences(text);
        assert_eq!(fences.len(), 1);
        assert_eq!(fences[0].start, 6); // position of first \n```
        assert_eq!(fences[0].end, 26); // after the closing \n```
    }

    #[test]
    fn find_multiple_code_fences() {
        let text = "Intro\n```\nblock1\n```\nMiddle\n```\nblock2\n```\nEnd";
        assert_eq!(find_code_fences(text).len(), 2);
    }

    #[test]
    fn find_no_code_fences_returns_empty() {
        assert_eq!(find_code_fences("No code fences here").len(), 0);
    }

    // --- isInsideCodeFence: ported from store.test.ts `describe("isInsideCodeFence")` ---

    #[test]
    fn inside_code_fence_true_within() {
        let f = [CodeFenceRegion { start: 10, end: 30 }];
        assert!(is_inside_code_fence(15, &f));
        assert!(is_inside_code_fence(20, &f));
    }

    #[test]
    fn inside_code_fence_false_outside() {
        let f = [CodeFenceRegion { start: 10, end: 30 }];
        assert!(!is_inside_code_fence(5, &f));
        assert!(!is_inside_code_fence(35, &f));
    }

    #[test]
    fn inside_code_fence_false_at_boundaries() {
        let f = [CodeFenceRegion { start: 10, end: 30 }];
        assert!(!is_inside_code_fence(10, &f)); // at start
        assert!(!is_inside_code_fence(30, &f)); // at end
    }

    #[test]
    fn inside_code_fence_multiple() {
        let f = [
            CodeFenceRegion { start: 10, end: 30 },
            CodeFenceRegion { start: 50, end: 70 },
        ];
        assert!(is_inside_code_fence(20, &f));
        assert!(is_inside_code_fence(60, &f));
        assert!(!is_inside_code_fence(40, &f));
    }

    // --- findBestCutoff: ported from store.test.ts `describe("findBestCutoff")` ---

    #[test]
    fn cutoff_prefers_higher_score() {
        let bps = [
            mk(100, 1, BreakKind::Newline),
            mk(150, 100, BreakKind::H1),
            mk(180, 20, BreakKind::Blank),
        ];
        assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 150);
    }

    #[test]
    fn cutoff_h2_at_window_edge_beats_blank_at_target() {
        let bps = [mk(100, 90, BreakKind::H2), mk(195, 20, BreakKind::Blank)];
        assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 100);
    }

    #[test]
    fn cutoff_high_score_overcomes_distance() {
        let bps = [mk(150, 100, BreakKind::H1), mk(195, 1, BreakKind::Newline)];
        assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 150);
    }

    #[test]
    fn cutoff_returns_target_when_no_breaks_in_window() {
        let bps = [mk(10, 100, BreakKind::H1)]; // before window
        assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &[]), 200);
    }

    #[test]
    fn cutoff_skips_break_points_inside_code_fences() {
        let bps = [mk(150, 100, BreakKind::H1), mk(180, 20, BreakKind::Blank)];
        let fences = [CodeFenceRegion {
            start: 140,
            end: 160,
        }];
        assert_eq!(find_best_cutoff(&bps, 200, 100, 0.7, &fences), 180);
    }

    #[test]
    fn cutoff_handles_empty_break_points() {
        assert_eq!(find_best_cutoff(&[], 200, 100, 0.7, &[]), 200);
    }

    // --- mergeBreakPoints: ported from store.test.ts `describe("mergeBreakPoints")` ---

    #[test]
    fn merge_keeps_highest_score_per_position() {
        let regex = [mk(10, 20, BreakKind::Blank), mk(50, 1, BreakKind::Newline)];
        let ast = [
            mk(10, 90, BreakKind::AstFunc),
            mk(100, 100, BreakKind::AstClass),
        ];
        let merged = merge_break_points(&regex, &ast);
        assert_eq!(merged.len(), 3);
        let at10 = merged.iter().find(|b| b.pos == 10).unwrap();
        assert_eq!(at10.score, 90);
        assert_eq!(at10.kind, BreakKind::AstFunc);
        assert_eq!(merged.iter().find(|b| b.pos == 50).unwrap().score, 1);
        assert_eq!(merged.iter().find(|b| b.pos == 100).unwrap().score, 100);
    }

    #[test]
    fn merge_returns_sorted_by_position() {
        let a = [mk(100, 10, BreakKind::Newline)];
        let b = [mk(5, 20, BreakKind::Blank)];
        let merged = merge_break_points(&a, &b);
        assert_eq!(merged[0].pos, 5);
        assert_eq!(merged[1].pos, 100);
    }

    // --- chunkDocument integration: ported from store.test.ts ---

    #[test]
    fn chunk_prefers_paragraph_breaks() {
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.".repeat(50);
        let chunks = chunk_document(
            &content,
            ChunkStrategy::Auto,
            None,
            Some(500),
            Some(0),
            None,
        );
        assert!(chunks.len() > 1);
    }

    #[test]
    fn chunk_handles_utf8_without_splitting_codepoints() {
        // With overlap 0 the chunks tile exactly; reconstructing must yield the
        // original — only possible if no multi-byte codepoint was split.
        let content = "こんにちは世界".repeat(500);
        let chunks = chunk_document(
            &content,
            ChunkStrategy::Auto,
            None,
            Some(1000),
            Some(0),
            None,
        );
        assert!(chunks.len() > 1);
        let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(joined, content);
    }

    #[test]
    fn chunk_default_params_use_3600_char_chunks() {
        let content = "Word ".repeat(2500); // ~12500 chars
        let chunks = chunk_document(&content, ChunkStrategy::Auto, None, None, None, None);
        assert!(chunks.len() > 1);
        assert!(chunks[0].text.len() > 2800);
        assert!(chunks[0].text.len() <= CHUNK_SIZE_CHARS); // 3600
    }

    #[test]
    fn chunk_prefers_headings_over_arbitrary_breaks() {
        let section1 = "Introduction text here. ".repeat(70); // 1680 chars
        let section2 = "Main content text here. ".repeat(50);
        let content = format!("{section1}\n# Main Section\n{section2}");
        let heading_pos = content.find("\n# Main Section").unwrap();
        let chunks = chunk_document(
            &content,
            ChunkStrategy::Auto,
            None,
            Some(2000),
            Some(0),
            Some(800),
        );
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].text.len(), heading_pos);
    }

    #[test]
    fn chunk_handles_mixed_markdown_elements() {
        let block = r#"# Introduction

This is the introduction paragraph with some text.

## Section 1

Some content in section 1.

- List item 1
- List item 2
- List item 3

## Section 2

```javascript
function hello() {
  console.log("Hello");
}
```

More text after the code block.

---

## Section 3

Final section content.
"#;
        let content = block.repeat(10);
        let chunks = chunk_document(
            &content,
            ChunkStrategy::Auto,
            None,
            Some(500),
            Some(75),
            Some(200),
        );
        assert!(chunks.len() > 5);
        for c in &chunks {
            assert!(!c.text.is_empty());
            assert!(content.is_char_boundary(c.pos));
        }
    }

    // ======================================================================
    // AST-aware chunking integration: ported from tobi/qmd's
    // test/ast-chunking.test.ts
    // ======================================================================

    // A multi-function TS source large enough that AST break points (score 90
    // at function boundaries) shift chunk cutoffs away from the regex
    // blank-line breaks (score 20). Mirrors the `largeTS` fixture in
    // ast-chunking.test.ts.
    fn handlers_sample() -> String {
        let mut parts: Vec<String> = Vec::new();
        for i in 0..30 {
            parts.push(format!(
                "\nexport function handler{i}(req: Request, res: Response): void {{\n  const startTime = Date.now();\n  const userId = req.params.userId;\n  const sessionToken = req.headers.authorization;\n\n  if (!userId || !sessionToken) {{\n    res.status(400).json({{ error: \"Missing required parameters\" }});\n    return;\n  }}\n\n  const result = processBusinessLogic{i}(userId, sessionToken);\n  const elapsed = Date.now() - startTime;\n  res.json({{ data: result, processingTimeMs: elapsed }});\n}}\n"
            ));
        }
        parts.join("\n")
    }

    // Count how many of the 30 functions straddle a chunk boundary.
    // Byte-based throughout (chunk.pos and text.len() are byte offsets).
    fn count_split_functions(source: &str, chunks: &[Chunk]) -> usize {
        let mut splits = 0;
        for i in 0..30 {
            let needle = format!("function handler{i}(");
            let Some(func_start) = source.find(&needle) else {
                continue;
            };
            let next_needle = format!("function handler{}(", i + 1);
            let func_end = source[func_start + 1..]
                .find(&next_needle)
                .map(|p| func_start + 1 + p)
                .unwrap_or(source.len());
            let mut covering = std::collections::HashSet::new();
            for (ci, c) in chunks.iter().enumerate() {
                let chunk_start = c.pos;
                let chunk_end = chunk_start + c.text.len();
                if chunk_start < func_end && chunk_end > func_start {
                    covering.insert(ci);
                }
            }
            if covering.len() > 1 {
                splits += 1;
            }
        }
        splits
    }

    // --- mergeBreakPoints: ast-chunking.test.ts variant (4 positions) ---

    #[test]
    fn merge_ast_variant_higher_score_wins() {
        let regex = [
            mk(10, 20, BreakKind::Blank),
            mk(50, 1, BreakKind::Newline),
            mk(100, 20, BreakKind::Blank),
        ];
        let ast = [
            mk(10, 90, BreakKind::AstFunc),
            mk(75, 100, BreakKind::AstClass),
            mk(100, 60, BreakKind::AstImport),
        ];
        let merged = merge_break_points(&regex, &ast);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged.iter().find(|b| b.pos == 10).unwrap().score, 90); // AST wins (90>20)
        assert_eq!(merged.iter().find(|b| b.pos == 50).unwrap().score, 1); // regex only
        assert_eq!(merged.iter().find(|b| b.pos == 75).unwrap().score, 100); // AST only
        assert_eq!(merged.iter().find(|b| b.pos == 100).unwrap().score, 60); // AST wins (60>20)
    }

    // --- AST vs Regex chunking ---

    #[test]
    fn ast_splits_fewer_functions_than_regex() {
        let large_ts = handlers_sample();
        let regex_chunks = chunk_document(&large_ts, ChunkStrategy::Auto, None, None, None, None);
        let ast_chunks = chunk_document(
            &large_ts,
            ChunkStrategy::Auto,
            Some("handlers.ts"),
            None,
            None,
            None,
        );
        assert!(
            count_split_functions(&large_ts, &ast_chunks)
                <= count_split_functions(&large_ts, &regex_chunks)
        );
        // Prove AST actually moved boundaries — otherwise the `<=` above (and
        // the bypass test below) would pass trivially.
        assert_ne!(ast_chunks, regex_chunks);
    }

    #[test]
    fn markdown_identical_ast_vs_regex() {
        let mut sections: Vec<String> = Vec::new();
        for i in 0..15 {
            sections.push(format!(
                "# Section {i}\n\n{}\n",
                "Lorem ipsum dolor sit amet. ".repeat(40)
            ));
        }
        let large_md = sections.join("\n");
        let md_regex = chunk_document(&large_md, ChunkStrategy::Auto, None, None, None, None);
        let md_ast = chunk_document(
            &large_md,
            ChunkStrategy::Auto,
            Some("readme.md"),
            None,
            None,
            None,
        );
        assert_eq!(md_ast, md_regex);
    }

    #[test]
    fn regex_strategy_bypasses_ast() {
        let large_ts = handlers_sample();
        let regex_baseline =
            chunk_document(&large_ts, ChunkStrategy::Regex, None, None, None, None);
        let auto_with_ast = chunk_document(
            &large_ts,
            ChunkStrategy::Auto,
            Some("handlers.ts"),
            None,
            None,
            None,
        );
        let regex_with_filepath = chunk_document(
            &large_ts,
            ChunkStrategy::Regex,
            Some("handlers.ts"),
            None,
            None,
            None,
        );
        // Sanity: AST is genuinely active in Auto mode for this sample.
        assert_ne!(auto_with_ast, regex_baseline);
        // The Regex strategy bypasses AST even with a code filepath present.
        assert_eq!(regex_with_filepath, regex_baseline);
    }

    #[test]
    fn no_filepath_falls_back_to_regex() {
        let large_ts = handlers_sample();
        let no_path = chunk_document(&large_ts, ChunkStrategy::Auto, None, None, None, None);
        let regex = chunk_document(&large_ts, ChunkStrategy::Regex, None, None, None, None);
        assert_eq!(no_path, regex);
    }

    #[test]
    fn small_file_single_chunk_ast() {
        let chunks = chunk_document(
            "export const x = 1;",
            ChunkStrategy::Auto,
            Some("s.ts"),
            None,
            None,
            None,
        );
        assert_eq!(chunks.len(), 1);
    }

    // --- chunkDocumentWithBreakPoints equivalence ---

    #[test]
    fn chunk_with_break_points_matches_chunk_document() {
        let content = "a".repeat(5000) + "\n\n" + &"b".repeat(5000);
        let via_chunk_document =
            chunk_document(&content, ChunkStrategy::Auto, None, None, None, None);
        let via_break_points = chunk_document_with_break_points(
            &content,
            &scan_break_points(&content),
            &find_code_fences(&content),
            CHUNK_SIZE_CHARS,
            CHUNK_OVERLAP_CHARS,
            CHUNK_WINDOW_CHARS,
        );
        assert_eq!(via_chunk_document, via_break_points);
    }
}
