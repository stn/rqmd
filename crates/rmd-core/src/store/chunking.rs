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

/// Chunking strategy. The token-based variant is out of scope this pass;
/// `Auto` and `Regex` both currently use the regex (manual-scan) algorithm.
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
pub fn chunk_document(
    content: &str,
    _strategy: ChunkStrategy,
    max_chars: Option<usize>,
    overlap_chars: Option<usize>,
    window_chars: Option<usize>,
) -> Vec<Chunk> {
    let break_points = scan_break_points(content);
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
        let chunks = chunk_document("hello", ChunkStrategy::Auto, Some(100), Some(10), Some(20));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello");
    }

    #[test]
    fn chunk_long_doc_overlaps() {
        let text = "a".repeat(50) + "\n\n" + &"b".repeat(50);
        let chunks = chunk_document(&text, ChunkStrategy::Auto, Some(40), Some(8), Some(16));
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.text.len() <= 40 || c.pos == 0);
        }
    }
}
