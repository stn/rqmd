//! `chunk_document_by_tokens`: char-based chunking + per-chunk tokenizer
//! refinement.
//!
//! Port of `tobi/qmd/src/store.ts` lines 2412–2500. Recursion is unrolled
//! into an explicit stack to avoid `BoxFuture` recursion through `async fn`.

use std::sync::Arc;

use crate::store::chunking::{chunk_document, ChunkStrategy};
use crate::store::{
    CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CHUNK_WINDOW_TOKENS,
};
use tokio_util::sync::CancellationToken;

use crate::llm::traits::Llm;

use super::Result;

const AVG_CHARS_PER_TOKEN: usize = 3;

/// One chunk produced by token-aware chunking. `tokens` is the LLM's
/// tokenizer count of `text`, guaranteed `≤ max_tokens` unless the
/// detokenize-fallback path fired (in which case it equals `max_tokens`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk {
    pub text: String,
    pub pos: usize,
    pub tokens: usize,
}

/// Chunk `content` so every chunk fits in `max_tokens` tokens of `llm`'s
/// tokenizer. Mirrors TS `chunkDocumentByTokens` (`store.ts:2412`).
///
/// Two-phase: (1) char-based `chunk_document` with `max_tokens * 3` chars,
/// (2) per-chunk re-tokenize and recursive split if over budget. The
/// recursion uses an explicit stack — Rust's async fn cannot recurse
/// without `BoxFuture`, and the stack form is closer to TS's recursive
/// shape anyway.
///
/// `cancel`: polled once per stack iteration. Mid-tokenize cancellation
/// is not possible (the FFI call is opaque); on cancel we return the
/// partial result accumulated so far.
#[allow(clippy::too_many_arguments)]
pub async fn chunk_document_by_tokens(
    llm: Arc<dyn Llm>,
    content: &str,
    max_tokens: Option<usize>,
    overlap_tokens: Option<usize>,
    window_tokens: Option<usize>,
    filepath: Option<&str>,
    strategy: ChunkStrategy,
    cancel: Option<&CancellationToken>,
) -> Result<Vec<TokenChunk>> {
    let max_tokens = max_tokens.unwrap_or(CHUNK_SIZE_TOKENS);
    let overlap_tokens = overlap_tokens.unwrap_or(CHUNK_OVERLAP_TOKENS);
    let window_tokens = window_tokens.unwrap_or(CHUNK_WINDOW_TOKENS);
    let max_chars = max_tokens.saturating_mul(AVG_CHARS_PER_TOKEN);
    let overlap_chars = overlap_tokens.saturating_mul(AVG_CHARS_PER_TOKEN);
    let window_chars = window_tokens.saturating_mul(AVG_CHARS_PER_TOKEN);

    let initial = chunk_document(
        content,
        strategy,
        filepath,
        Some(max_chars),
        Some(overlap_chars),
        Some(window_chars),
    );
    let mut results: Vec<TokenChunk> = Vec::with_capacity(initial.len());
    let mut stack: Vec<(String, usize)> = initial
        .into_iter()
        .rev()
        .map(|c| (c.text, c.pos))
        .collect();

    while let Some((text, pos)) = stack.pop() {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Ok(results);
        }

        let tokens = llm.tokenize(&text).await?;
        if tokens.len() <= max_tokens || text.len() <= 1 {
            results.push(TokenChunk {
                text,
                pos,
                tokens: tokens.len(),
            });
            continue;
        }

        // Refine using actual chars-per-token from this chunk.
        let actual = (text.len() as f64 / tokens.len() as f64).max(0.1);
        let mut safe_max_chars = ((max_tokens as f64) * actual * 0.95).floor() as usize;
        if safe_max_chars == 0 || safe_max_chars >= text.len() {
            safe_max_chars = (text.len() / 2).max(1);
        }
        let next_overlap = ((overlap_chars as f64) * actual / 2.0).floor() as usize;
        let next_overlap = next_overlap.min(safe_max_chars.saturating_sub(1));
        let next_window = ((window_chars as f64) * actual / 2.0).max(0.0) as usize;

        let mut sub = chunk_document(
            &text,
            strategy,
            filepath,
            Some(safe_max_chars),
            Some(next_overlap),
            Some(next_window),
        );

        let unchanged = sub.len() <= 1
            || sub
                .first()
                .map(|c| c.text.len() == text.len())
                .unwrap_or(false);
        if unchanged {
            // Half-split fallback.
            let half = (text.len() / 2).max(1);
            sub = chunk_document(&text, strategy, filepath, Some(half), Some(0), Some(0));
        }
        let still_unchanged = sub.len() <= 1
            || sub
                .first()
                .map(|c| c.text.len() == text.len())
                .unwrap_or(false);
        if still_unchanged {
            // Detokenize fallback: emit a max_tokens-truncated chunk and move on.
            let take = max_tokens.max(1).min(tokens.len());
            let truncated_text = llm.detokenize(&tokens[..take]).await?;
            results.push(TokenChunk {
                text: truncated_text,
                pos,
                tokens: take,
            });
            continue;
        }

        // Push sub-chunks back onto the stack in reverse for front-to-back
        // processing. Each sub-chunk's text is sliced from the parent so its
        // absolute position is `pos + sc.pos`.
        for sc in sub.into_iter().rev() {
            let start = sc.pos;
            let end = (start + sc.text.len()).min(text.len());
            let slice = text[start..end].to_string();
            stack.push((slice, pos + start));
        }
    }
    Ok(results)
}
