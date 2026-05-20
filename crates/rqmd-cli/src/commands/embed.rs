//! `rqmd embed` — generate / refresh vector embeddings for indexed documents.
//!
//! Maps to qmd's `vectorIndex` CLI handler (`src/cli/qmd.ts` lines 3545–3567)
//! and `generateEmbeddings` orchestration in `src/store.ts` (lines 1511–1700).

use std::io::IsTerminal;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rqmd_core::store_ops::{generate_embeddings, EmbedOptions, EmbedProgress};

use crate::cli::EmbedArgs;
use crate::color::Palette;
use crate::format_helpers::format_bytes;
use crate::state::IndexState;

pub async fn run(args: EmbedArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    // Reject `-c X -c Y` rather than silently dropping all but the first.
    if args.collection.len() > 1 {
        bail!(
            "embed accepts at most one --collection (got {})",
            args.collection.len()
        );
    }
    let collection = args.collection.into_iter().next();

    let chunk_strategy = args.chunk_strategy.map(Into::into);

    let llm = state.llama_cpp()?;
    let store = state.store_mut()?;

    // Progress rendering: in-place updates on TTY, line-per-event when piped.
    //
    // Throttle is `AtomicU64` of millis since `start`. Lock-free; the
    // single-writer pattern (generate_embeddings invokes the callback
    // sequentially) means a CAS isn't strictly needed, but the atomic
    // store gives us Send+Sync without a Mutex panic surface.
    let is_tty = std::io::stderr().is_terminal();
    let start = Instant::now();
    let last_ms = Arc::new(AtomicU64::new(0));
    let on_progress: Arc<dyn Fn(EmbedProgress) + Send + Sync> = {
        let last_ms = last_ms.clone();
        Arc::new(move |ep: EmbedProgress| {
            // "Done" must include errors because failed chunks only
            // increment `errors`, never `chunks_embedded`
            // (see rqmd-core/src/store_ops/embed.rs lines 293/300/345/367).
            // Without this, partial-failure runs would drop the final
            // progress frame inside the 100 ms throttle window.
            let done = ep.total_chunks > 0
                && ep.chunks_embedded.saturating_add(ep.errors) >= ep.total_chunks;
            let now_ms = start.elapsed().as_millis() as u64;
            let prev = last_ms.load(Ordering::Acquire);
            let due = now_ms.saturating_sub(prev) >= 100;
            if !due && !done {
                return;
            }
            last_ms.store(now_ms, Ordering::Release);

            // Bar reflects "settled" work (both successful and errored chunks)
            // so a 60/40 split renders as 100% at completion, not 60%.
            let settled = ep.chunks_embedded.saturating_add(ep.errors);
            let pct = if ep.total_chunks > 0 {
                settled * 100 / ep.total_chunks
            } else {
                0
            };
            let bar = render_bar(pct, 20);
            let line = format!(
                "Embedding: {} {pct:>3}% {}/{}  {} / {}  errors:{}",
                bar,
                ep.chunks_embedded,
                ep.total_chunks,
                format_bytes(ep.bytes_processed as u64),
                format_bytes(ep.total_bytes as u64),
                ep.errors,
            );
            if is_tty {
                eprint!("\r\x1b[K{line}");
            } else {
                eprintln!("{line}");
            }
        })
    };

    let opts = EmbedOptions {
        force: args.force,
        // `llama_cpp()` already pinned the embed model URI on the
        // LlamaCpp handle; leave this `None` so `generate_embeddings`
        // resolves it the same way.
        model: None,
        collection,
        max_docs_per_batch: args.max_docs_per_batch,
        max_batch_bytes: args.max_batch_mb.map(|mb| mb * 1024 * 1024),
        chunk_strategy,
        on_progress: Some(on_progress),
    };

    let result = generate_embeddings(store, llm, opts)
        .await
        .context("embed failed")?;

    if is_tty {
        // Terminate the in-place progress line so the summary starts cleanly.
        eprintln!();
    }

    // Summary goes to stderr alongside the progress events so
    // `rqmd embed | tee log` produces a coherent log (stdout from embed is
    // intentionally empty — it carries no machine-readable data).
    eprintln!(
        "{}\u{2713}{} Embedded {} chunks from {} documents in {}ms",
        p.green(),
        p.reset(),
        result.chunks_embedded,
        result.docs_processed,
        result.duration_ms,
    );
    if result.errors > 0 {
        eprintln!(
            "{}\u{26A0}{} {} chunks failed",
            p.yellow(),
            p.reset(),
            result.errors,
        );
    }
    Ok(())
}

fn render_bar(pct: usize, width: usize) -> String {
    let filled = (pct.min(100) * width) / 100;
    let mut s = String::with_capacity(width * 4 + 2);
    s.push('[');
    for i in 0..width {
        if i < filled {
            s.push('\u{2588}'); // █
        } else {
            s.push('\u{00B7}'); // ·
        }
    }
    s.push(']');
    s
}
