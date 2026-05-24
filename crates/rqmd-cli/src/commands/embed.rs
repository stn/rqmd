//! `rqmd embed` — generate / refresh vector embeddings for indexed documents.
//!
//! Output is a faithful port of qmd's `vectorIndex` CLI handler
//! (`src/cli/qmd.ts:1878`) for drop-in parity: force/precheck/model/batch
//! headers, a byte-based progress line with ETA + throughput (plus cursor and
//! OSC 9;4 taskbar progress), and the `✓ Done!` / failure summary. Human
//! output goes to **stdout**; the live progress line and cursor/OSC escapes go
//! to **stderr**. Colour follows qmd's `useColor` (stdout TTY + `NO_COLOR`);
//! progress drawing follows qmd's `isTTY` (stderr TTY).

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rqmd_core::llm::config::resolve_embed_model;
use rqmd_core::llm::format::embedding_fingerprint;
use rqmd_core::store::embeddings::get_pending_embedding_docs;
use rqmd_core::store::{DEFAULT_EMBED_MAX_BATCH_BYTES, DEFAULT_EMBED_MAX_DOCS_PER_BATCH};
use rqmd_core::store_ops::{EmbedOptions, EmbedProgress, generate_embeddings};

use crate::cli::EmbedArgs;
use crate::format_helpers::{format_bytes, format_count, format_eta, short_model_name};
use crate::state::IndexState;

pub async fn run(args: EmbedArgs, state: &mut IndexState) -> Result<()> {
    // Reject `-c X -c Y` rather than silently dropping all but the first.
    if args.collection.len() > 1 {
        bail!(
            "embed accepts at most one --collection (got {})",
            args.collection.len()
        );
    }

    // Validate batch limits up front (qmd `parseEmbedBatchOption`), before the
    // "already embedded" precheck / model resolve — so invalid flags fail fast
    // regardless of index state. The same check also runs in
    // `generate_embeddings` (defense-in-depth for the MCP / RqmdStore path).
    if args.max_docs_per_batch == Some(0) {
        bail!("maxDocsPerBatch must be a positive integer");
    }
    if args.max_batch_mb == Some(0) {
        bail!("maxBatchBytes must be a positive integer");
    }

    let collection = args.collection.into_iter().next();
    let chunk_strategy = args.chunk_strategy.map(Into::into);

    // Colour gating mirrors qmd `useColor` (stdout TTY + NO_COLOR); progress
    // drawing mirrors qmd `isTTY` (stderr TTY).
    let col = EmbedColor::new();
    let is_tty = std::io::stderr().is_terminal();

    if args.force {
        println!(
            "{}Force re-indexing: clearing all vectors...{}",
            col.yellow(),
            col.reset()
        );
    }

    // Resolve the embed model exactly as `generate_embeddings` will (it uses
    // `resolve_embed_model(None)` when `opts.model` is None), so the precheck
    // and the `Model:` header agree with what actually gets embedded.
    let model = resolve_embed_model(None);
    let fingerprint = embedding_fingerprint(&model);

    let llm = state.llama_cpp()?;
    let store = state.store_mut()?;

    // Precheck: when nothing needs embedding (and not forcing), say so and
    // stop — qmd's `getHashesNeedingEmbedding == 0 && !force` path.
    if !args.force {
        let needing = store
            .with_connection(|c| {
                get_pending_embedding_docs(c, collection.as_deref(), &model, &fingerprint)
            })
            .context("counting documents needing embedding")?
            .len();
        if needing == 0 {
            println!(
                "{}✓ All content hashes already have embeddings.{}",
                col.green(),
                col.reset()
            );
            return Ok(());
        }
    }

    println!(
        "{}Model: {}{}\n",
        col.dim(),
        short_model_name(&model),
        col.reset()
    );
    if args.max_docs_per_batch.is_some() || args.max_batch_mb.is_some() {
        let max_docs = args
            .max_docs_per_batch
            .unwrap_or(DEFAULT_EMBED_MAX_DOCS_PER_BATCH);
        let max_bytes = args
            .max_batch_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(DEFAULT_EMBED_MAX_BATCH_BYTES);
        println!(
            "{}Batch: {} docs / {}{}\n",
            col.dim(),
            max_docs,
            format_bytes(max_bytes as u64),
            col.reset()
        );
    }

    // cursor.hide() (unconditional, matching qmd) + progress.indeterminate()
    // (OSC 9;4, stderr-TTY-gated). stderr is unbuffered, so no flush needed.
    eprint!("\x1b[?25l");
    if is_tty {
        eprint!("\x1b]9;4;3\x07");
    }

    let start = Instant::now();

    let on_progress: Arc<dyn Fn(EmbedProgress) + Send + Sync> =
        Arc::new(move |ep: EmbedProgress| {
            if ep.total_bytes == 0 {
                return;
            }
            // Percent is byte-based (qmd): the final chunk count is discovered
            // lazily, so chunks/total would look wrong while large docs remain.
            let percent = ((ep.bytes_processed as f64 / ep.total_bytes as f64) * 100.0).min(100.0);
            if is_tty {
                eprint!("\x1b]9;4;1;{}\x07", percent.round() as i64);
            }

            let elapsed = start.elapsed().as_secs_f64();
            let bytes_per_sec = if elapsed > 0.0 {
                ep.bytes_processed as f64 / elapsed
            } else {
                0.0
            };
            let remaining = (ep.total_bytes as f64 - ep.bytes_processed as f64).max(0.0);
            let eta_sec = if bytes_per_sec > 0.0 {
                remaining / bytes_per_sec
            } else {
                f64::INFINITY
            };

            let bar = render_progress_bar(percent, 30);
            let percent_str = format!("{:>3}", percent.round() as i64);
            let throughput = if bytes_per_sec > 0.0 {
                format!("{}/s", format_bytes(bytes_per_sec as u64))
            } else {
                ".../s".to_string()
            };
            let eta = if elapsed > 2.0 && eta_sec.is_finite() {
                format_eta(eta_sec)
            } else {
                "...".to_string()
            };
            let input_str = format!(
                "{}/{} input",
                format_bytes(ep.bytes_processed as u64),
                format_bytes(ep.total_bytes as u64)
            );
            let chunk_str = format!("{} chunks", format_count(ep.chunks_embedded));
            let err_str = if ep.errors > 0 {
                format!(
                    " {}{} err{}",
                    col.yellow(),
                    format_count(ep.errors),
                    col.reset()
                )
            } else {
                String::new()
            };

            if is_tty {
                eprint!(
                    "\r{}{}{} {}{}% input{} {}{}{} · {} · {} · ETA {}{}   ",
                    col.cyan(),
                    bar,
                    col.reset(),
                    col.bold(),
                    percent_str,
                    col.reset(),
                    col.dim(),
                    chunk_str,
                    err_str,
                    input_str,
                    throughput,
                    eta,
                    col.reset(),
                );
            }
        });

    let opts = EmbedOptions {
        force: args.force,
        // `generate_embeddings` resolves the embed model the same way we did
        // above (resolve_embed_model(None)); leave this None so they agree.
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

    // progress.clear() (OSC, TTY-gated) + cursor.show() (unconditional).
    if is_tty {
        eprint!("\x1b]9;4;0\x07");
    }
    eprint!("\x1b[?25h");

    let total_time_sec = result.duration_ms as f64 / 1000.0;

    if result.chunks_embedded == 0 && result.docs_processed == 0 {
        println!(
            "{}✓ No non-empty documents to embed.{}",
            col.green(),
            col.reset()
        );
    } else {
        // Finalised 100% bar (overwrites the in-place progress line), then the
        // summary. 36 trailing spaces wipe any residue of the longer line.
        println!(
            "\r{}{}{} {}100%{}{}",
            col.green(),
            render_progress_bar(100.0, 30),
            col.reset(),
            col.bold(),
            col.reset(),
            " ".repeat(36),
        );
        println!(
            "\n{}✓ Done!{} Embedded {}{}{} chunks from {}{}{} documents in {}{}{}",
            col.green(),
            col.reset(),
            col.bold(),
            result.chunks_embedded,
            col.reset(),
            col.bold(),
            result.docs_processed,
            col.reset(),
            col.bold(),
            format_eta(total_time_sec),
            col.reset(),
        );
        if result.errors > 0 {
            println!(
                "{}⚠ {} chunks still failed after retries{}",
                col.yellow(),
                format_count(result.errors),
                col.reset()
            );
            for f in result.failures.iter().take(8) {
                println!(
                    "  {}{}#{} ({} attempts): {}{}",
                    col.dim(),
                    f.path,
                    f.seq,
                    f.attempts,
                    f.reason,
                    col.reset(),
                );
            }
            if result.failures.len() > 8 {
                println!(
                    "  {}...and {} more{}",
                    col.dim(),
                    format_count(result.failures.len() - 8),
                    col.reset(),
                );
            }
        }
    }
    Ok(())
}

/// stdout-gated ANSI colours, matching qmd's `useColor`
/// (`!NO_COLOR && process.stdout.isTTY`). Distinct from the shared stderr-gated
/// [`crate::color::Palette`] because qmd routes embed's human output to stdout.
#[derive(Clone, Copy)]
struct EmbedColor {
    enabled: bool,
}

impl EmbedColor {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal(),
        }
    }
    fn code(self, s: &'static str) -> &'static str {
        if self.enabled { s } else { "" }
    }
    fn reset(self) -> &'static str {
        self.code("\x1b[0m")
    }
    fn dim(self) -> &'static str {
        self.code("\x1b[2m")
    }
    fn bold(self) -> &'static str {
        self.code("\x1b[1m")
    }
    fn cyan(self) -> &'static str {
        self.code("\x1b[36m")
    }
    fn yellow(self) -> &'static str {
        self.code("\x1b[33m")
    }
    fn green(self) -> &'static str {
        self.code("\x1b[32m")
    }
}

/// 30-wide bar of `█` (filled) / `░` (empty). Port of qmd `renderProgressBar`
/// (`src/cli/qmd.ts:1817`); filled count uses round-to-nearest.
fn render_progress_bar(percent: f64, width: usize) -> String {
    let filled = (((percent / 100.0) * width as f64).round() as usize).min(width);
    let empty = width - filled;
    let mut s = String::with_capacity(width * 3);
    for _ in 0..filled {
        s.push('\u{2588}'); // █
    }
    for _ in 0..empty {
        s.push('\u{2591}'); // ░
    }
    s
}
