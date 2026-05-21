//! `rqmd get` — show a single document.
//!
//! Maps to qmd's `qmd get` in `src/cli/qmd.ts` (lines 919–1103, 3369–3378).
//! Most resolution work lives in `rqmd_core::store::lookup::find_document`.

use anyhow::{Result, anyhow};
use rqmd_core::store::lookup::{FindDocumentOptions, FindDocumentOutcome, find_document};
use rqmd_core::store::snippet::add_line_numbers;
use rqmd_core::store::virtual_path::{build_virtual_path, is_virtual_path, parse_virtual_path};

use crate::cli::GetArgs;
use crate::color::Palette;
use crate::state::IndexState;

pub fn run(a: GetArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    // Strip a trailing `:NN` line suffix and use it as `--from` if not already set.
    let (clean_input, parsed_from) = strip_line_suffix(&a.file);
    let from_line = a.from.or(parsed_from).map(|n| n.max(1));

    // Honour a `?index=` carried in a `qmd://` link (qmd `getDocument`,
    // `qmd.ts:931`): switch the active index *before* opening the store, then
    // strip the query for lookup — `find_document` matches the raw string
    // against `qmd://collection/path`, so a `?index=` suffix would never match.
    let lookup_input = if is_virtual_path(&clean_input) {
        match parse_virtual_path(&clean_input) {
            Ok(vp) => {
                if let Some(idx) = &vp.index_name {
                    state.set_index_name(idx);
                }
                build_virtual_path(&vp.collection, &vp.path, None)
            }
            Err(_) => clean_input.clone(),
        }
    } else {
        clean_input.clone()
    };

    let store = state.store_mut()?;
    let outcome = store.with_connection(|conn| {
        find_document(
            conn,
            &lookup_input,
            FindDocumentOptions { include_body: true },
        )
    })?;

    match outcome {
        FindDocumentOutcome::Found(doc) => {
            let body = doc.body.as_deref().unwrap_or("");
            let sliced = slice_lines(body, from_line, a.lines);
            let output = if a.line_numbers {
                add_line_numbers(&sliced, from_line)
            } else {
                sliced
            };
            if let Some(ctx) = &doc.context {
                println!("Folder Context: {ctx}\n---\n");
            }
            println!("{output}");
            Ok(())
        }
        FindDocumentOutcome::NotFound(nf) => {
            let mut msg = format!("Document not found: {}", nf.query);
            if !nf.similar_files.is_empty() {
                msg.push_str(&format!(
                    "\n{}Did you mean:{}\n  {}",
                    p.dim(),
                    p.reset(),
                    nf.similar_files.join("\n  ")
                ));
            }
            Err(anyhow!(msg))
        }
    }
}

fn strip_line_suffix(input: &str) -> (String, Option<usize>) {
    if let Some(idx) = input.rfind(':')
        && let Some(suffix) = input.get(idx + 1..)
        && !suffix.is_empty()
        && suffix.bytes().all(|b| b.is_ascii_digit())
        && let Ok(n) = suffix.parse::<usize>()
    {
        return (input[..idx].to_string(), Some(n));
    }
    (input.to_string(), None)
}

fn slice_lines(body: &str, from_line: Option<usize>, max_lines: Option<usize>) -> String {
    if from_line.is_none() && max_lines.is_none() {
        return body.to_string();
    }
    let lines: Vec<&str> = body.split('\n').collect();
    let start = from_line.unwrap_or(1).saturating_sub(1);
    if start >= lines.len() {
        return String::new();
    }
    let end = match max_lines {
        Some(n) => (start + n).min(lines.len()),
        None => lines.len(),
    };
    lines[start..end].join("\n")
}
