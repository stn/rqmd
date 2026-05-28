//! `rqmd multi-get` — batch fetch documents by glob or comma-separated list.
//!
//! Maps to qmd's `multi-get` in `src/cli/qmd.ts` (lines 1106–1337, 3380–3390).
//! Resolution is fully handled by `rqmd_core::store::lookup::find_documents`.

use anyhow::Result;
use rqmd_core::FindDocumentsOptions;
use rqmd_core::store::DEFAULT_MULTI_GET_MAX_BYTES;
use rqmd_core::store::lookup::find_documents;

use crate::cli::MultiGetArgs;
use crate::output::{OutputFormat, write_multi_get};
use crate::state::IndexState;

pub fn run(a: MultiGetArgs, state: &mut IndexState) -> Result<()> {
    let store = state.store_mut()?;
    let max_bytes = a.max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES);
    let result = store.with_connection(|conn| {
        find_documents(
            conn,
            &a.pattern,
            FindDocumentsOptions {
                include_body: true,
                max_bytes,
            },
        )
    })?;

    for err in &result.errors {
        eprintln!("{err}");
    }

    let format = OutputFormat::from(&a.format);
    write_multi_get(&result.docs, a.lines, format);
    Ok(())
}
