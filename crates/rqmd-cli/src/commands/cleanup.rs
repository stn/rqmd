//! `rqmd cleanup` — clear LLM cache, drop orphan vectors / inactive docs, vacuum.
//!
//! Maps to qmd's `qmd cleanup` block in `src/cli/qmd.ts` (lines 3786–3813).

use anyhow::Result;
use rqmd_core::store::{cache, maintenance};

use crate::color::Palette;
use crate::state::IndexState;

pub fn run(state: &mut IndexState, p: &Palette) -> Result<()> {
    let store = state.store_mut()?;
    let vec_available = store.vec_available;

    let (cache_count, orphaned_vecs, inactive_docs) = store.with_connection_mut(|conn| {
        let cache_count = cache::delete_llm_cache(conn)?;
        let orphaned_vecs = maintenance::cleanup_orphaned_vectors(conn, vec_available)?;
        let inactive_docs = maintenance::delete_inactive_documents(conn)?;
        maintenance::vacuum_database(conn)?;
        Ok::<_, rqmd_core::store::Error>((cache_count, orphaned_vecs, inactive_docs))
    })?;

    println!(
        "{}✓{} Cleared {cache_count} cached API responses",
        p.green(),
        p.reset()
    );
    if orphaned_vecs > 0 {
        println!(
            "{}✓{} Removed {orphaned_vecs} orphaned embedding chunks",
            p.green(),
            p.reset()
        );
    } else {
        println!("{}No orphaned embeddings to remove{}", p.dim(), p.reset());
    }
    if inactive_docs > 0 {
        println!(
            "{}✓{} Removed {inactive_docs} inactive document records",
            p.green(),
            p.reset()
        );
    }
    println!("{}✓{} Database vacuumed", p.green(), p.reset());
    Ok(())
}
