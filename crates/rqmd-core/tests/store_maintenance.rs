//! Equivalent tests for `cleanup_orphaned_vectors`.
//!
//! The TS unit test (`store.helpers.unit.test.ts`) mocks `db.prepare` to
//! simulate sqlite-vec being unavailable. A rusqlite `Connection` cannot be
//! mocked, so we exercise the three branches against a real in-memory store
//! instead — `false`, table-absent, and a real orphan deletion — which also
//! distinguishes the early-return / swallow / delete paths (a single
//! "returns 0" assertion could not).

use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::embeddings::{ensure_vec_table, insert_embedding};
use rqmd_core::store::maintenance::cleanup_orphaned_vectors;
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::Store;
use tempfile::NamedTempFile;

fn open() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

/// Branch 1: `vec_available = false` → early return, no-op.
#[test]
fn cleanup_returns_zero_when_vec_unavailable() {
    let (_t, store) = open();
    let n = store
        .with_connection(|c| cleanup_orphaned_vectors(c, false))
        .unwrap();
    assert_eq!(n, 0);
}

/// Branch 2: `vec_available = true` but `vectors_vec` does not exist yet (the
/// schema only creates it on the first embed) → the probe query fails and
/// cleanup degrades to a no-op. Mirrors the TS "schema entry exists but
/// sqlite-vec module unavailable" case.
#[test]
fn cleanup_returns_zero_when_vec_table_absent() {
    let (_t, store) = open();
    let exists: i64 = store.with_connection(|c| {
        c.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )
        .unwrap()
    });
    assert_eq!(exists, 0, "fresh store should have no vectors_vec table");

    let n = store
        .with_connection(|c| cleanup_orphaned_vectors(c, true))
        .unwrap();
    assert_eq!(n, 0);
}

/// Branch 3: `vec_available = true`, `vectors_vec` exists, and an orphaned
/// embedding (no active document for its hash) is present → it is deleted and
/// counted, while an embedding belonging to an active document is preserved.
#[test]
fn cleanup_deletes_orphaned_vectors_and_keeps_active() {
    let (_t, store) = open();
    let now = now_rfc3339();

    store.with_connection(|c| {
        ensure_vec_table(c, 4).unwrap();

        // Active document + its embedding — must survive.
        let keep_hash = hash_content("keep body");
        insert_content(c, &keep_hash, "keep body", &now).unwrap();
        insert_document(c, "docs", "keep.md", "Keep", &keep_hash, &now, &now).unwrap();
        insert_embedding(c, &keep_hash, 0, 0, &[1.0; 4], "m", &now, 1).unwrap();

        // Orphan embedding — content row exists but no document references it.
        let orphan_hash = hash_content("orphan body");
        insert_content(c, &orphan_hash, "orphan body", &now).unwrap();
        insert_embedding(c, &orphan_hash, 0, 0, &[2.0; 4], "m", &now, 1).unwrap();
    });

    let removed = store
        .with_connection(|c| cleanup_orphaned_vectors(c, true))
        .unwrap();
    assert_eq!(removed, 1, "exactly the orphan embedding should be removed");

    let remaining: i64 = store.with_connection(|c| {
        c.query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap()
    });
    assert_eq!(remaining, 1, "the active document's embedding must remain");
}
