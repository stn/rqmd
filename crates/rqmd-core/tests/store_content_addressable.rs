//! Integration tests for content-addressable storage and legacy-path
//! migration, ported from store.test.ts `describe("Content-Addressable
//! Storage")`.

use rqmd_core::Store;
use rqmd_core::db::rusqlite::params;
use rqmd_core::store::documents::{
    deactivate_document, find_active_document, find_or_migrate_legacy_document, hash_content,
    insert_content, insert_document,
};
use rqmd_core::store::path::now_rfc3339;
use tempfile::NamedTempFile;

fn open() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

fn add_doc(store: &Store, collection: &str, path: &str, body: &str) -> String {
    let now = now_rfc3339();
    let hash = hash_content(body);
    store
        .with_connection(|c| insert_content(c, &hash, body, &now))
        .unwrap();
    store
        .with_connection(|c| insert_document(c, collection, path, "title", &hash, &now, &now))
        .unwrap();
    hash
}

fn count0(store: &Store, sql: &str) -> i64 {
    store
        .with_connection(|c| c.query_row(sql, [], |r| r.get::<_, i64>(0)))
        .unwrap()
}

fn count1(store: &Store, sql: &str, a: &str) -> i64 {
    store
        .with_connection(|c| c.query_row(sql, params![a], |r| r.get::<_, i64>(0)))
        .unwrap()
}

fn count2(store: &Store, sql: &str, a: &str, b: &str) -> i64 {
    store
        .with_connection(|c| c.query_row(sql, params![a, b], |r| r.get::<_, i64>(0)))
        .unwrap()
}

#[test]
fn same_content_gets_same_hash_across_collections() {
    let (_t, store) = open();
    let content = "# Same Content\n\nThis is the same content in two places.";
    let h1 = add_doc(&store, "collection1", "doc1.md", content);
    let h2 = add_doc(&store, "collection2", "doc2.md", content);

    assert_eq!(h1, h2);
    assert_eq!(h1, hash_content(content));
    // Only one content row for the shared hash.
    assert_eq!(
        count1(&store, "SELECT COUNT(*) FROM content WHERE hash = ?", &h1),
        1
    );
}

#[test]
fn removing_one_collection_preserves_content_used_by_another() {
    let (_t, store) = open();
    let shared = "# Shared Content\n\nThis is shared.";
    let shared_hash = add_doc(&store, "collection1", "shared1.md", shared);
    add_doc(&store, "collection2", "shared2.md", shared);

    let unique = "# Unique Content\n\nThis is unique to collection1.";
    let unique_hash = add_doc(&store, "collection1", "unique.md", unique);

    // Remove collection1's documents, then prune orphaned content (CLI behaviour).
    store
        .with_connection(|c| {
            c.execute(
                "DELETE FROM documents WHERE collection = ?",
                params!["collection1"],
            )
            .map(|_| ())
        })
        .unwrap();
    store
        .with_connection(|c| {
            c.execute(
                "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
                [],
            )
            .map(|_| ())
        })
        .unwrap();

    // Shared content survives (collection2 still references it); unique is gone.
    assert_eq!(
        count1(
            &store,
            "SELECT COUNT(*) FROM content WHERE hash = ?",
            &shared_hash
        ),
        1
    );
    assert_eq!(
        count1(
            &store,
            "SELECT COUNT(*) FROM content WHERE hash = ?",
            &unique_hash
        ),
        0
    );
}

#[test]
fn deduplicates_content_across_many_collections() {
    let (_t, store) = open();
    let shared = "# Common Header\n\nThis appears everywhere.";
    let shared_hash = hash_content(shared);
    for i in 0..5 {
        add_doc(
            &store,
            &format!("collection{i}"),
            &format!("doc{i}.md"),
            shared,
        );
    }

    assert_eq!(
        count0(&store, "SELECT COUNT(*) FROM documents WHERE active = 1"),
        5
    );
    assert_eq!(
        count1(
            &store,
            "SELECT COUNT(*) FROM content WHERE hash = ?",
            &shared_hash
        ),
        1
    );
}

#[test]
fn different_content_gets_different_hashes() {
    let (_t, store) = open();
    let h1 = add_doc(&store, "docs", "doc1.md", "# Content One");
    let h2 = add_doc(&store, "docs", "doc2.md", "# Content Two");

    assert_ne!(h1, h2);
    assert_eq!(count0(&store, "SELECT COUNT(*) FROM content"), 2);
}

#[test]
fn reindexing_deactivated_path_reactivates_without_unique_violation() {
    let (_t, store) = open();
    let now = now_rfc3339();

    let old = "# First Version";
    let old_hash = hash_content(old);
    store
        .with_connection(|c| insert_content(c, &old_hash, old, &now))
        .unwrap();
    store
        .with_connection(|c| {
            insert_document(c, "docs", "docs/foo.md", "foo", &old_hash, &now, &now)
        })
        .unwrap();

    // Simulate file removal during an update pass.
    store
        .with_connection(|c| deactivate_document(c, "docs", "docs/foo.md"))
        .unwrap();
    assert!(
        store
            .with_connection(|c| find_active_document(c, "docs", "docs/foo.md"))
            .unwrap()
            .is_none()
    );

    // File comes back: re-insert must reactivate, not violate UNIQUE(collection,path).
    let new = "# Second Version";
    let new_hash = hash_content(new);
    store
        .with_connection(|c| insert_content(c, &new_hash, new, &now))
        .unwrap();
    store
        .with_connection(|c| {
            insert_document(c, "docs", "docs/foo.md", "foo", &new_hash, &now, &now)
        })
        .unwrap();

    let doc = store
        .with_connection(|c| find_active_document(c, "docs", "docs/foo.md"))
        .unwrap()
        .expect("doc should be active again");
    assert_eq!(doc.hash, new_hash);
    assert_eq!(
        count2(
            &store,
            "SELECT COUNT(*) FROM documents WHERE collection = ? AND path = ?",
            "docs",
            "docs/foo.md"
        ),
        1
    );
}

#[test]
fn migrate_renames_lowercase_path_to_case_preserved() {
    let (_t, mut store) = open();
    let now = now_rfc3339();
    let content = "# My Skill";
    let hash = hash_content(content);
    store
        .with_connection(|c| insert_content(c, &hash, content, &now))
        .unwrap();
    // Legacy index: path stored lowercase.
    store
        .with_connection(|c| {
            insert_document(c, "docs", "skills/skill.md", "My Skill", &hash, &now, &now)
        })
        .unwrap();

    let migrated = store
        .with_connection_mut(|c| find_or_migrate_legacy_document(c, "docs", "skills/SKILL.md"))
        .unwrap()
        .expect("migration should find the legacy row");
    assert_eq!(migrated.hash, hash);

    // Old lowercase path no longer active; new case-preserved path is.
    assert!(
        store
            .with_connection(|c| find_active_document(c, "docs", "skills/skill.md"))
            .unwrap()
            .is_none()
    );
    let now_active = store
        .with_connection(|c| find_active_document(c, "docs", "skills/SKILL.md"))
        .unwrap()
        .expect("case-preserved path should be active");
    assert_eq!(now_active.hash, hash);

    // FTS reflects the new path (documents_au trigger via rebuild).
    let filepath: String = store
        .with_connection(|c| {
            c.query_row(
                "SELECT filepath FROM documents_fts WHERE rowid = ?",
                params![migrated.id],
                |r| r.get::<_, String>(0),
            )
        })
        .unwrap();
    assert!(filepath.contains("SKILL.md"), "fts filepath = {filepath}");
}

#[test]
fn migrate_returns_none_when_no_document() {
    let (_t, mut store) = open();
    let r = store
        .with_connection_mut(|c| find_or_migrate_legacy_document(c, "docs", "readme.md"))
        .unwrap();
    assert!(r.is_none());
}

#[test]
fn migrate_returns_existing_doc_when_canonical_present() {
    let (_t, mut store) = open();
    let now = now_rfc3339();
    let content = "# Content";
    let hash = hash_content(content);
    store
        .with_connection(|c| insert_content(c, &hash, content, &now))
        .unwrap();
    // Both lowercase and case-preserved paths exist (partial prior migration).
    store
        .with_connection(|c| insert_document(c, "docs", "readme.md", "Readme", &hash, &now, &now))
        .unwrap();
    store
        .with_connection(|c| insert_document(c, "docs", "README.md", "README", &hash, &now, &now))
        .unwrap();

    // Fast path: canonical doc returned directly, legacy row untouched.
    let r = store
        .with_connection_mut(|c| find_or_migrate_legacy_document(c, "docs", "README.md"))
        .unwrap()
        .expect("canonical doc should be found");
    assert_eq!(r.hash, hash);

    assert!(
        store
            .with_connection(|c| find_active_document(c, "docs", "readme.md"))
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .with_connection(|c| find_active_document(c, "docs", "README.md"))
            .unwrap()
            .is_some()
    );
}
