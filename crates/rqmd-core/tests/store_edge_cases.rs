//! Store creation + edge-case tests, ported from store.test.ts
//! `describe("Store Creation")` (WAL mode) and `describe("Edge Cases")`.

use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::lookup::{find_document, FindDocumentOptions, FindDocumentOutcome};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::search::search_fts;
use rqmd_core::Store;
use tempfile::NamedTempFile;

fn open() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

fn insert(store: &Store, collection: &str, path: &str, title: &str, body: &str) {
    let now = now_rfc3339();
    let hash = hash_content(body);
    store
        .with_connection(|c| insert_content(c, &hash, body, &now))
        .unwrap();
    store
        .with_connection(|c| insert_document(c, collection, path, title, &hash, &now, &now))
        .unwrap();
}

fn search(store: &Store, query: &str) -> Vec<rqmd_core::store::search::SearchResult> {
    store
        .with_connection(|c| search_fts(c, query, Some(20), None))
        .unwrap()
}

#[test]
fn store_uses_wal_journal_mode() {
    let (_t, store) = open();
    let mode: String = store
        .with_connection(|c| c.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0)))
        .unwrap();
    assert_eq!(mode.to_lowercase(), "wal");
}

#[test]
fn empty_database_search_returns_empty() {
    let (_t, store) = open();
    assert!(search(&store, "anything").is_empty());
    let outcome = store
        .with_connection(|c| find_document(c, "nonexistent.md", FindDocumentOptions::default()))
        .unwrap();
    assert!(matches!(outcome, FindDocumentOutcome::NotFound(_)));
}

#[test]
fn handles_very_long_document_bodies() {
    let (_t, store) = open();
    let long_body = "word ".repeat(100_000); // ~600KB
    insert(&store, "docs", "long.md", "Long", &long_body);
    assert_eq!(search(&store, "word").len(), 1);
}

#[test]
fn handles_unicode_content() {
    let (_t, store) = open();
    insert(
        &store,
        "docs",
        "unicode.md",
        "日本語タイトル",
        "# 日本語\n\n内容は日本語で書かれています。\n\nEmoji: 🎉🚀✨",
    );

    assert!(!search(&store, "日本語").is_empty());

    let outcome = store
        .with_connection(|c| {
            find_document(
                c,
                "qmd://docs/unicode.md",
                FindDocumentOptions { include_body: true },
            )
        })
        .unwrap();
    match outcome {
        FindDocumentOutcome::Found(d) => {
            assert_eq!(d.title, "日本語タイトル");
            assert!(d.body.unwrap().contains("🎉"));
        }
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}

#[test]
fn handles_special_characters_in_paths() {
    let (_t, store) = open();
    insert(&store, "docs", "file with spaces.md", "Spaced", "Content");
    let outcome = store
        .with_connection(|c| {
            find_document(c, "file with spaces.md", FindDocumentOptions::default())
        })
        .unwrap();
    assert!(matches!(outcome, FindDocumentOutcome::Found(_)));
}

#[test]
fn handles_many_documents() {
    let (_t, store) = open();
    for i in 0..10 {
        insert(
            &store,
            "docs",
            &format!("doc{i}.md"),
            "Doc",
            &format!("Content {i} searchterm"),
        );
    }
    assert_eq!(search(&store, "searchterm").len(), 10);
}
