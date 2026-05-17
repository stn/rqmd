//! End-to-end test for `search_fts` against a real in-memory store.

use rmd_core::store::documents::{hash_content, insert_content, insert_document};
use rmd_core::store::path::now_rfc3339;
use rmd_core::store::search::{search_fts, SearchSource};
use rmd_core::Store;
use tempfile::NamedTempFile;

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

#[test]
fn search_fts_returns_matching_documents() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(&store, "docs", "a.md", "Alpha", "the quick brown fox");
    insert(&store, "docs", "b.md", "Beta", "the lazy dog barked");
    insert(&store, "docs", "c.md", "Gamma", "completely unrelated text");

    let hits = store
        .with_connection(|c| search_fts(c, "fox", Some(10), None))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.display_path, "docs/a.md");
    assert_eq!(hits[0].source, SearchSource::Fts);
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_fts_filters_by_collection() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(&store, "docs", "a.md", "Alpha", "the quick brown fox");
    insert(
        &store,
        "notes",
        "x.md",
        "Note",
        "the quick brown fox lives here too",
    );

    let hits = store
        .with_connection(|c| search_fts(c, "fox", Some(10), Some("docs")))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.collection_name, "docs");
}

#[test]
fn search_fts_handles_cjk_query() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(
        &store,
        "docs",
        "ja.md",
        "日本語",
        "これは日本語のテストです",
    );
    insert(
        &store,
        "docs",
        "en.md",
        "English",
        "this is an English test",
    );

    let hits = store
        .with_connection(|c| search_fts(c, "日本語", Some(10), None))
        .unwrap();
    assert!(!hits.is_empty(), "expected at least one CJK match");
    assert!(hits.iter().any(|h| h.doc.display_path == "docs/ja.md"));
}

#[test]
fn search_fts_supports_negation() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(
        &store,
        "docs",
        "good.md",
        "G",
        "the quick brown fox is good",
    );
    insert(&store, "docs", "bad.md", "B", "the quick brown fox is bad");

    let hits = store
        .with_connection(|c| search_fts(c, "fox -bad", Some(10), None))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.display_path, "docs/good.md");
}
