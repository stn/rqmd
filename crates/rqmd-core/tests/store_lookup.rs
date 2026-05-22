//! Integration tests for document lookup, ported from store.test.ts
//! `describe("Document Retrieval")` and `describe("Fuzzy Matching")`:
//! find_document, get_document_body, find_documents (multi-get),
//! find_similar_files, match_files_by_glob.

use rqmd_core::collections::Collection;
use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::lookup::{
    find_document, find_documents, find_similar_files, get_document_body, match_files_by_glob,
    FindDocumentOptions, FindDocumentOutcome, FindDocumentsOptions,
};
use rqmd_core::store::path::{homedir, now_rfc3339};
use rqmd_core::store::search::MultiGetResult;
use rqmd_core::store::store_config::upsert_store_collection;
use rqmd_core::Store;
use tempfile::NamedTempFile;

fn open() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
}

fn insert(store: &Store, collection: &str, path: &str, title: &str, body: &str) -> String {
    let now = now_rfc3339();
    let hash = hash_content(body);
    store
        .with_connection(|c| insert_content(c, &hash, body, &now))
        .unwrap();
    store
        .with_connection(|c| insert_document(c, collection, path, title, &hash, &now, &now))
        .unwrap();
    hash
}

fn add_collection(store: &Store, name: &str, path: &str) {
    let coll = Collection {
        path: path.to_string(),
        pattern: "**/*.md".to_string(),
        ignore: None,
        context: None,
        update: None,
        include_by_default: None,
    };
    store
        .with_connection(|c| upsert_store_collection(c, name, &coll))
        .unwrap();
}

fn find(store: &Store, name: &str, include_body: bool) -> FindDocumentOutcome {
    store
        .with_connection(|c| find_document(c, name, FindDocumentOptions { include_body }))
        .unwrap()
}

// ============================================================================
// find_document
// ============================================================================

#[test]
fn find_document_by_exact_virtual_path() {
    let (_t, store) = open();
    insert(
        &store,
        "docs",
        "mydoc.md",
        "My Document",
        "Document content here",
    );

    let r = find(&store, "qmd://docs/mydoc.md", false);
    match r {
        FindDocumentOutcome::Found(d) => {
            assert_eq!(d.title, "My Document");
            assert_eq!(d.display_path, "docs/mydoc.md");
            assert_eq!(d.filepath, "qmd://docs/mydoc.md");
            assert!(d.body.is_none()); // not included by default
        }
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}

#[test]
fn find_document_by_partial_path() {
    let (_t, store) = open();
    insert(&store, "docs", "sub/mydoc.md", "My Document", "body");
    assert!(matches!(
        find(&store, "mydoc.md", false),
        FindDocumentOutcome::Found(_)
    ));
}

#[test]
fn find_document_includes_body_when_requested() {
    let (_t, store) = open();
    insert(
        &store,
        "docs",
        "mydoc.md",
        "My Document",
        "The actual body content",
    );
    match find(&store, "qmd://docs/mydoc.md", true) {
        FindDocumentOutcome::Found(d) => {
            assert_eq!(d.body.as_deref(), Some("The actual body content"))
        }
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}

#[test]
fn find_document_not_found_returns_suggestions() {
    let (_t, store) = open();
    insert(&store, "docs", "similar.md", "Similar", "body");
    match find(&store, "simlar.md", false) {
        FindDocumentOutcome::NotFound(nf) => {
            assert!(nf.similar_files.contains(&"similar.md".to_string()));
        }
        FindDocumentOutcome::Found(_) => panic!("expected NotFound"),
    }
}

#[test]
fn find_document_handles_line_suffix() {
    let (_t, store) = open();
    insert(&store, "docs", "mydoc.md", "My Document", "body");
    assert!(matches!(
        find(&store, "mydoc.md:100", false),
        FindDocumentOutcome::Found(_)
    ));
}

#[test]
fn find_document_by_docid() {
    let (_t, store) = open();
    let hash = insert(
        &store,
        "docs",
        "mydoc.md",
        "My Document",
        "unique body for docid",
    );
    let docid: String = hash.chars().take(6).collect();
    match find(&store, &docid, false) {
        FindDocumentOutcome::Found(d) => assert_eq!(d.display_path, "docs/mydoc.md"),
        FindDocumentOutcome::NotFound(_) => panic!("expected Found via docid"),
    }
}

#[test]
fn find_document_expands_home_relative_path() {
    let (_t, store) = open();
    let home = homedir().display().to_string();
    add_collection(&store, "home", &home);
    insert(&store, "home", "docs/mydoc.md", "Home Doc", "body");
    assert!(matches!(
        find(&store, "~/docs/mydoc.md", false),
        FindDocumentOutcome::Found(_)
    ));
}

#[test]
fn find_document_by_absolute_path_normalizes_separators() {
    let (_t, store) = open();
    // Collection root stored with backslashes (as on Windows). The absolute-path
    // branch must normalize separators to match the handelized `documents.path`;
    // without normalization the backslash root never matches (fails even on Unix).
    add_collection(&store, "docs", r"C:\data\docs");
    insert(&store, "docs", "api.md", "API", "body");
    assert!(
        matches!(
            find(&store, r"C:\data\docs\api.md", false),
            FindDocumentOutcome::Found(_)
        ),
        "backslash absolute path should resolve"
    );
    assert!(
        matches!(
            find(&store, "C:/data/docs/api.md", false),
            FindDocumentOutcome::Found(_)
        ),
        "forward-slash absolute path should resolve"
    );
}

// ============================================================================
// get_document_body
// ============================================================================

fn body(store: &Store, filepath: &str, from: Option<usize>, max: Option<usize>) -> Option<String> {
    store
        .with_connection(|c| get_document_body(c, filepath, from, max))
        .unwrap()
}

#[test]
fn get_document_body_returns_full_body() {
    let (_t, store) = open();
    insert(
        &store,
        "docs",
        "mydoc.md",
        "Doc",
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
    );
    assert_eq!(
        body(&store, "qmd://docs/mydoc.md", None, None).as_deref(),
        Some("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
    );
}

#[test]
fn get_document_body_supports_line_range() {
    let (_t, store) = open();
    insert(
        &store,
        "docs",
        "mydoc.md",
        "Doc",
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
    );
    assert_eq!(
        body(&store, "qmd://docs/mydoc.md", Some(2), Some(2)).as_deref(),
        Some("Line 2\nLine 3")
    );
}

#[test]
fn get_document_body_returns_none_for_missing() {
    let (_t, store) = open();
    assert!(body(&store, "qmd://docs/nonexistent.md", None, None).is_none());
}

#[test]
fn get_document_body_out_of_range_start_is_empty() {
    let (_t, store) = open();
    insert(&store, "docs", "mydoc.md", "Doc", "Line 1\nLine 2\nLine 3");
    assert_eq!(
        body(&store, "qmd://docs/mydoc.md", Some(100), Some(5)).as_deref(),
        Some("")
    );
}

// ============================================================================
// find_documents (multi-get)
// ============================================================================

fn find_many(
    store: &Store,
    pattern: &str,
    opts: FindDocumentsOptions,
) -> rqmd_core::store::lookup::FindDocumentsResult {
    store
        .with_connection(|c| find_documents(c, pattern, opts))
        .unwrap()
}

#[test]
fn find_documents_by_glob_pattern() {
    let (_t, store) = open();
    insert(&store, "docs", "journals/2024-01.md", "J1", "a");
    insert(&store, "docs", "journals/2024-02.md", "J2", "b");
    insert(&store, "docs", "other/file.md", "O", "c");

    let r = find_many(
        &store,
        "journals/2024-*.md",
        FindDocumentsOptions::default(),
    );
    assert!(r.errors.is_empty());
    assert_eq!(r.docs.len(), 2);
}

#[test]
fn find_documents_by_comma_separated_list() {
    let (_t, store) = open();
    insert(&store, "docs", "doc1.md", "D1", "a");
    insert(&store, "docs", "doc2.md", "D2", "b");

    let r = find_many(&store, "doc1.md, doc2.md", FindDocumentsOptions::default());
    assert!(r.errors.is_empty());
    assert_eq!(r.docs.len(), 2);
}

#[test]
fn find_documents_reports_errors_for_missing() {
    let (_t, store) = open();
    insert(&store, "docs", "doc1.md", "D1", "a");

    let r = find_many(
        &store,
        "doc1.md, nonexistent.md",
        FindDocumentsOptions::default(),
    );
    assert_eq!(r.docs.len(), 1);
    assert_eq!(r.errors.len(), 1);
    assert!(r.errors[0].contains("not found"));
}

#[test]
fn find_documents_skips_large_files() {
    let (_t, store) = open();
    insert(&store, "docs", "large.md", "L", &"x".repeat(20000)); // 20KB

    let opts = FindDocumentsOptions {
        include_body: false,
        max_bytes: 10000,
    };
    let r = find_many(&store, "large.md", opts);
    assert_eq!(r.docs.len(), 1);
    match &r.docs[0] {
        MultiGetResult::Skipped { skip_reason, .. } => assert!(skip_reason.contains("too large")),
        MultiGetResult::Found(_) => panic!("expected Skipped"),
    }
}

#[test]
fn find_documents_includes_body_when_requested() {
    let (_t, store) = open();
    insert(&store, "docs", "doc1.md", "D1", "The content");

    let opts = FindDocumentsOptions {
        include_body: true,
        ..Default::default()
    };
    let r = find_many(&store, "doc1.md", opts);
    match &r.docs[0] {
        MultiGetResult::Found(d) => assert_eq!(d.body.as_deref(), Some("The content")),
        MultiGetResult::Skipped { .. } => panic!("expected Found"),
    }
}

#[test]
fn find_documents_supports_brace_expansion() {
    let (_t, store) = open();
    insert(&store, "docs", "doc1.md", "D1", "a");
    insert(&store, "docs", "doc2.md", "D2", "b");
    insert(&store, "docs", "doc3.md", "D3", "c");

    let r = find_many(&store, "{doc1,doc2}.md", FindDocumentsOptions::default());
    assert!(r.errors.is_empty());
    assert_eq!(r.docs.len(), 2);
}

#[test]
fn find_documents_supports_brace_expansion_with_collection_prefix() {
    let (_t, store) = open();
    insert(&store, "docs", "readme.md", "R", "a");
    insert(&store, "docs", "changelog.md", "C", "b");

    let r = find_many(
        &store,
        "docs/{readme,changelog}.md",
        FindDocumentsOptions::default(),
    );
    assert!(r.errors.is_empty());
    assert_eq!(r.docs.len(), 2);
}

// ============================================================================
// find_similar_files / match_files_by_glob
// ============================================================================

#[test]
fn find_similar_files_finds_similar_paths() {
    let (_t, store) = open();
    insert(&store, "docs", "docs/readme.md", "R", "a");
    insert(&store, "docs", "docs/readmi.md", "R", "b"); // typo

    let similar = store
        .with_connection(|c| find_similar_files(c, "docs/readme.md", Some(3), Some(5)))
        .unwrap();
    assert!(similar.contains(&"docs/readme.md".to_string()));
}

#[test]
fn find_similar_files_respects_max_distance() {
    let (_t, store) = open();
    insert(&store, "docs", "abc.md", "A", "a");
    insert(&store, "docs", "xyz.md", "X", "b"); // very different

    let similar = store
        .with_connection(|c| find_similar_files(c, "abc.md", Some(1), Some(5)))
        .unwrap();
    assert!(similar.contains(&"abc.md".to_string()));
    assert!(!similar.contains(&"xyz.md".to_string()));
}

#[test]
fn match_files_by_glob_matches_patterns() {
    let (_t, store) = open();
    insert(&store, "docs", "journals/2024-01.md", "J1", "a");
    insert(&store, "docs", "journals/2024-02.md", "J2", "b");
    insert(&store, "docs", "docs/readme.md", "R", "c");

    let matches = store
        .with_connection(|c| match_files_by_glob(c, "journals/*.md"))
        .unwrap();
    assert_eq!(matches.len(), 2);
    assert!(matches
        .iter()
        .all(|m| m.display_path.starts_with("journals/")));
}

#[test]
fn match_files_by_glob_matches_collection_path_patterns() {
    let (_t, store) = open();
    insert(&store, "mycoll", "readme.md", "R", "a");
    insert(&store, "mycoll", "changelog.md", "C", "b");

    let matches = store
        .with_connection(|c| match_files_by_glob(c, "mycoll/*.md"))
        .unwrap();
    assert_eq!(matches.len(), 2);
}

#[test]
fn match_files_by_glob_matches_brace_expansion() {
    let (_t, store) = open();
    insert(&store, "mycoll", "readme.md", "R", "a");
    insert(&store, "mycoll", "changelog.md", "C", "b");
    insert(&store, "mycoll", "license.md", "L", "c");

    let matches = store
        .with_connection(|c| match_files_by_glob(c, "mycoll/{readme,changelog}.md"))
        .unwrap();
    assert_eq!(matches.len(), 2);
}
