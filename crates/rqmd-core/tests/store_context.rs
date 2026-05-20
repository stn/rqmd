//! Integration tests for path/collection/global context resolution, ported
//! from store.test.ts `describe("Path Context")` and the hierarchical-context
//! case of `describe("findDocument")`.

use rqmd_core::collections::Collection;
use rqmd_core::store::context::{get_context_for_file, insert_context};
use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::lookup::{find_document, FindDocumentOptions, FindDocumentOutcome};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::store_config::{set_store_global_context, upsert_store_collection};
use rqmd_core::Store;
use tempfile::NamedTempFile;

fn open() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    (tmp, store)
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

fn add_doc(store: &Store, collection: &str, path: &str) {
    let now = now_rfc3339();
    let hash = hash_content(&format!("body for {collection}/{path}"));
    store
        .with_connection(|c| {
            insert_content(c, &hash, &format!("body for {collection}/{path}"), &now)
        })
        .unwrap();
    store
        .with_connection(|c| insert_document(c, collection, path, "title", &hash, &now, &now))
        .unwrap();
}

fn set_context(store: &Store, collection: &str, prefix: &str, ctx: &str) {
    store
        .with_connection(|c| insert_context(c, collection, prefix, ctx))
        .unwrap();
}

fn context_for(store: &Store, filepath: &str) -> Option<String> {
    store
        .with_connection(|c| get_context_for_file(c, filepath))
        .unwrap()
}

#[test]
fn get_context_for_file_returns_none_when_unset() {
    let (_t, store) = open();
    assert!(context_for(&store, "/some/random/path.md").is_none());
}

#[test]
fn get_context_for_file_returns_matching_context() {
    let (_t, store) = open();
    add_collection(&store, "collection", "/test/collection");
    set_context(&store, "collection", "/docs", "Documentation files");
    add_doc(&store, "collection", "docs/readme.md");

    assert_eq!(
        context_for(&store, "/test/collection/docs/readme.md").as_deref(),
        Some("Documentation files")
    );
}

#[test]
fn get_context_for_file_returns_all_matching_contexts() {
    let (_t, store) = open();
    add_collection(&store, "collection", "/test/collection");
    set_context(&store, "collection", "/", "General test files");
    set_context(&store, "collection", "/docs", "Documentation files");
    set_context(&store, "collection", "/docs/api", "API documentation");

    add_doc(&store, "collection", "readme.md");
    add_doc(&store, "collection", "docs/guide.md");
    add_doc(&store, "collection", "docs/api/reference.md");

    assert_eq!(
        context_for(&store, "/test/collection/readme.md").as_deref(),
        Some("General test files")
    );
    assert_eq!(
        context_for(&store, "/test/collection/docs/guide.md").as_deref(),
        Some("General test files\n\nDocumentation files")
    );
    assert_eq!(
        context_for(&store, "/test/collection/docs/api/reference.md").as_deref(),
        Some("General test files\n\nDocumentation files\n\nAPI documentation")
    );
}

#[test]
fn find_document_includes_hierarchical_contexts() {
    let (_t, store) = open();
    add_collection(&store, "archive", "/archive");
    store
        .with_connection(|c| set_store_global_context(c, Some("Global context for all documents")))
        .unwrap();
    set_context(&store, "archive", "/", "Archive collection context");
    set_context(&store, "archive", "/podcasts", "Podcast episodes");
    set_context(
        &store,
        "archive",
        "/podcasts/external",
        "External podcast interviews",
    );

    add_doc(&store, "archive", "podcasts/external/2024-jan-interview.md");

    let outcome = store
        .with_connection(|c| {
            find_document(
                c,
                "qmd://archive/podcasts/external/2024-jan-interview.md",
                FindDocumentOptions::default(),
            )
        })
        .unwrap();
    match outcome {
        FindDocumentOutcome::Found(d) => assert_eq!(
            d.context.as_deref(),
            Some(
                "Global context for all documents\n\n\
                 Archive collection context\n\n\
                 Podcast episodes\n\n\
                 External podcast interviews"
            )
        ),
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}
