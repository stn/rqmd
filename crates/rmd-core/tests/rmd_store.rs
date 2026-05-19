//! Integration tests for [`RmdStore`].
//!
//! Tests that don't need LLM model files. The lazy-LLM contract
//! ([`RmdStore::llm_initialized`]) makes it possible to exercise every
//! collection / context / retrieval / update path without ever
//! constructing a [`LlamaCpp`] instance.

use std::fs;
use std::sync::{Arc, Mutex};

use rmd_core::collections::{Collection, ConfigData};
use rmd_core::store::lookup::FindDocumentOutcome;
use rmd_core::{
    AddCollectionOptions, RmdStore, RmdStoreError, RmdStoreOptions, SearchOptions, UpdateOptions,
    UpdateProgress,
};
use tempfile::TempDir;

fn make_docs_dir() -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.md"), "# Alpha\n\nbody one").unwrap();
    fs::create_dir_all(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("sub/b.md"), "# Beta\n\nbody two").unwrap();
    dir
}

fn open_db_only(workspace: &TempDir) -> RmdStore {
    let db_path = workspace.path().join("index.sqlite");
    RmdStore::open(RmdStoreOptions {
        db_path,
        ..Default::default()
    })
    .expect("open db-only")
}

fn open_yaml(workspace: &TempDir) -> (RmdStore, std::path::PathBuf) {
    let db_path = workspace.path().join("index.sqlite");
    let yaml_path = workspace.path().join("rmd.yml");
    let store = RmdStore::open(RmdStoreOptions {
        db_path,
        config_path: Some(yaml_path.clone()),
        config: None,
    })
    .expect("open yaml");
    (store, yaml_path)
}

// ============================================================================
// Constructor
// ============================================================================

#[test]
fn open_yaml_path() {
    let ws = TempDir::new().unwrap();
    let (store, yaml_path) = open_yaml(&ws);
    assert_eq!(store.db_path(), ws.path().join("index.sqlite"));
    // YAML didn't exist before open; it should not be created by open()
    // (Config writes lazily on first mutation).
    assert!(!yaml_path.exists());
}

#[test]
fn open_inline_config() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut data = ConfigData::default();
    data.collections.insert(
        "docs".into(),
        Collection {
            path: docs.path().to_string_lossy().into_owned(),
            pattern: "**/*.md".into(),
            ..Default::default()
        },
    );
    let store = RmdStore::open(RmdStoreOptions {
        db_path: ws.path().join("index.sqlite"),
        config: Some(data),
        ..Default::default()
    })
    .expect("open inline");
    assert_eq!(store.db_path(), ws.path().join("index.sqlite"));
}

#[test]
fn open_db_only_mode() {
    let ws = TempDir::new().unwrap();
    let store = open_db_only(&ws);
    assert!(!store.llm_initialized());
}

#[test]
fn open_rejects_both_config_inputs() {
    let ws = TempDir::new().unwrap();
    let result = RmdStore::open(RmdStoreOptions {
        db_path: ws.path().join("index.sqlite"),
        config_path: Some(ws.path().join("rmd.yml")),
        config: Some(ConfigData::default()),
    });
    let err = result
        .map(|_| ())
        .expect_err("should reject both");
    assert!(matches!(err, RmdStoreError::InvalidOptions(_)));
}

// ============================================================================
// Collection mutations (DB-first, optional YAML write-through)
// ============================================================================

#[test]
fn mutator_writes_through_to_yaml_and_db() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let (mut store, yaml_path) = open_yaml(&ws);

    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                pattern: Some("**/*.md".into()),
                ignore: Some(vec!["sub/**".into()]),
            },
        )
        .expect("add_collection");

    // DB has the collection (with ignore).
    let listing = store.list_collections().expect("list");
    assert_eq!(listing.len(), 1);
    assert_eq!(listing[0].name, "docs");

    // YAML was written by Config::add_collection's internal save().
    let yaml = fs::read_to_string(&yaml_path).expect("yaml exists");
    assert!(yaml.contains("docs"));
    assert!(yaml.contains("**/*.md"));
    // TS parity: `Config::add_collection` does NOT carry `ignore` through
    // to YAML — it persists in SQLite only. The YAML should NOT contain
    // the ignore pattern.
    assert!(
        !yaml.contains("sub/**"),
        "ignore should not flow to YAML (TS parity)"
    );
}

#[test]
fn mutator_db_only_mode_writes_db_no_yaml() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);

    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .expect("add_collection");

    // DB has the collection.
    let listing = store.list_collections().expect("list");
    assert_eq!(listing.len(), 1);

    // No YAML file was created.
    assert!(!ws.path().join("rmd.yml").exists());
}

#[test]
fn remove_collection_returns_false_for_missing() {
    let ws = TempDir::new().unwrap();
    let mut store = open_db_only(&ws);
    let removed = store.remove_collection("nonexistent").expect("ok");
    assert!(!removed);
}

#[test]
fn rename_collection_round_trip() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);
    store
        .add_collection(
            "old",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(store.rename_collection("old", "new").unwrap());
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, vec!["new".to_string()]);
}

#[test]
fn set_and_get_global_context() {
    let ws = TempDir::new().unwrap();
    let mut store = open_db_only(&ws);
    assert_eq!(store.get_global_context().unwrap(), None);
    store
        .set_global_context(Some("hello".into()))
        .expect("set");
    assert_eq!(
        store.get_global_context().unwrap(),
        Some("hello".to_string())
    );
}

// ============================================================================
// Lazy LLM
// ============================================================================

#[test]
fn lazy_llm_not_constructed_for_non_llm_ops() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);

    assert!(!store.llm_initialized());

    // Each of these should not touch the LLM.
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!store.llm_initialized(), "add_collection triggered LLM");

    store.list_collections().unwrap();
    assert!(!store.llm_initialized(), "list_collections triggered LLM");

    store.get("a.md", false).ok();
    assert!(!store.llm_initialized(), "get triggered LLM");

    store.multi_get("*.md", false, None).ok();
    assert!(!store.llm_initialized(), "multi_get triggered LLM");

    // resolved_models reads config; never instantiates LLM.
    let _uris = store.resolved_models();
    assert!(!store.llm_initialized(), "resolved_models triggered LLM");
}

// ============================================================================
// search() option validation (no LLM required for the early-return path)
// ============================================================================

#[tokio::test]
async fn search_options_both_none_returns_error() {
    let ws = TempDir::new().unwrap();
    let store = open_db_only(&ws);
    let err = store
        .search(SearchOptions::default())
        .await
        .expect_err("should error");
    assert!(matches!(err, RmdStoreError::MissingSearchQuery));
    // The validation error should not trigger LLM construction.
    assert!(!store.llm_initialized());
}

// ============================================================================
// update() — exercises &mut Store + Config::clone snapshot path
// ============================================================================

#[tokio::test]
async fn update_borrow_compat() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();

    let counter = Arc::new(Mutex::new(0_usize));
    let counter_cb = counter.clone();
    let opts = UpdateOptions {
        on_progress: Some(Arc::new(move |_: UpdateProgress| {
            *counter_cb.lock().unwrap() += 1;
        })),
        ..Default::default()
    };
    let result = store.update(opts).await.expect("update");
    assert_eq!(result.collections, 1);
    assert_eq!(result.indexed, 2);
    assert!(*counter.lock().unwrap() >= 2, "progress callback fired");
}

#[tokio::test]
async fn update_filters_to_subset_of_collections() {
    let ws = TempDir::new().unwrap();
    let docs1 = make_docs_dir();
    let docs2 = make_docs_dir();
    let mut store = open_db_only(&ws);
    store
        .add_collection(
            "a",
            AddCollectionOptions {
                path: docs1.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    store
        .add_collection(
            "b",
            AddCollectionOptions {
                path: docs2.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    let result = store
        .update(UpdateOptions {
            collections: Some(vec!["a".into()]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.collections, 1);
    assert_eq!(result.indexed, 2);
}

// ============================================================================
// Retrieval after update
// ============================================================================

#[tokio::test]
async fn get_after_update_returns_document() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    store.update(UpdateOptions::default()).await.unwrap();

    let outcome = store.get("a.md", true).expect("get");
    match outcome {
        FindDocumentOutcome::Found(doc) => {
            assert_eq!(doc.title, "Alpha");
            assert!(doc.body.unwrap().contains("body one"));
        }
        FindDocumentOutcome::NotFound(_) => panic!("a.md should be found"),
    }
}

#[tokio::test]
async fn search_lex_after_update() {
    let ws = TempDir::new().unwrap();
    let docs = make_docs_dir();
    let mut store = open_db_only(&ws);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.path().to_string_lossy().into_owned(),
                ..Default::default()
            },
        )
        .unwrap();
    store.update(UpdateOptions::default()).await.unwrap();

    let hits = store.search_lex("Alpha", Some(10), None).expect("search");
    assert!(
        hits.iter().any(|h| h.doc.title == "Alpha"),
        "expected Alpha in {:?}",
        hits.iter().map(|h| &h.doc.title).collect::<Vec<_>>()
    );
    // search_lex must not construct the LLM.
    assert!(!store.llm_initialized());
}

// ============================================================================
// Lifecycle
// ============================================================================

#[tokio::test]
async fn shutdown_idempotent_when_llm_never_constructed() {
    let ws = TempDir::new().unwrap();
    let store = open_db_only(&ws);
    // Two no-op shutdowns (LLM never constructed) — must not panic.
    store.shutdown().await;
    store.shutdown().await;
    store.close().await;
}
