//! Integration tests for [`RqmdStore`].
//!
//! Tests that don't need LLM model files. The lazy-LLM contract
//! ([`RqmdStore::llm_initialized`]) makes it possible to exercise every
//! collection / context / retrieval / update path without ever
//! constructing a [`LlamaCpp`] instance.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use rqmd_core::collections::{Collection, ConfigData, ContextMap};
use rqmd_core::store::lookup::FindDocumentOutcome;
use rqmd_core::store_ops::{ExpandedQuery, ExpandedQueryType};
use rqmd_core::{
    AddCollectionOptions, Config, RqmdStore, RqmdStoreError, RqmdStoreOptions, SearchOptions,
    StoreOpsEmbedOptions, UpdateOptions, UpdateProgress,
};
use tempfile::TempDir;

fn make_docs_dir() -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.md"), "# Alpha\n\nbody one").unwrap();
    fs::create_dir_all(dir.path().join("sub")).unwrap();
    fs::write(dir.path().join("sub/b.md"), "# Beta\n\nbody two").unwrap();
    dir
}

fn open_db_only(workspace: &TempDir) -> RqmdStore {
    let db_path = workspace.path().join("index.sqlite");
    RqmdStore::open(RqmdStoreOptions {
        db_path,
        ..Default::default()
    })
    .expect("open db-only")
}

fn open_yaml(workspace: &TempDir) -> (RqmdStore, std::path::PathBuf) {
    let db_path = workspace.path().join("index.sqlite");
    let yaml_path = workspace.path().join("rqmd.yml");
    let store = RqmdStore::open(RqmdStoreOptions {
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
    let store = RqmdStore::open(RqmdStoreOptions {
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
    let result = RqmdStore::open(RqmdStoreOptions {
        db_path: ws.path().join("index.sqlite"),
        config_path: Some(ws.path().join("rqmd.yml")),
        config: Some(ConfigData::default()),
    });
    let err = result
        .map(|_| ())
        .expect_err("should reject both");
    assert!(matches!(err, RqmdStoreError::InvalidOptions(_)));
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
    assert!(!ws.path().join("rqmd.yml").exists());
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
    assert!(matches!(err, RqmdStoreError::MissingSearchQuery));
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

// ============================================================================
// sdk.test.ts parity
//
// Direct port of `tobi/qmd/test/sdk.test.ts` (the public-SDK suite) onto
// `RqmdStore`. Fixture and assertions mirror the TS file block-by-block.
// Cases that are genuinely N/A in Rust are intentionally omitted with a note:
//   * "throws if dbPath is missing" — `db_path: PathBuf` has no empty-string contract.
//   * "close() makes subsequent ops throw" — `close(self)` consumes the value, so
//     post-close use is a compile error, not a runtime case.
//   * `type exports` — compile-time in Rust; the crate compiling enforces the surface.
// Cases that require a real GGUF model (the `embed` happy-path and LLM query
// expansion, which sdk itself skips under CI) are `#[ignore]` at the bottom.
// ============================================================================

/// Reproduces sdk.test.ts's fixture: a `docs/` (3 files) and `notes/` (3 files)
/// tree with identical contents (the BM25 search asserts depend on the text).
fn make_sdk_fixture() -> TempDir {
    let dir = TempDir::new().unwrap();
    let docs = dir.path().join("docs");
    let notes = dir.path().join("notes");
    fs::create_dir_all(&docs).unwrap();
    fs::create_dir_all(&notes).unwrap();
    fs::write(
        docs.join("readme.md"),
        "# Getting Started\n\nThis is the getting started guide for the project.\n",
    )
    .unwrap();
    fs::write(
        docs.join("auth.md"),
        "# Authentication\n\nAuthentication uses JWT tokens for session management.\n\
         Users log in with email and password.\n",
    )
    .unwrap();
    fs::write(
        docs.join("api.md"),
        "# API Reference\n\n## Endpoints\n\n### POST /login\nAuthenticate a user.\n\n\
         ### GET /users\nList all users.\n",
    )
    .unwrap();
    fs::write(
        notes.join("meeting-2025-01.md"),
        "# January Planning Meeting\n\nDiscussed Q1 roadmap and resource allocation.\n",
    )
    .unwrap();
    fs::write(
        notes.join("meeting-2025-02.md"),
        "# February Standup\n\nReviewed sprint progress. Authentication feature is on track.\n",
    )
    .unwrap();
    fs::write(
        notes.join("ideas.md"),
        "# Project Ideas\n\n- Build a search engine\n- Create a knowledge base\n\
         - Implement vector search\n",
    )
    .unwrap();
    dir
}

fn docs_dir(fx: &TempDir) -> String {
    fx.path().join("docs").to_string_lossy().into_owned()
}

fn notes_dir(fx: &TempDir) -> String {
    fx.path().join("notes").to_string_lossy().into_owned()
}

fn db(ws: &TempDir) -> PathBuf {
    ws.path().join("index.sqlite")
}

fn coll(path: &str) -> Collection {
    Collection {
        path: path.to_string(),
        pattern: "**/*.md".to_string(),
        ..Default::default()
    }
}

/// Open an inline-config store with the given `(name, path)` collections.
fn open_inline_with(db_path: PathBuf, collections: Vec<(&str, String)>) -> RqmdStore {
    let mut data = ConfigData::default();
    for (name, path) in collections {
        data.collections.insert(name.to_string(), coll(&path));
    }
    RqmdStore::open(RqmdStoreOptions {
        db_path,
        config: Some(data),
        ..Default::default()
    })
    .expect("open inline")
}

/// Open a db-only store at the same path a prior session used.
fn open_db_only_at(db_path: PathBuf) -> RqmdStore {
    RqmdStore::open(RqmdStoreOptions {
        db_path,
        ..Default::default()
    })
    .expect("open db-only")
}

async fn open_indexed(ws: &TempDir, collections: Vec<(&str, String)>) -> RqmdStore {
    let mut store = open_inline_with(db(ws), collections);
    store.update(UpdateOptions::default()).await.expect("update");
    store
}

// ---------------------------------------------------------------------------
// createStore
// ---------------------------------------------------------------------------

#[test]
fn create_store_with_inline_config() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    assert_eq!(store.db_path(), db(&ws));
    // Step-0 sync makes the inline collection visible immediately.
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"docs".to_string()));
}

#[test]
fn create_store_with_yaml_config_file() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let yaml = ws.path().join("config.yml");
    {
        let mut cfg = Config::from_file(&yaml).unwrap();
        cfg.add_collection("docs", docs_dir(&fx), Some("**/*.md"))
            .unwrap();
    }
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config_path: Some(yaml),
        config: None,
    })
    .unwrap();
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"docs".to_string()));
}

#[test]
fn opens_with_just_db_path_lists_empty() {
    let ws = TempDir::new().unwrap();
    let store = open_db_only_at(db(&ws));
    assert!(store.list_collections().unwrap().is_empty());
}

#[test]
fn creates_database_file_on_disk() {
    let ws = TempDir::new().unwrap();
    let dbp = db(&ws);
    let _store = RqmdStore::open(RqmdStoreOptions {
        db_path: dbp.clone(),
        config: Some(ConfigData::default()),
        ..Default::default()
    })
    .unwrap();
    assert!(dbp.exists());
}

// ---------------------------------------------------------------------------
// collection management
// ---------------------------------------------------------------------------

#[test]
fn add_collection_adds_collection() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![]);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"docs".to_string()));
}

#[test]
fn add_collection_with_default_pattern() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![]);
    store
        .add_collection(
            "notes",
            AddCollectionOptions {
                path: notes_dir(&fx),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(store
        .list_collections()
        .unwrap()
        .iter()
        .any(|c| c.name == "notes"));
}

#[test]
fn remove_collection_removes_existing() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    let removed = store.remove_collection("docs").unwrap();
    assert!(removed);
    assert!(!store
        .list_collections()
        .unwrap()
        .iter()
        .any(|c| c.name == "docs"));
}

#[test]
fn rename_collection_returns_false_for_missing_source_store() {
    let ws = TempDir::new().unwrap();
    let mut store = open_db_only_at(db(&ws));
    assert!(!store.rename_collection("nonexistent", "new-name").unwrap());
}

#[test]
fn rename_collection_errors_if_target_exists_store() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(
        db(&ws),
        vec![("a", docs_dir(&fx)), ("b", notes_dir(&fx))],
    );
    // Renaming onto an existing name yields a typed DuplicateCollection error
    // (message contains "already exists", matching qmd).
    let err = store.rename_collection("a", "b").unwrap_err();
    assert!(
        matches!(
            err,
            RqmdStoreError::Collections(rqmd_core::collections::Error::DuplicateCollection(ref n))
                if n == "b"
        ),
        "expected DuplicateCollection(\"b\"), got: {err:?}"
    );
    assert!(format!("{err}").contains("already exists"));
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
}

#[test]
fn list_collections_empty_for_empty_config() {
    let ws = TempDir::new().unwrap();
    let store = open_inline_with(db(&ws), vec![]);
    assert!(store.list_collections().unwrap().is_empty());
}

#[test]
fn multiple_collections_can_be_added() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![]);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    store
        .add_collection(
            "notes",
            AddCollectionOptions {
                path: notes_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"docs".to_string()));
    assert!(names.contains(&"notes".to_string()));
}

// ---------------------------------------------------------------------------
// context management
// ---------------------------------------------------------------------------

fn has_ctx(store: &RqmdStore, collection: &str, path: &str, ctx: &str) -> bool {
    store
        .list_contexts()
        .unwrap()
        .iter()
        .any(|e| e.collection == collection && e.path == path && e.context == ctx)
}

fn open_docs_notes_inline(ws: &TempDir, fx: &TempDir) -> RqmdStore {
    open_inline_with(
        db(ws),
        vec![("docs", docs_dir(fx)), ("notes", notes_dir(fx))],
    )
}

#[test]
fn add_context_adds_to_collection_path() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    let added = store.add_context("docs", "/auth", "Authentication docs").unwrap();
    assert!(added);
    assert!(has_ctx(&store, "docs", "/auth", "Authentication docs"));
}

#[test]
fn add_context_returns_false_for_missing_collection() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    assert!(!store
        .add_context("nonexistent", "/path", "Some context")
        .unwrap());
}

#[test]
fn remove_context_removes_existing() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.add_context("docs", "/auth", "Authentication docs").unwrap();
    assert!(store.remove_context("docs", "/auth").unwrap());
    assert!(!store
        .list_contexts()
        .unwrap()
        .iter()
        .any(|e| e.path == "/auth"));
}

#[test]
fn remove_context_returns_false_for_missing() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    assert!(!store.remove_context("docs", "/nonexistent").unwrap());
}

#[test]
fn set_global_context_with_none_clears() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.set_global_context(Some("Some context".into())).unwrap();
    store.set_global_context(None).unwrap();
    assert_eq!(store.get_global_context().unwrap(), None);
}

#[test]
fn list_contexts_includes_global() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.set_global_context(Some("Global context".into())).unwrap();
    assert!(has_ctx(&store, "*", "/", "Global context"));
}

#[test]
fn list_contexts_across_multiple_collections() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.add_context("docs", "/", "Documentation").unwrap();
    store.add_context("notes", "/", "Personal notes").unwrap();
    let n_root = store
        .list_contexts()
        .unwrap()
        .iter()
        .filter(|e| e.path == "/" && e.collection != "*")
        .count();
    assert_eq!(n_root, 2);
}

#[test]
fn multiple_contexts_on_same_collection() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.add_context("docs", "/auth", "Auth docs").unwrap();
    store.add_context("docs", "/api", "API docs").unwrap();
    let mut paths: Vec<_> = store
        .list_contexts()
        .unwrap()
        .into_iter()
        .filter(|e| e.collection == "docs")
        .map(|e| e.path)
        .collect();
    paths.sort();
    assert_eq!(paths, vec!["/api".to_string(), "/auth".to_string()]);
}

#[test]
fn add_context_overwrites_existing() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_docs_notes_inline(&ws, &fx);
    store.add_context("docs", "/auth", "Old context").unwrap();
    store.add_context("docs", "/auth", "New context").unwrap();
    let entries: Vec<_> = store
        .list_contexts()
        .unwrap()
        .into_iter()
        .filter(|e| e.path == "/auth")
        .collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].context, "New context");
}

// ---------------------------------------------------------------------------
// inline config isolation
// ---------------------------------------------------------------------------

#[test]
fn inline_config_does_not_write_files() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    store
        .add_collection(
            "notes",
            AddCollectionOptions {
                path: notes_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    store.add_context("docs", "/", "Documentation").unwrap();
    // Inline mode performs no YAML write-through.
    assert!(!ws.path().join("index.yml").exists());
    assert!(!ws.path().join("rqmd.yml").exists());
}

#[test]
fn inline_config_mutations_persist_within_session() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![]);
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    store.add_context("docs", "/", "My docs").unwrap();
    assert!(store
        .list_collections()
        .unwrap()
        .iter()
        .any(|c| c.name == "docs"));
    assert!(has_ctx(&store, "docs", "/", "My docs"));
}

#[test]
fn two_stores_with_different_inline_configs_independent() {
    let ws1 = TempDir::new().unwrap();
    let ws2 = TempDir::new().unwrap();
    let fx = make_sdk_fixture();

    let _store1 = open_inline_with(db(&ws1), vec![("docs", docs_dir(&fx))]);
    let store2 = open_inline_with(db(&ws2), vec![("notes", notes_dir(&fx))]);

    let names: Vec<_> = store2
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"notes".to_string()));
    assert!(!names.contains(&"docs".to_string()));
}

// ---------------------------------------------------------------------------
// YAML config file mode
// ---------------------------------------------------------------------------

#[test]
fn loads_collections_from_yaml() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let yaml = ws.path().join("config.yml");
    {
        let mut cfg = Config::from_file(&yaml).unwrap();
        cfg.add_collection("docs", docs_dir(&fx), Some("**/*.md"))
            .unwrap();
        cfg.add_collection("notes", notes_dir(&fx), Some("**/*.md"))
            .unwrap();
    }
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config_path: Some(yaml),
        config: None,
    })
    .unwrap();
    let names: Vec<_> = store
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert!(names.contains(&"docs".to_string()));
    assert!(names.contains(&"notes".to_string()));
}

#[test]
fn add_collection_persists_to_yaml() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let yaml = ws.path().join("config-persist.yml");
    fs::write(&yaml, "collections: {}\n").unwrap();

    let mut store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config_path: Some(yaml.clone()),
        config: None,
    })
    .unwrap();
    store
        .add_collection(
            "newcol",
            AddCollectionOptions {
                path: docs_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    drop(store);

    let reloaded = Config::from_file(&yaml).unwrap();
    let nc = reloaded.get_collection("newcol").expect("newcol persisted");
    assert_eq!(nc.collection.path, docs_dir(&fx));
}

#[test]
fn context_persists_to_yaml() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let yaml = ws.path().join("config-ctx.yml");
    {
        let mut cfg = Config::from_file(&yaml).unwrap();
        cfg.add_collection("docs", docs_dir(&fx), Some("**/*.md"))
            .unwrap();
    }
    let mut store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config_path: Some(yaml.clone()),
        config: None,
    })
    .unwrap();
    assert!(store.add_context("docs", "/api", "API documentation").unwrap());
    drop(store);

    let reloaded = Config::from_file(&yaml).unwrap();
    assert_eq!(
        reloaded.contexts("docs").unwrap().get("/api"),
        Some(&"API documentation".to_string())
    );
}

#[test]
fn non_existent_config_file_returns_empty() {
    let ws = TempDir::new().unwrap();
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config_path: Some(ws.path().join("nonexistent-config.yml")),
        config: None,
    })
    .unwrap();
    assert!(store.list_collections().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// searchLex (BM25)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_lex_returns_results() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("authentication", None, None).unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn search_lex_result_shape() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("authentication", None, None).unwrap();
    assert!(!results.is_empty());
    let r = &results[0];
    assert!(!r.doc.filepath.is_empty());
    assert!(!r.doc.title.is_empty());
    assert!(!r.doc.docid.is_empty());
    assert!(!r.doc.collection_name.is_empty());
    assert!(r.score > 0.0);
}

#[tokio::test]
async fn search_lex_respects_limit() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("meeting", Some(1), None).unwrap();
    assert!(results.len() <= 1);
}

#[tokio::test]
async fn search_lex_with_collection_filter() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("authentication", None, Some("notes")).unwrap();
    for r in &results {
        assert_eq!(r.doc.collection_name, "notes");
    }
}

#[tokio::test]
async fn search_lex_empty_for_non_matching() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("xyznonexistentterm123", None, None).unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn search_lex_finds_documents_across_collections() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store.search_lex("authentication", Some(10), None).unwrap();
    let collections: std::collections::HashSet<_> =
        results.iter().map(|r| r.doc.collection_name.clone()).collect();
    assert!(!collections.is_empty());
}

// ---------------------------------------------------------------------------
// get and multiGet
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_retrieves_a_document_by_path() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    match store.get("qmd://docs/auth.md", false).unwrap() {
        FindDocumentOutcome::Found(d) => {
            assert_eq!(d.title, "Authentication");
            assert_eq!(d.collection_name, "docs");
        }
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}

#[tokio::test]
async fn get_with_include_body_returns_body() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    match store.get("qmd://docs/auth.md", true).unwrap() {
        FindDocumentOutcome::Found(d) => {
            assert!(d.body.unwrap().contains("JWT tokens"));
        }
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    }
}

#[tokio::test]
async fn get_returns_not_found_for_missing_document() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    assert!(matches!(
        store.get("qmd://docs/nonexistent.md", false).unwrap(),
        FindDocumentOutcome::NotFound(_)
    ));
}

#[tokio::test]
async fn get_by_docid() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    let docid = match store.get("qmd://docs/readme.md", false).unwrap() {
        FindDocumentOutcome::Found(d) => d.docid,
        FindDocumentOutcome::NotFound(_) => panic!("expected Found"),
    };
    match store.get(&docid, false).unwrap() {
        FindDocumentOutcome::Found(d) => assert_eq!(d.docid, docid),
        FindDocumentOutcome::NotFound(_) => panic!("expected Found via docid"),
    }
}

#[tokio::test]
async fn multi_get_retrieves_multiple_documents() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    let bundle = store.multi_get("docs/*.md", false, None).unwrap();
    assert!(!bundle.docs.is_empty());
}

// ---------------------------------------------------------------------------
// index health
// ---------------------------------------------------------------------------

#[test]
fn get_status_returns_valid_structure() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    let status = store.status().unwrap();
    // Fresh (no update) — zero documents, nothing pending embedding.
    assert_eq!(status.total_documents, 0);
    assert_eq!(status.needs_embedding, 0);
}

#[test]
fn get_index_health_returns_valid_structure() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    let health = store.index_health().unwrap();
    // Fresh store: nothing indexed, nothing pending embedding.
    assert_eq!(health.needs_embedding, 0);
    assert_eq!(health.total_docs, 0);
}

#[test]
fn fresh_store_has_zero_documents() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    assert_eq!(store.status().unwrap().total_documents, 0);
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_indexes_files_and_returns_correct_stats() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    let result = store.update(UpdateOptions::default()).await.unwrap();
    assert_eq!(result.collections, 1);
    assert_eq!(result.indexed, 3);
    assert_eq!(result.updated, 0);
    assert_eq!(result.unchanged, 0);
    assert_eq!(result.removed, 0);
}

#[tokio::test]
async fn second_update_shows_unchanged_files() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);
    store.update(UpdateOptions::default()).await.unwrap();
    let result = store.update(UpdateOptions::default()).await.unwrap();
    assert_eq!(result.indexed, 0);
    assert_eq!(result.unchanged, 3);
}

#[tokio::test]
async fn update_multiple_collections() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store =
        open_inline_with(db(&ws), vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]);
    let result = store.update(UpdateOptions::default()).await.unwrap();
    assert_eq!(result.collections, 2);
    assert_eq!(result.indexed, 6);
}

// ---------------------------------------------------------------------------
// config initialization
// ---------------------------------------------------------------------------

#[test]
fn inline_config_with_global_context_is_preserved() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut data = ConfigData {
        global_context: Some("System knowledge base".into()),
        ..Default::default()
    };
    data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config: Some(data),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        store.get_global_context().unwrap(),
        Some("System knowledge base".to_string())
    );
}

#[test]
fn inline_config_with_pre_existing_contexts_is_preserved() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut ctx = ContextMap::new();
    ctx.insert("/auth".into(), "Authentication docs".into());
    let mut c = coll(&docs_dir(&fx));
    c.context = Some(ctx);
    let mut data = ConfigData::default();
    data.collections.insert("docs".into(), c);
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config: Some(data),
        ..Default::default()
    })
    .unwrap();
    assert!(has_ctx(&store, "docs", "/auth", "Authentication docs"));
}

#[test]
fn inline_config_with_empty_collections_works() {
    let ws = TempDir::new().unwrap();
    let store = open_inline_with(db(&ws), vec![]);
    assert!(store.list_collections().unwrap().is_empty());
    assert!(store.list_contexts().unwrap().is_empty());
}

#[test]
fn inline_config_with_multiple_collection_options() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut docs_c = coll(&docs_dir(&fx));
    docs_c.ignore = Some(vec!["drafts/**".into()]);
    docs_c.include_by_default = Some(true);
    let mut notes_c = coll(&notes_dir(&fx));
    notes_c.include_by_default = Some(false);
    let mut data = ConfigData::default();
    data.collections.insert("docs".into(), docs_c);
    data.collections.insert("notes".into(), notes_c);
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config: Some(data),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(store.list_collections().unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// DB-only mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reopen_store_with_just_db_path_after_config_update() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let dbp = db(&ws);

    {
        let mut data = ConfigData {
            global_context: Some("Test knowledge base".into()),
            ..Default::default()
        };
        data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
        data.collections.insert("notes".into(), coll(&notes_dir(&fx)));
        let mut s1 = RqmdStore::open(RqmdStoreOptions {
            db_path: dbp.clone(),
            config: Some(data),
            ..Default::default()
        })
        .unwrap();
        s1.update(UpdateOptions::default()).await.unwrap();
        assert_eq!(s1.status().unwrap().total_documents, 6);
        s1.close().await;
    }

    let s2 = open_db_only_at(dbp);
    let mut names: Vec<_> = s2
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    names.sort();
    assert_eq!(names, vec!["docs".to_string(), "notes".to_string()]);
    assert!(!s2.search_lex("authentication", None, None).unwrap().is_empty());
    assert_eq!(
        s2.get_global_context().unwrap(),
        Some("Test knowledge base".to_string())
    );
    assert_eq!(s2.status().unwrap().total_documents, 6);
}

#[test]
fn config_sync_populates_store_collections() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut ctx = ContextMap::new();
    ctx.insert("/auth".into(), "Auth documentation".into());
    let mut c = coll(&docs_dir(&fx));
    c.context = Some(ctx);
    let mut data = ConfigData::default();
    data.collections.insert("docs".into(), c);
    let store = RqmdStore::open(RqmdStoreOptions {
        db_path: db(&ws),
        config: Some(data),
        ..Default::default()
    })
    .unwrap();
    let cols = store.list_collections().unwrap();
    assert_eq!(cols.len(), 1);
    assert_eq!(cols[0].name, "docs");
    assert_eq!(cols[0].pwd, docs_dir(&fx));
    assert!(has_ctx(&store, "docs", "/auth", "Auth documentation"));
}

/// sdk.test.ts "config hash skip: second init with same config skips sync".
/// Strengthened beyond the TS end-state check: a DB-only collection added
/// between two opens with the *same* config must survive, which is only true
/// if the config-hash gate actually skips the (delete-not-in-config) re-sync.
#[test]
fn config_hash_skip_preserves_db_only_collection() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let dbp = db(&ws);
    let make = || {
        let mut data = ConfigData::default();
        data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
        data
    };

    // First init with config A — syncs `docs` and records the config hash.
    {
        let _s1 = RqmdStore::open(RqmdStoreOptions {
            db_path: dbp.clone(),
            config: Some(make()),
            ..Default::default()
        })
        .unwrap();
    }

    // DB-only session adds `notes` (no config → no sync → hash unchanged).
    {
        let mut s2 = open_db_only_at(dbp.clone());
        s2.add_collection(
            "notes",
            AddCollectionOptions {
                path: notes_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
    }

    // Re-open with the SAME config A — hash matches, sync is skipped, so the
    // DB-only `notes` survives (without the gate it would be deleted).
    let s3 = RqmdStore::open(RqmdStoreOptions {
        db_path: dbp,
        config: Some(make()),
        ..Default::default()
    })
    .unwrap();
    let mut names: Vec<_> = s3
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    names.sort();
    assert_eq!(names, vec!["docs".to_string(), "notes".to_string()]);
}

/// Contract for the delete-not-in-config pass: re-opening with a *different*
/// config (different hash) re-syncs and drops collections absent from the new
/// config. Mirrors qmd `syncConfigToDb` (`store.ts:1117-1123`).
#[test]
fn changed_config_resyncs_and_drops_missing_collection() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let dbp = db(&ws);

    {
        let mut data = ConfigData::default();
        data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
        let _s1 = RqmdStore::open(RqmdStoreOptions {
            db_path: dbp.clone(),
            config: Some(data),
            ..Default::default()
        })
        .unwrap();
    }

    let mut data2 = ConfigData::default();
    data2.collections.insert("notes".into(), coll(&notes_dir(&fx)));
    let s2 = RqmdStore::open(RqmdStoreOptions {
        db_path: dbp,
        config: Some(data2),
        ..Default::default()
    })
    .unwrap();
    let names: Vec<_> = s2
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, vec!["notes".to_string()]);
}

#[test]
fn db_only_mode_supports_collection_mutations() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let dbp = db(&ws);

    {
        let mut data = ConfigData::default();
        data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
        let _s1 = RqmdStore::open(RqmdStoreOptions {
            db_path: dbp.clone(),
            config: Some(data),
            ..Default::default()
        })
        .unwrap();
    }

    {
        let mut s2 = open_db_only_at(dbp.clone());
        s2.add_collection(
            "notes",
            AddCollectionOptions {
                path: notes_dir(&fx),
                pattern: Some("**/*.md".into()),
                ignore: None,
            },
        )
        .unwrap();
        let mut names: Vec<_> = s2
            .list_collections()
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["docs".to_string(), "notes".to_string()]);
    }

    let s3 = open_db_only_at(dbp);
    let mut names: Vec<_> = s3
        .list_collections()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    names.sort();
    assert_eq!(names, vec!["docs".to_string(), "notes".to_string()]);
}

#[test]
fn db_only_mode_supports_context_mutations() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let dbp = db(&ws);

    {
        let mut data = ConfigData::default();
        data.collections.insert("docs".into(), coll(&docs_dir(&fx)));
        let mut s1 = RqmdStore::open(RqmdStoreOptions {
            db_path: dbp.clone(),
            config: Some(data),
            ..Default::default()
        })
        .unwrap();
        assert!(s1.add_context("docs", "/api", "API docs").unwrap());
        s1.set_global_context(Some("Global context".into())).unwrap();
    }

    let s2 = open_db_only_at(dbp);
    assert!(has_ctx(&s2, "docs", "/api", "API docs"));
    assert!(has_ctx(&s2, "*", "/", "Global context"));
}

// ---------------------------------------------------------------------------
// search (unified API) — non-LLM paths
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_with_pre_expanded_queries_and_rerank_false() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store =
        open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store
        .search(SearchOptions {
            queries: Some(vec![
                ExpandedQuery {
                    type_: ExpandedQueryType::Lex,
                    query: "authentication JWT".into(),
                    line: None,
                },
                ExpandedQuery {
                    type_: ExpandedQueryType::Lex,
                    query: "login session".into(),
                    line: None,
                },
            ]),
            rerank: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!results.is_empty());
    store.close().await;
}

#[tokio::test]
async fn search_forwards_candidate_limit_to_structured_search() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store =
        open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store
        .search(SearchOptions {
            queries: Some(vec![
                ExpandedQuery {
                    type_: ExpandedQueryType::Lex,
                    query: "authentication".into(),
                    line: None,
                },
                ExpandedQuery {
                    type_: ExpandedQueryType::Lex,
                    query: "meeting".into(),
                    line: None,
                },
            ]),
            limit: Some(5),
            candidate_limit: Some(1),
            rerank: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    store.close().await;
}

// ---------------------------------------------------------------------------
// embed (validation only — happy path needs a real model; see #[ignore] below)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embed_rejects_invalid_batch_limits() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_inline_with(db(&ws), vec![("docs", docs_dir(&fx))]);

    let e1 = store
        .embed(StoreOpsEmbedOptions {
            max_docs_per_batch: Some(0),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(format!("{e1}").contains("maxDocsPerBatch"));

    let e2 = store
        .embed(StoreOpsEmbedOptions {
            max_batch_bytes: Some(0),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert!(format!("{e2}").contains("maxBatchBytes"));

    store.close().await;
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn close_is_async_and_does_not_throw() {
    let ws = TempDir::new().unwrap();
    let store = open_inline_with(db(&ws), vec![]);
    store.close().await;
}

// ---------------------------------------------------------------------------
// GGUF-only happy paths (sdk skips these under CI; here they need a real model)
// ---------------------------------------------------------------------------

/// Distinct embedded-vector count for a collection (seq-0 chunks of active docs).
fn vec_count(store: &RqmdStore, collection: &str) -> i64 {
    store.internal().with_connection(|c| {
        c.query_row(
            "SELECT COUNT(DISTINCT v.hash) FROM documents d \
             LEFT JOIN content_vectors v ON v.hash = d.hash AND v.seq = 0 \
             WHERE d.active = 1 AND d.collection = ?1",
            rqmd_core::db::rusqlite::params![collection],
            |r| r.get(0),
        )
        .unwrap()
    })
}

#[tokio::test]
#[ignore = "requires a real GGUF embed model"]
async fn embed_forwards_batch_limit_options_gguf() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store = open_indexed(&ws, vec![("docs", docs_dir(&fx))]).await;
    let result = store
        .embed(StoreOpsEmbedOptions {
            max_docs_per_batch: Some(1),
            max_batch_bytes: Some(1024 * 1024),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.docs_processed, 3);
    assert_eq!(result.chunks_embedded, 3);
    store.close().await;
}

#[tokio::test]
#[ignore = "requires a real GGUF embed model"]
async fn embed_scopes_pending_documents_to_collection_gguf() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store =
        open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let result = store
        .embed(StoreOpsEmbedOptions {
            collection: Some("docs".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.docs_processed, 3);
    assert_eq!(vec_count(&store, "docs"), 3);
    assert_eq!(vec_count(&store, "notes"), 0);
    store.close().await;
}

#[tokio::test]
#[ignore = "requires a real GGUF embed model"]
async fn embed_with_force_only_clears_requested_collection_gguf() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let mut store =
        open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    store.embed(StoreOpsEmbedOptions::default()).await.unwrap();
    assert_eq!(vec_count(&store, "docs"), 3);
    assert_eq!(vec_count(&store, "notes"), 3);

    let result = store
        .embed(StoreOpsEmbedOptions {
            force: true,
            collection: Some("docs".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(result.docs_processed, 3);
    assert_eq!(vec_count(&store, "docs"), 3);
    assert_eq!(vec_count(&store, "notes"), 3);
    store.close().await;
}

#[tokio::test]
#[ignore = "requires real GGUF generate/embed models"]
async fn search_with_query_expansion_gguf() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store =
        open_indexed(&ws, vec![("docs", docs_dir(&fx)), ("notes", notes_dir(&fx))]).await;
    let results = store
        .search(SearchOptions {
            query: Some("authentication".into()),
            rerank: Some(false),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!results.is_empty());
    store.close().await;
}

#[tokio::test]
#[ignore = "requires real GGUF generate/embed models"]
async fn expand_query_gguf() {
    let ws = TempDir::new().unwrap();
    let fx = make_sdk_fixture();
    let store = open_docs_notes_inline(&ws, &fx);
    let queries = store.expand_query("authentication", None).await.unwrap();
    assert!(!queries.is_empty());
    store.close().await;
}
