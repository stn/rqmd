//! End-to-end test for `reindex_collection`.

use rqmd_core::store::reindex::reindex_collection;
use rqmd_core::Store;
use std::fs;
use tempfile::{NamedTempFile, TempDir};

fn write(dir: &TempDir, rel: &str, body: &str) {
    let p = dir.path().join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(p, body).unwrap();
}

#[test]
fn reindex_initial_run_indexes_files() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();

    write(&docs, "a.md", "# Alpha\n\nbody one");
    write(&docs, "sub/b.md", "# Beta\n\nbody two");

    let result = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    assert_eq!(result.indexed, 2);
    assert_eq!(result.updated, 0);
    assert_eq!(result.unchanged, 0);
    assert_eq!(result.removed, 0);
}

#[test]
fn reindex_second_run_reports_unchanged() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# Alpha\n\nbody");

    let _ = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();
    let second = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    assert_eq!(second.indexed, 0);
    assert_eq!(second.unchanged, 1);
}

#[test]
fn reindex_updates_changed_content() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# Alpha\n\nfirst");

    let _ = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    write(&docs, "a.md", "# Alpha\n\nsecond");

    let second = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    assert_eq!(second.indexed, 0);
    assert_eq!(second.updated, 1);
    assert_eq!(second.unchanged, 0);
}

#[test]
fn reindex_deactivates_missing_files() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# Alpha\n\none");
    write(&docs, "b.md", "# Beta\n\ntwo");

    let _ = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    fs::remove_file(docs.path().join("b.md")).unwrap();

    let second = store
        .with_connection_mut(|c| reindex_collection(c, docs.path(), "**/*.md", "docs", &[], |_| {}))
        .unwrap();

    assert_eq!(second.removed, 1);
}
