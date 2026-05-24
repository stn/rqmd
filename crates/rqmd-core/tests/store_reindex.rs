//! End-to-end test for `reindex_collection`.

use rqmd_core::Store;
use rqmd_core::store::reindex::reindex_collection;
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
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
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
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();
    let second = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
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
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    write(&docs, "a.md", "# Alpha\n\nsecond");

    let second = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
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
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    fs::remove_file(docs.path().join("b.md")).unwrap();

    let second = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    assert_eq!(second.removed, 1);
}

// --- `.qmdignore` / `.gitignore` behaviour ---------------------------------
//
// rqmd deliberately does NOT read `.gitignore` (unlike the `ignore` crate's
// default). It honours only a dedicated `.qmdignore` (per directory + global).

/// `.gitignore` is not consulted, so a file it lists is still indexed.
/// (File-level counterpart of `reindex_indexes_files_under_gitignored_dir`;
/// both are kept because they reproduce the original bug at file and directory
/// granularity.)
#[test]
fn reindex_ignores_gitignore() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# A\n\none");
    write(&docs, "b.md", "# B\n\ntwo");
    write(&docs, ".gitignore", "b.md\n");

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    assert_eq!(result.indexed, 2); // .gitignore not read → b.md still indexed
}

/// Reproduces the original bug: a directory listed in `.gitignore` (like the
/// vault's `9-old`) must still be indexed because `.gitignore` is ignored.
#[test]
fn reindex_indexes_files_under_gitignored_dir() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# A\n\none");
    write(&docs, "archive/x.md", "# X\n\ntwo");
    write(&docs, ".gitignore", "archive/\n");

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    assert_eq!(result.indexed, 2); // archive/ indexed despite .gitignore
}

/// A directory pattern in `.qmdignore` prunes the whole subtree.
#[test]
fn reindex_respects_qmdignore_directory() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# A\n\none");
    write(&docs, "archive/x.md", "# X\n\ntwo");
    write(&docs, ".qmdignore", "archive/\n");

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    assert_eq!(result.indexed, 1); // only a.md; archive/ pruned
}

/// A file pattern in `.qmdignore` excludes matching `.md` files — this is the
/// case the override-whitelist short-circuit used to break.
#[test]
fn reindex_respects_qmdignore_file_pattern() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "keep.md", "# Keep\n\none");
    write(&docs, "note.draft.md", "# Draft\n\ntwo");
    write(&docs, ".qmdignore", "*.draft.md\n");

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(c, docs.path(), "**/*.md", "docs", &[], &[], |_| {})
        })
        .unwrap();

    assert_eq!(result.indexed, 1); // only keep.md
}

/// A global `~/.qmdignore`-style file (passed via `extra_ignore_files`) applies
/// to the walk. NOTE: `add_ignore` roots patterns at the CWD, not the
/// collection, so global patterns must be NON-ANCHORED (no leading `/`).
#[test]
fn reindex_respects_global_ignore_file() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "a.md", "# A\n\none");
    write(&docs, "archive/x.md", "# X\n\ntwo");

    let global_dir = TempDir::new().unwrap();
    let global = global_dir.path().join("global.qmdignore");
    fs::write(&global, "archive/\n").unwrap(); // non-anchored

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(
                c,
                docs.path(),
                "**/*.md",
                "docs",
                &[],
                std::slice::from_ref(&global),
                |_| {},
            )
        })
        .unwrap();

    assert_eq!(result.indexed, 1); // archive/ excluded by global ignore
}

/// A per-directory `.qmdignore` overrides the global file (custom ignore has
/// higher precedence than the explicit global ignore).
#[test]
fn reindex_local_qmdignore_overrides_global() {
    let db = NamedTempFile::new().unwrap();
    let mut store = Store::open(db.path()).unwrap();
    let docs = TempDir::new().unwrap();
    write(&docs, "keep.md", "# Keep\n\nbody");
    write(&docs, ".qmdignore", "!keep.md\n"); // local un-ignore wins

    let global_dir = TempDir::new().unwrap();
    let global = global_dir.path().join("global.qmdignore");
    fs::write(&global, "keep.md\n").unwrap(); // global would ignore it

    let result = store
        .with_connection_mut(|c| {
            reindex_collection(
                c,
                docs.path(),
                "**/*.md",
                "docs",
                &[],
                std::slice::from_ref(&global),
                |_| {},
            )
        })
        .unwrap();

    assert_eq!(result.indexed, 1); // local .qmdignore beats global
}
