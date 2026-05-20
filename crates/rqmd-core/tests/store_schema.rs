//! `Store::open` creates the full schema and `Store::open` on an existing
//! file is idempotent.

use rqmd_core::Store;
use tempfile::NamedTempFile;

fn expected_tables() -> &'static [&'static str] {
    &[
        "content",
        "documents",
        "llm_cache",
        "content_vectors",
        "store_collections",
        "store_config",
        "documents_fts",
    ]
}

#[test]
fn open_creates_every_table() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).expect("open");

    for table in expected_tables() {
        let exists: i64 = store
            .with_connection(|c| {
                c.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = ?",
                    rusqlite::params![table],
                    |row| row.get::<_, i64>(0),
                )
            })
            .expect("sqlite_master query");
        assert_eq!(exists, 1, "table {table} should exist");
    }
}

#[test]
fn open_is_idempotent() {
    let tmp = NamedTempFile::new().unwrap();
    {
        let _ = Store::open(tmp.path()).expect("first open");
    }
    {
        let _ = Store::open(tmp.path()).expect("second open");
    }

    let store = Store::open(tmp.path()).expect("third open");
    // CJK rebuild marker should be set after initialise.
    let version: Option<String> = store
        .with_connection(|c| {
            c.query_row(
                "SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'",
                [],
                |row| row.get::<_, String>(0),
            )
        })
        .ok();
    assert_eq!(version.as_deref(), Some("1"));
}

#[test]
fn open_creates_triggers() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).expect("open");

    for trigger in &["documents_ai", "documents_ad", "documents_au"] {
        let exists: i64 = store
            .with_connection(|c| {
                c.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                    rusqlite::params![trigger],
                    |row| row.get::<_, i64>(0),
                )
            })
            .expect("sqlite_master query");
        assert_eq!(exists, 1, "trigger {trigger} should exist");
    }
}
