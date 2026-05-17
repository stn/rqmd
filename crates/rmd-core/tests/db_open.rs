//! Integration tests for `rmd_core::db` — covers file-backed open,
//! deterministic open failures, the `sqlite-vec` probe, and a vec0
//! virtual-table round-trip to confirm the bundled extension actually
//! links and executes on this host.

use rmd_core::db::{self, Error};
use tempfile::TempDir;

#[test]
fn open_database_creates_file_backed_connection() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("test.sqlite");

    let conn = db::open_database(&path).unwrap();
    conn.execute("CREATE TABLE t (x INTEGER)", []).unwrap();
    conn.execute("INSERT INTO t VALUES (?1)", [42i64]).unwrap();
    let n: i64 = conn.query_row("SELECT x FROM t", [], |r| r.get(0)).unwrap();
    assert_eq!(n, 42);
    drop(conn);

    assert!(path.exists(), "expected sqlite file at {}", path.display());
}

#[test]
fn open_database_returns_open_error_with_path_on_failure() {
    // Opening a *directory* as a database deterministically fails with
    // SQLITE_CANTOPEN on every platform.
    let tmp = TempDir::new().unwrap();
    let dir_as_db = tmp.path(); // the directory itself

    let err = db::open_database(dir_as_db).expect_err("opening a directory must fail");
    match err {
        Error::Open { path, source: _ } => {
            assert_eq!(path, dir_as_db);
        }
        other => panic!("expected Error::Open, got: {other:?}"),
    }
}

#[test]
fn probe_sqlite_vec_succeeds_on_fresh_connection() {
    let conn = db::open_in_memory().unwrap();
    db::probe_sqlite_vec(&conn).expect("sqlite-vec should be available on a fresh connection");
}

#[test]
fn vec_version_returns_non_empty_string() {
    let conn = db::open_in_memory().unwrap();
    let v: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .unwrap();
    assert!(!v.is_empty(), "vec_version() returned empty string");
}

#[test]
fn vec0_virtual_table_roundtrip() {
    let conn = db::open_in_memory().unwrap();

    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vec0(\
            id INTEGER PRIMARY KEY,\
            e float[4]\
         )",
    )
    .unwrap();

    // Insert three vectors. sqlite-vec accepts JSON-encoded float arrays.
    let rows: &[(i64, &str)] = &[
        (1, "[1.0, 0.0, 0.0, 0.0]"),
        (2, "[0.0, 1.0, 0.0, 0.0]"),
        (3, "[0.9, 0.1, 0.0, 0.0]"),
    ];
    for (id, vec) in rows {
        conn.execute(
            "INSERT INTO t(id, e) VALUES (?1, ?2)",
            rusqlite::params![id, vec],
        )
        .unwrap();
    }

    // KNN query: nearest neighbour of [1, 0, 0, 0] should be id=1
    // (exact match), with id=3 next.
    let mut stmt = conn
        .prepare(
            "SELECT id FROM t \
             WHERE e MATCH ?1 AND k = 2 \
             ORDER BY distance",
        )
        .unwrap();
    let ids: Vec<i64> = stmt
        .query_map(rusqlite::params!["[1.0, 0.0, 0.0, 0.0]"], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();

    assert_eq!(ids, vec![1, 3]);
}
