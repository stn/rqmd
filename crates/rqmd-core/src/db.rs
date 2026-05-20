//! SQLite connection helpers with `sqlite-vec` pre-registered.
//!
//! Port of `tobi/qmd`'s `src/db.ts` into Rust. The TypeScript original was a
//! cross-runtime shim that smoothed over `bun:sqlite` vs `better-sqlite3`
//! and worked around Apple SQLite's `SQLITE_OMIT_LOAD_EXTENSION`. Rust has
//! one mature binding (`rusqlite`) and the `bundled` feature ships its own
//! SQLite build with `load_extension` enabled, so the shim collapses to a
//! thin module:
//!
//! * [`open_database`] / [`open_in_memory`] open a connection.
//! * Before the first connection is opened, `sqlite-vec` is registered as a
//!   global auto-extension so every subsequent `Connection` has the `vec0`
//!   virtual-table module and `vec_version()` available.
//! * [`probe_sqlite_vec`] is a per-connection sanity check — it does **not**
//!   load the extension (auto-extension runs at connection-open time and
//!   cannot retroactively attach to an existing handle).
//!
//! Callers are expected to use `rusqlite` directly: `Connection`, `Statement`,
//! `Transaction`, `params!`, etc. Re-exports live in this module so the
//! public surface stays in one place (`rqmd_core::db::Connection`, …).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use rusqlite::{self, params, Connection, Row, Statement, Transaction};

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to open database at {path}: {source}", path = path.display())]
    Open {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },

    #[error("sqlite operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("sqlite-vec extension unavailable: {0}")]
    VecUnavailable(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Public API
// ============================================================================

/// Open a file-backed SQLite database with `sqlite-vec` pre-registered.
///
/// The first call in the process performs the global
/// `sqlite3_auto_extension` registration; later calls reuse the cached
/// outcome. If registration ever failed, that error is returned every time.
pub fn open_database<P: AsRef<Path>>(path: P) -> Result<Connection> {
    register_sqlite_vec()?;
    let path = path.as_ref();
    Connection::open(path).map_err(|source| Error::Open {
        path: path.to_path_buf(),
        source,
    })
}

/// Open an in-memory database. Convenience wrapper that avoids any
/// path-vs-`:memory:` ambiguity.
pub fn open_in_memory() -> Result<Connection> {
    register_sqlite_vec()?;
    Ok(Connection::open_in_memory()?)
}

/// Probe whether `sqlite-vec` is usable on `conn` by querying
/// `vec_version()`.
///
/// This is **not** a loader — auto-extensions only fire when a connection is
/// opened, and that already happened in [`open_database`]. The probe exists
/// so callers can detect a registration failure and degrade gracefully
/// (mirroring qmd's `store.ts` BM25-only fallback).
pub fn probe_sqlite_vec(conn: &Connection) -> Result<()> {
    match conn.query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0)) {
        Ok(v) if !v.is_empty() => Ok(()),
        Ok(_) => Err(Error::VecUnavailable(
            "vec_version() returned empty string".into(),
        )),
        Err(e) => Err(Error::VecUnavailable(e.to_string())),
    }
}

// ============================================================================
// Internals
// ============================================================================

/// Outcome of the one-shot global `sqlite3_auto_extension` registration.
static VEC_REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Register `sqlite-vec` as a SQLite auto-extension, exactly once per process.
///
/// `sqlite3_auto_extension` only affects connections opened *after*
/// registration, so this must run before any [`Connection::open`] call that
/// expects `vec0` to be available — which is why [`open_database`] and
/// [`open_in_memory`] call it before opening.
fn register_sqlite_vec() -> Result<()> {
    // The type `sqlite3_auto_extension` expects in this rusqlite version.
    // Spelling it out keeps the transmute below honest: if the FFI signature
    // ever changes, this alias breaks compilation rather than silently
    // accepting a mismatched function pointer.
    type ExtInit = unsafe extern "C" fn(
        db: *mut rusqlite::ffi::sqlite3,
        pz_err_msg: *mut *mut std::os::raw::c_char,
        p_api: *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::os::raw::c_int;

    let outcome = VEC_REGISTERED.get_or_init(|| {
        // SAFETY:
        //   * `sqlite_vec::sqlite3_vec_init` is declared as
        //     `extern "C" fn()` (zero arg) in the `sqlite-vec` crate so it
        //     can be exposed without depending on a specific libsqlite3-sys
        //     version. The real C ABI is the three-argument `ExtInit`
        //     prototype above — pointer-transmute through `*const ()` is
        //     the documented way to register it (see the sqlite-vec README).
        //   * Guarded by `OnceLock::get_or_init`, so registered at most once
        //     per process.
        let rc = unsafe {
            let init_ptr: *const () = sqlite_vec::sqlite3_vec_init as *const ();
            let init: ExtInit = std::mem::transmute(init_ptr);
            rusqlite::ffi::sqlite3_auto_extension(Some(init))
        };
        if rc == rusqlite::ffi::SQLITE_OK {
            Ok(())
        } else {
            Err(format!("sqlite3_auto_extension returned {rc}"))
        }
    });
    outcome
        .as_ref()
        .map(|_| ())
        .map_err(|msg| Error::VecUnavailable(msg.clone()))
}

// ============================================================================
// Unit tests (pure logic)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_sqlite_vec_is_idempotent() {
        // Calling twice must not panic and must return Ok both times.
        register_sqlite_vec().unwrap();
        register_sqlite_vec().unwrap();
    }

    #[test]
    fn open_in_memory_has_vec_available() {
        let conn = open_in_memory().unwrap();
        probe_sqlite_vec(&conn).unwrap();
    }
}
