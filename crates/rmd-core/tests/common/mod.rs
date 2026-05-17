//! Shared test helpers for the `rmd-core` integration tests.
//!
//! Each test binary that needs it does `mod common;` and pulls in
//! [`EnvGuard`] for safely snapshotting/restoring the env variables that
//! `paths::config_dir()` and friends consult.

#![allow(dead_code)] // not every test file uses every helper

use std::ffi::{OsStr, OsString};

/// RAII snapshot/restore for the env vars rmd-core reads.
///
/// `EnvGuard::capture` records the current values; mutating helpers update
/// them in place; `Drop` restores the original values (including absence).
///
/// Mutating env vars is `unsafe` since Rust 1.86 because reads from other
/// threads are not synchronized. Tests that use this guard must be
/// annotated with `#[serial_test::serial(env)]` so only one runs at a time.
pub struct EnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    pub fn capture(keys: &[&'static str]) -> Self {
        let saved = keys.iter().map(|&k| (k, std::env::var_os(k))).collect();
        Self { saved }
    }

    pub fn set(&self, key: &str, value: impl AsRef<OsStr>) {
        // SAFETY: `serial_test::serial(env)` serializes env-mutating tests.
        unsafe { std::env::set_var(key, value) };
    }

    pub fn remove(&self, key: &str) {
        // SAFETY: see `set`.
        unsafe { std::env::remove_var(key) };
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.saved {
            match v {
                Some(val) => {
                    // SAFETY: see `set`.
                    unsafe { std::env::set_var(k, val) };
                }
                None => {
                    // SAFETY: see `set`.
                    unsafe { std::env::remove_var(k) };
                }
            }
        }
    }
}

/// The env vars `paths::rmd_homedir()` and `paths::config_dir()` consult.
pub const PATH_ENV_KEYS: &[&str] = &["HOME", "USERPROFILE", "RMD_CONFIG_DIR", "XDG_CONFIG_HOME"];
