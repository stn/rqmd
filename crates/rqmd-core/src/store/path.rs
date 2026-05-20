//! Path utilities and the production-mode flag.
//!
//! Port of the path-handling portion of `tobi/qmd`'s `src/store.ts`
//! (lines 339–567), plus a small RFC-3339 timestamp helper used by
//! [`super::reindex`].
//!
//! ## Path semantics
//!
//! The TS module deliberately did not use Node's `path.resolve` because qmd
//! supports three flavours of absolute path that need to coexist:
//!
//! * Unix: `/usr/local`
//! * Windows native: `C:\path` (or `C:/path` after normalisation)
//! * Git Bash on Windows: `/c/Users/...`
//!
//! [`resolve`] preserves whichever flavour the caller passed in, using
//! forward slashes throughout for consistency.
//!
//! ## Production mode
//!
//! [`enable_production_mode`] / [`_reset_production_mode_for_testing`] gate
//! [`default_db_path`]'s fallback behaviour. When the flag is off (the
//! default), `default_db_path` returns [`super::Error::DbPathNotSet`] unless
//! `RQMD_INDEX_PATH` is set — this prevents tests from accidentally writing
//! to the global cache. The CLI binary calls `enable_production_mode` at
//! startup. (Matches qmd `store.ts:522–531`.)

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::paths as crate_paths;

use super::{Error, Result};

// ============================================================================
// Absolute-path detection
// ============================================================================

/// Check if `path` is absolute by qmd's combined Unix / Windows / Git-Bash
/// rules. Returns `false` for the empty string.
///
/// Mirrors `isAbsolutePath` (`store.ts:353–377`).
pub fn is_absolute_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }

    if path.starts_with('/') {
        let bytes = path.as_bytes();
        if !is_wsl() && bytes.len() >= 3 && bytes[2] == b'/' {
            let drive = bytes[1];
            if matches!(drive, b'c'..=b'z' | b'C'..=b'Z') {
                return true;
            }
        }
        return true;
    }

    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return true;
    }

    false
}

/// Whether this process is running inside WSL. Inspected per call so tests
/// can flip the env vars between operations.
fn is_wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some() || std::env::var_os("WSL_INTEROP").is_some()
}

/// Convert backslashes to forward slashes.
///
/// Mirrors `normalizePathSeparators` (`store.ts:383–385`).
pub fn normalize_path_separators(path: &str) -> Cow<'_, str> {
    if path.contains('\\') {
        Cow::Owned(path.replace('\\', "/"))
    } else {
        Cow::Borrowed(path)
    }
}

/// Return the relative path from `prefix` to `path`, or `None` if `path` is
/// not under `prefix`. Returns `Some("")` if they are equal.
///
/// Mirrors `getRelativePathFromPrefix` (`store.ts:400–425`).
pub fn get_relative_path_from_prefix(path: &str, prefix: &str) -> Option<String> {
    if prefix.is_empty() {
        return None;
    }

    let np = normalize_path_separators(path);
    let pp = normalize_path_separators(prefix);

    if np == pp {
        return Some(String::new());
    }

    let prefix_with_slash: Cow<'_, str> = if pp.ends_with('/') {
        Cow::Borrowed(pp.as_ref())
    } else {
        Cow::Owned(format!("{}/", pp))
    };

    np.strip_prefix(prefix_with_slash.as_ref())
        .map(|s| s.to_string())
}

// ============================================================================
// resolve
// ============================================================================

/// Concatenate path segments, treating absolute segments as resets.
///
/// Mirrors `resolve(...paths)` (`store.ts:427–519`). All separators are
/// normalised to `/`; Windows drive letters and Git-Bash drives are
/// preserved in the output.
pub fn resolve(parts: &[&str]) -> String {
    assert!(
        !parts.is_empty(),
        "resolve: at least one path segment is required"
    );

    let normalised: Vec<String> = parts
        .iter()
        .map(|p| normalize_path_separators(p).into_owned())
        .collect();

    let mut result;
    let mut windows_drive = String::new();

    let first = &normalised[0];
    if is_absolute_path(first) {
        result = first.clone();
        if let Some(drive) = take_windows_drive(first) {
            windows_drive = drive;
            result = first[2..].to_string();
        } else if let Some(drive) = take_git_bash_drive(first) {
            windows_drive = drive;
            result = first[2..].to_string();
        }
    } else {
        let pwd_raw = std::env::var("PWD").unwrap_or_else(|_| {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".into())
        });
        let pwd = normalize_path_separators(&pwd_raw).into_owned();
        if let Some(drive) = take_windows_drive(&pwd) {
            windows_drive = drive;
            result = format!("{}/{}", &pwd[2..], first);
        } else {
            result = format!("{}/{}", pwd, first);
        }
    }

    for p in &normalised[1..] {
        if is_absolute_path(p) {
            result = p.clone();
            if let Some(drive) = take_windows_drive(p) {
                windows_drive = drive;
                result = p[2..].to_string();
            } else if let Some(drive) = take_git_bash_drive(p) {
                windows_drive = drive;
                result = p[2..].to_string();
            } else {
                windows_drive.clear();
            }
        } else {
            result.push('/');
            result.push_str(p);
        }
    }

    // Normalise `.` and `..` components.
    let mut normalised_parts: Vec<&str> = Vec::new();
    for part in result.split('/').filter(|s| !s.is_empty()) {
        match part {
            "." => {}
            ".." => {
                normalised_parts.pop();
            }
            _ => normalised_parts.push(part),
        }
    }

    let final_path = format!("/{}", normalised_parts.join("/"));
    if windows_drive.is_empty() {
        final_path
    } else {
        format!("{}{}", windows_drive, final_path)
    }
}

fn take_windows_drive(p: &str) -> Option<String> {
    let bytes = p.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        Some(p[..2].to_string())
    } else {
        None
    }
}

fn take_git_bash_drive(p: &str) -> Option<String> {
    if is_wsl() {
        return None;
    }
    let bytes = p.as_bytes();
    if bytes.len() >= 3 && bytes[0] == b'/' && bytes[2] == b'/' {
        let drive = bytes[1];
        if matches!(drive, b'c'..=b'z' | b'C'..=b'Z') {
            return Some(format!("{}:", drive.to_ascii_uppercase() as char));
        }
    }
    None
}

// ============================================================================
// Production-mode flag
// ============================================================================

static PRODUCTION_MODE: AtomicBool = AtomicBool::new(false);

/// Enable production mode. The CLI binary should call this once at startup;
/// tests must never call it. Mirrors `enableProductionMode`
/// (`store.ts:524–526`).
pub fn enable_production_mode() {
    PRODUCTION_MODE.store(true, Ordering::Relaxed);
}

/// Reset production mode. Test-only.
#[doc(hidden)]
pub fn _reset_production_mode_for_testing() {
    PRODUCTION_MODE.store(false, Ordering::Relaxed);
}

fn production_mode_enabled() -> bool {
    PRODUCTION_MODE.load(Ordering::Relaxed)
}

// ============================================================================
// default_db_path
// ============================================================================

/// Resolve the default SQLite index path.
///
/// Precedence (mirrors `getDefaultDbPath` at `store.ts:533–551`):
///
/// 1. `RQMD_INDEX_PATH` env var — always honoured.
/// 2. If production mode is not enabled, returns [`Error::DbPathNotSet`].
/// 3. Otherwise `<cache_dir>/<index_name>.sqlite` where `cache_dir` follows
///    [`crate_paths::cache_dir`] precedence.
pub fn default_db_path(index_name: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("RQMD_INDEX_PATH")
        && !p.is_empty()
    {
        return Ok(PathBuf::from(p));
    }

    if !production_mode_enabled() {
        return Err(Error::DbPathNotSet);
    }

    let dir = crate_paths::cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let name = index_name.unwrap_or("index");
    Ok(dir.join(format!("{name}.sqlite")))
}

// ============================================================================
// Miscellaneous
// ============================================================================

/// Current working directory. Mirrors `getPwd()` (`store.ts:553–555`).
pub fn pwd() -> PathBuf {
    if let Ok(p) = std::env::var("PWD")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Canonicalise a path; fall back to the input on error. Mirrors
/// `getRealPath` (`store.ts:557–567`).
pub fn real_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Home directory (delegates to [`crate_paths::rqmd_homedir`]). Matches
/// TS `homedir()` (`store.ts:339`) within the renamed `rqmd` namespace.
pub fn homedir() -> PathBuf {
    crate_paths::rqmd_homedir()
}

/// Current time formatted as RFC 3339 (`YYYY-MM-DDTHH:MM:SS.sssZ`), used
/// for `created_at` / `modified_at` columns. Avoids pulling in a
/// time-handling crate for one call site.
pub fn now_rfc3339() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339_utc(dur.as_secs(), dur.subsec_millis())
}

/// Format `(seconds_since_epoch, milliseconds)` as RFC 3339 in UTC.
/// Algorithm: Howard Hinnant's days-from-civil (date.h), public domain.
pub fn format_rfc3339_utc(epoch_secs: u64, millis: u32) -> String {
    let days = (epoch_secs / 86_400) as i64;
    let time_of_day = epoch_secs % 86_400;
    let hour = time_of_day / 3_600;
    let minute = (time_of_day % 3_600) / 60;
    let second = time_of_day % 60;

    let (year, month, day) = civil_from_days(days);

    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z"
    )
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

/// Inverse of [`civil_from_days`]. Returns days since 1970-01-01 (negative
/// before the epoch). Public for status.rs's `days_stale` calculation.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) as i64 + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parse an ISO-8601/RFC-3339 timestamp prefix (`YYYY-MM-DD...`) and return
/// days since 1970-01-01. Returns `None` if the first 10 bytes do not look
/// like a valid date. The remainder of the string (time component, timezone)
/// is ignored — sufficient for whole-day staleness arithmetic.
pub fn parse_rfc3339_to_epoch_days(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() < 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year: i32 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
    let month: u32 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
    let day: u32 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

/// Days between `now` and `then_rfc3339`. Negative if `then` is in the future.
pub fn days_since_rfc3339(then_rfc3339: &str) -> Option<i64> {
    let then_days = parse_rfc3339_to_epoch_days(then_rfc3339)?;
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    let now_days = now_secs.div_euclid(86_400);
    Some(now_days - then_days)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Env vars that the path helpers consult. Saved/restored wholesale by
    /// [`EnvGuard`] so env-mutating tests (all `#[serial]`) cannot leak state.
    const ENV_KEYS: &[&str] = &["PWD", "WSL_DISTRO_NAME", "WSL_INTEROP", "HOME", "RQMD_INDEX_PATH"];

    /// RAII guard that snapshots `ENV_KEYS` on construction and restores them
    /// on drop — even if the test panics. `set`/`unset` mutate the process
    /// environment (`unsafe` since Rust 2024). Pair with `#[serial]`.
    ///
    /// Invariant: only ever `set`/`unset` keys listed in [`ENV_KEYS`].
    /// Anything outside that set is not snapshotted and would therefore leak
    /// past the guard's `Drop`. (All current call sites honour this.)
    struct EnvGuard(Vec<(&'static str, Option<String>)>);

    impl EnvGuard {
        fn new() -> Self {
            EnvGuard(ENV_KEYS.iter().map(|k| (*k, std::env::var(k).ok())).collect())
        }
        fn set(&self, k: &str, v: &str) {
            unsafe { std::env::set_var(k, v) };
        }
        fn unset(&self, k: &str) {
            unsafe { std::env::remove_var(k) };
        }
        /// PWD set to `pwd`, WSL detection vars cleared — the common setup for
        /// `resolve` tests.
        fn with_pwd_no_wsl(pwd: &str) -> Self {
            let g = Self::new();
            g.set("PWD", pwd);
            g.unset("WSL_DISTRO_NAME");
            g.unset("WSL_INTEROP");
            g
        }
        /// WSL detection vars cleared (for Git-Bash drive resolution).
        fn no_wsl() -> Self {
            let g = Self::new();
            g.unset("WSL_DISTRO_NAME");
            g.unset("WSL_INTEROP");
            g
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                unsafe {
                    match v {
                        Some(val) => std::env::set_var(k, val),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    #[test]
    fn is_absolute_unix() {
        assert!(is_absolute_path("/usr"));
        assert!(is_absolute_path("/"));
        assert!(!is_absolute_path("usr"));
        assert!(!is_absolute_path(""));
    }

    #[test]
    fn is_absolute_windows_native() {
        assert!(is_absolute_path("C:\\Users"));
        assert!(is_absolute_path("c:/users"));
        assert!(!is_absolute_path("CC:/users"));
    }

    #[test]
    fn normalize_separators_roundtrip() {
        assert_eq!(normalize_path_separators("a/b/c"), "a/b/c");
        assert_eq!(normalize_path_separators("a\\b\\c"), "a/b/c");
    }

    #[test]
    fn relative_from_prefix() {
        assert_eq!(
            get_relative_path_from_prefix("/a/b/c", "/a"),
            Some("b/c".into())
        );
        assert_eq!(get_relative_path_from_prefix("/a", "/a"), Some("".into()));
        assert_eq!(get_relative_path_from_prefix("/a/b", "/c"), None);
        assert_eq!(get_relative_path_from_prefix("/a/b", ""), None);
    }

    #[test]
    fn rfc3339_formatter_known_dates() {
        assert_eq!(format_rfc3339_utc(0, 0), "1970-01-01T00:00:00.000Z");
        // 2024-01-01T00:00:00Z = 1_704_067_200
        assert_eq!(
            format_rfc3339_utc(1_704_067_200, 0),
            "2024-01-01T00:00:00.000Z"
        );
        // Leap day: 2024-02-29T12:34:56.789Z = 1_709_210_096
        assert_eq!(
            format_rfc3339_utc(1_709_210_096, 789),
            "2024-02-29T12:34:56.789Z"
        );
    }

    #[test]
    fn days_from_civil_round_trips() {
        // Epoch.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        // 2024-01-01 is 19_723 days after epoch.
        assert_eq!(days_from_civil(2024, 1, 1), 19_723);
        // Leap day.
        assert_eq!(days_from_civil(2024, 2, 29), 19_782);
        // Round-trip against civil_from_days.
        for d in [0_i64, 1, 100, 365, 366, 19_723, 19_782, 50_000] {
            let (y, m, day) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, day), d);
        }
    }

    #[test]
    fn parse_rfc3339_to_epoch_days_handles_prefix() {
        assert_eq!(parse_rfc3339_to_epoch_days("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(
            parse_rfc3339_to_epoch_days("2024-01-01T12:34:56.789Z"),
            Some(19_723)
        );
        assert_eq!(parse_rfc3339_to_epoch_days("2024-02-29"), Some(19_782));
        // Garbage.
        assert_eq!(parse_rfc3339_to_epoch_days("not-a-date"), None);
        assert_eq!(parse_rfc3339_to_epoch_days(""), None);
        // Out-of-range month/day.
        assert_eq!(parse_rfc3339_to_epoch_days("2024-13-01"), None);
        assert_eq!(parse_rfc3339_to_epoch_days("2024-00-01"), None);
    }

    // ========================================================================
    // Ported from store-paths.test.ts
    // ========================================================================

    // ---- isAbsolutePath ----

    // NOTE: `is_absolute_path` reads the WSL env vars (via `is_wsl`) for any
    // leading-slash input, so these are `#[serial]` even though they only read
    // — env access races with the env-mutating `#[serial]` tests below, which
    // is UB under edition 2024 (the reason `set_var` is `unsafe`).
    #[test]
    #[serial]
    fn is_absolute_path_unix_absolute() {
        assert!(is_absolute_path("/path/to/file"));
        assert!(is_absolute_path("/"));
        assert!(is_absolute_path("/home/user/documents"));
        assert!(is_absolute_path("/usr/local/bin"));
    }

    #[test]
    fn is_absolute_path_unix_relative() {
        assert!(!is_absolute_path("path/to/file"));
        assert!(!is_absolute_path("./path/to/file"));
        assert!(!is_absolute_path("../path/to/file"));
        assert!(!is_absolute_path("./file"));
        assert!(!is_absolute_path("../file"));
        assert!(!is_absolute_path("file.txt"));
    }

    #[test]
    fn is_absolute_path_windows_forward_slash() {
        assert!(is_absolute_path("C:/path/to/file"));
        assert!(is_absolute_path("C:/"));
        assert!(is_absolute_path("D:/Users/Documents"));
        assert!(is_absolute_path("Z:/"));
        assert!(is_absolute_path("c:/lowercase"));
    }

    #[test]
    fn is_absolute_path_windows_backslash() {
        assert!(is_absolute_path("C:\\path\\to\\file"));
        assert!(is_absolute_path("C:\\"));
        assert!(is_absolute_path("D:\\Users\\Documents"));
        assert!(is_absolute_path("Z:\\"));
        assert!(is_absolute_path("c:\\lowercase"));
    }

    #[test]
    fn is_absolute_path_windows_relative() {
        assert!(!is_absolute_path("path\\to\\file"));
        assert!(!is_absolute_path(".\\path\\to\\file"));
        assert!(!is_absolute_path("..\\path\\to\\file"));
        assert!(!is_absolute_path(".\\file"));
        assert!(!is_absolute_path("..\\file"));
        assert!(!is_absolute_path("file.txt"));
    }

    #[test]
    #[serial]
    fn is_absolute_path_git_bash() {
        // Returns true for any leading-slash path regardless of WSL detection.
        assert!(is_absolute_path("/c/Users/name/file"));
        assert!(is_absolute_path("/C/Users/name/file"));
        assert!(is_absolute_path("/d/Projects"));
        assert!(is_absolute_path("/D/Projects"));
        assert!(is_absolute_path("/z/"));
    }

    #[test]
    #[serial]
    fn is_absolute_path_edge_cases() {
        assert!(!is_absolute_path(""));
        assert!(is_absolute_path("C:")); // drive letter only
        assert!(!is_absolute_path("C")); // just a letter
        assert!(!is_absolute_path(":"));
        assert!(is_absolute_path("/a")); // short Unix path
        assert!(is_absolute_path("/1/")); // number after slash (not Git Bash)
    }

    // ---- normalizePathSeparators ----

    #[test]
    fn normalize_path_separators_backslashes() {
        assert_eq!(
            normalize_path_separators("C:\\Users\\name\\file.txt"),
            "C:/Users/name/file.txt"
        );
        assert_eq!(
            normalize_path_separators("D:\\Projects\\qmd\\src"),
            "D:/Projects/qmd/src"
        );
        assert_eq!(normalize_path_separators("\\path\\to\\file"), "/path/to/file");
    }

    #[test]
    fn normalize_path_separators_mixed() {
        assert_eq!(
            normalize_path_separators("C:\\Users/name\\file.txt"),
            "C:/Users/name/file.txt"
        );
        assert_eq!(
            normalize_path_separators("path\\to/file/here"),
            "path/to/file/here"
        );
    }

    #[test]
    fn normalize_path_separators_unix_unchanged() {
        assert_eq!(normalize_path_separators("/path/to/file"), "/path/to/file");
        assert_eq!(normalize_path_separators("/usr/local/bin"), "/usr/local/bin");
        assert_eq!(normalize_path_separators("relative/path"), "relative/path");
    }

    #[test]
    fn normalize_path_separators_consecutive() {
        assert_eq!(normalize_path_separators("path\\\\to\\\\file"), "path//to//file");
        assert_eq!(normalize_path_separators("C:\\\\Users\\\\name"), "C://Users//name");
    }

    #[test]
    fn normalize_path_separators_edge_cases() {
        assert_eq!(normalize_path_separators(""), "");
        assert_eq!(normalize_path_separators("\\"), "/");
        assert_eq!(normalize_path_separators("\\\\"), "//");
        assert_eq!(normalize_path_separators("file.txt"), "file.txt");
    }

    // ---- getRelativePathFromPrefix ----

    #[test]
    fn get_relative_path_from_prefix_exact_match() {
        assert_eq!(get_relative_path_from_prefix("/home/user", "/home/user"), Some(String::new()));
        assert_eq!(get_relative_path_from_prefix("C:/Users/name", "C:/Users/name"), Some(String::new()));
        assert_eq!(get_relative_path_from_prefix("/path", "/path"), Some(String::new()));
    }

    #[test]
    fn get_relative_path_from_prefix_under_prefix() {
        assert_eq!(
            get_relative_path_from_prefix("/home/user/documents", "/home/user"),
            Some("documents".into())
        );
        assert_eq!(
            get_relative_path_from_prefix("/home/user/documents/file.txt", "/home/user"),
            Some("documents/file.txt".into())
        );
        assert_eq!(
            get_relative_path_from_prefix("C:/Users/name/Documents/file.txt", "C:/Users/name"),
            Some("Documents/file.txt".into())
        );
    }

    #[test]
    fn get_relative_path_from_prefix_not_under() {
        assert_eq!(get_relative_path_from_prefix("/home/other", "/home/user"), None);
        assert_eq!(get_relative_path_from_prefix("/usr/local", "/home/user"), None);
        assert_eq!(get_relative_path_from_prefix("C:/Users/other", "D:/Users"), None);
    }

    #[test]
    fn get_relative_path_from_prefix_windows_normalized() {
        assert_eq!(
            get_relative_path_from_prefix("C:\\Users\\name\\Documents", "C:\\Users\\name"),
            Some("Documents".into())
        );
        assert_eq!(
            get_relative_path_from_prefix("C:\\Users\\name\\Documents\\file.txt", "C:/Users/name"),
            Some("Documents/file.txt".into())
        );
    }

    #[test]
    fn get_relative_path_from_prefix_trailing_slash() {
        assert_eq!(
            get_relative_path_from_prefix("/home/user/documents", "/home/user/"),
            Some("documents".into())
        );
        assert_eq!(
            get_relative_path_from_prefix("C:/Users/name/Documents", "C:/Users/name/"),
            Some("Documents".into())
        );
    }

    #[test]
    fn get_relative_path_from_prefix_no_trailing_slash() {
        assert_eq!(
            get_relative_path_from_prefix("/home/user/documents", "/home/user"),
            Some("documents".into())
        );
        assert_eq!(
            get_relative_path_from_prefix("C:/Users/name/Documents", "C:/Users/name"),
            Some("Documents".into())
        );
    }

    #[test]
    fn get_relative_path_from_prefix_edge_cases() {
        // Empty prefix.
        assert_eq!(get_relative_path_from_prefix("/path/to/file", ""), None);
        // Substring but not in hierarchy.
        assert_eq!(get_relative_path_from_prefix("/home/username", "/home/user"), None);
        // Root prefix.
        assert_eq!(get_relative_path_from_prefix("/home/user", "/"), Some("home/user".into()));
    }

    // ---- resolve: Unix ----

    #[test]
    #[serial]
    fn resolve_unix_relative_paths() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["/base", "relative"]), "/base/relative");
        assert_eq!(resolve(&["/base", "a/b/c"]), "/base/a/b/c");
        assert_eq!(resolve(&["/home", "user/documents"]), "/home/user/documents");
    }

    #[test]
    #[serial]
    fn resolve_unix_absolute_paths() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["/base", "/absolute"]), "/absolute");
        assert_eq!(resolve(&["/home/user", "/usr/local"]), "/usr/local");
        assert_eq!(resolve(&["/any", "/"]), "/");
    }

    #[test]
    #[serial]
    fn resolve_unix_dot_and_dotdot() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["/base", "../other"]), "/other");
        assert_eq!(resolve(&["/base/sub", ".."]), "/base");
        assert_eq!(resolve(&["/base", "./file"]), "/base/file");
        assert_eq!(resolve(&["/base/a/b", "../../c"]), "/base/c");
    }

    #[test]
    #[serial]
    fn resolve_unix_multiple_segments() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["/a", "b", "c"]), "/a/b/c");
        assert_eq!(resolve(&["/a", "b", "../c"]), "/a/c");
        assert_eq!(resolve(&["/a", "b", "/c"]), "/c");
    }

    #[test]
    #[serial]
    fn resolve_unix_relative_without_base_uses_pwd() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["relative"]), "/home/user/relative");
        assert_eq!(resolve(&["a/b/c"]), "/home/user/a/b/c");
        assert_eq!(resolve(&["./file"]), "/home/user/file");
    }

    #[test]
    #[serial]
    fn resolve_unix_absolute_alone() {
        let _g = EnvGuard::with_pwd_no_wsl("/home/user");
        assert_eq!(resolve(&["/absolute/path"]), "/absolute/path");
        assert_eq!(resolve(&["/"]), "/");
    }

    // ---- resolve: Windows ----

    #[test]
    #[serial]
    fn resolve_windows_relative_paths() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:/base", "relative"]), "C:/base/relative");
        assert_eq!(resolve(&["C:/base", "a/b/c"]), "C:/base/a/b/c");
        assert_eq!(resolve(&["D:/Projects", "qmd/src"]), "D:/Projects/qmd/src");
    }

    #[test]
    #[serial]
    fn resolve_windows_absolute_paths() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:/base", "D:/other"]), "D:/other");
        assert_eq!(resolve(&["C:/Users", "C:/Program Files"]), "C:/Program Files");
        assert_eq!(resolve(&["D:/any", "E:/other"]), "E:/other");
    }

    #[test]
    #[serial]
    fn resolve_windows_backslashes() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:\\base", "relative"]), "C:/base/relative");
        assert_eq!(resolve(&["C:\\Users\\name", "Documents"]), "C:/Users/name/Documents");
        assert_eq!(resolve(&["C:\\base", "a\\b\\c"]), "C:/base/a/b/c");
    }

    #[test]
    #[serial]
    fn resolve_windows_dot_and_dotdot() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:/base", "../other"]), "C:/other");
        assert_eq!(resolve(&["C:/base/sub", ".."]), "C:/base");
        assert_eq!(resolve(&["C:/base", "./file"]), "C:/base/file");
        assert_eq!(resolve(&["C:/base/a/b", "../../c"]), "C:/base/c");
    }

    #[test]
    #[serial]
    fn resolve_windows_multiple_segments() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:/a", "b", "c"]), "C:/a/b/c");
        assert_eq!(resolve(&["C:/a", "b", "../c"]), "C:/a/c");
        assert_eq!(resolve(&["C:/a", "b", "D:/c"]), "D:/c");
    }

    #[test]
    #[serial]
    fn resolve_windows_relative_without_base_uses_pwd() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["relative"]), "C:/Users/name/relative");
        assert_eq!(resolve(&["a/b/c"]), "C:/Users/name/a/b/c");
        assert_eq!(resolve(&[".\\file"]), "C:/Users/name/file");
    }

    #[test]
    #[serial]
    fn resolve_windows_drive_letter_only() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["C:"]), "C:/");
        assert_eq!(resolve(&["D:"]), "D:/");
    }

    // ---- resolve: Git Bash ----

    #[test]
    #[serial]
    fn resolve_git_bash_to_windows() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/c/Users/name"]), "C:/Users/name");
        assert_eq!(resolve(&["/C/Users/name"]), "C:/Users/name");
        assert_eq!(resolve(&["/d/Projects"]), "D:/Projects");
        assert_eq!(resolve(&["/D/Projects"]), "D:/Projects");
    }

    #[test]
    #[serial]
    fn resolve_git_bash_relative() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/c/base", "relative"]), "C:/base/relative");
        assert_eq!(resolve(&["/d/Projects", "qmd/src"]), "D:/Projects/qmd/src");
    }

    #[test]
    #[serial]
    fn resolve_git_bash_dot_and_dotdot() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/c/base", "../other"]), "C:/other");
        assert_eq!(resolve(&["/c/base/sub", ".."]), "C:/base");
        assert_eq!(resolve(&["/c/base", "./file"]), "C:/base/file");
    }

    #[test]
    #[serial]
    fn resolve_git_bash_multiple_segments() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/c/a", "b", "c"]), "C:/a/b/c");
        assert_eq!(resolve(&["/c/a", "b", "/d/c"]), "D:/c");
    }

    // ---- resolve: edge cases ----

    // The `resolve_edge_*` cases use absolute first args (no PWD lookup) but
    // still reach `is_wsl` via `is_absolute_path`/`take_git_bash_drive`, so
    // they must be `#[serial]` alongside the env-mutating tests. Their outputs
    // are env-independent, hence no `EnvGuard` is needed.
    #[test]
    #[serial]
    fn resolve_edge_empty_segments_filtered() {
        assert_eq!(resolve(&["/base", "", "file"]), "/base/file");
        assert_eq!(resolve(&["C:/base", "", "file"]), "C:/base/file");
    }

    #[test]
    #[serial]
    fn resolve_edge_multiple_consecutive_slashes() {
        assert_eq!(resolve(&["/base//path///file"]), "/base/path/file");
        assert_eq!(resolve(&["C:/base//path///file"]), "C:/base/path/file");
    }

    #[test]
    #[serial]
    fn resolve_edge_trailing_slashes() {
        assert_eq!(resolve(&["/base/", "file"]), "/base/file");
        assert_eq!(resolve(&["C:/base/", "file"]), "C:/base/file");
    }

    #[test]
    #[serial]
    fn resolve_edge_complex_dotdot() {
        assert_eq!(resolve(&["/a/b/c/d", "../../../e"]), "/a/e");
        assert_eq!(resolve(&["C:/a/b/c/d", "../../../e"]), "C:/a/e");
    }

    #[test]
    #[serial]
    fn resolve_edge_too_many_dotdot() {
        assert_eq!(resolve(&["/base", "../../../../other"]), "/other");
        assert_eq!(resolve(&["C:/base", "../../../../other"]), "C:/other");
    }

    #[test]
    #[serial]
    fn resolve_edge_mixed_unix_and_windows() {
        let _g = EnvGuard::with_pwd_no_wsl("C:/Users/name");
        assert_eq!(resolve(&["/unix/path"]), "/unix/path");
        assert_eq!(resolve(&["relative"]), "C:/Users/name/relative");
    }

    #[test]
    #[should_panic(expected = "at least one path segment is required")]
    fn resolve_edge_no_arguments_panics() {
        let _ = resolve(&[]);
    }

    // ========================================================================
    // Ported from store.helpers.unit.test.ts — Path Utilities
    // ========================================================================

    #[test]
    #[serial]
    fn homedir_returns_home_env() {
        let g = EnvGuard::new();
        g.set("HOME", "/home/testuser");
        assert_eq!(homedir(), PathBuf::from("/home/testuser"));
    }

    #[test]
    #[serial]
    fn resolve_handles_absolute_paths() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/foo/bar"]), "/foo/bar");
        assert_eq!(resolve(&["/foo", "/bar"]), "/bar");
    }

    #[test]
    #[serial]
    fn resolve_handles_relative_paths() {
        let _g = EnvGuard::with_pwd_no_wsl("/work");
        assert_eq!(resolve(&["foo"]), "/work/foo");
        assert_eq!(resolve(&["foo", "bar"]), "/work/foo/bar");
    }

    #[test]
    #[serial]
    fn resolve_normalizes_dot_and_dotdot() {
        let _g = EnvGuard::no_wsl();
        assert_eq!(resolve(&["/foo/bar/./baz"]), "/foo/bar/baz");
        assert_eq!(resolve(&["/foo/bar/../baz"]), "/foo/baz");
        assert_eq!(resolve(&["/foo/bar/../../baz"]), "/baz");
    }

    #[test]
    #[serial]
    fn default_db_path_errs_in_test_mode_without_index_path() {
        let g = EnvGuard::new();
        g.unset("RQMD_INDEX_PATH");
        // Reset production mode in case another test enabled it (the flag is
        // a process-global AtomicBool).
        _reset_production_mode_for_testing();
        assert!(matches!(default_db_path(None), Err(Error::DbPathNotSet)));
    }

    #[test]
    #[serial]
    fn default_db_path_uses_index_path_when_set() {
        let g = EnvGuard::new();
        g.set("RQMD_INDEX_PATH", "/tmp/test-index.sqlite");
        // RQMD_INDEX_PATH takes precedence regardless of index name / mode.
        assert_eq!(default_db_path(None).unwrap(), PathBuf::from("/tmp/test-index.sqlite"));
        assert_eq!(
            default_db_path(Some("custom")).unwrap(),
            PathBuf::from("/tmp/test-index.sqlite")
        );
    }

    #[test]
    #[serial]
    fn pwd_returns_current_working_directory() {
        let g = EnvGuard::new();
        g.set("PWD", "/some/working/dir");
        let p = pwd();
        assert!(!p.as_os_str().is_empty());
        assert_eq!(p, PathBuf::from("/some/working/dir"));
    }

    // `std::env::temp_dir()` reads `TMPDIR`/`TEMP`/`TMP`; an env *read* still
    // races with the env-mutating `#[serial]` tests (the `environ` array is
    // shared), so this is serial too.
    #[test]
    #[serial]
    fn real_path_resolves_existing_directory() {
        let tmp = std::env::temp_dir();
        let resolved = real_path(&tmp);
        let expected = std::fs::canonicalize(&tmp).unwrap_or_else(|_| tmp.clone());
        assert_eq!(resolved, expected);
        assert!(resolved.exists());
    }
}
