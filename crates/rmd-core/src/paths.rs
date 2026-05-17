//! Path and environment-variable resolution for rmd's config layout.
//!
//! Mirrors the helpers in `tobi/qmd`'s `src/paths.ts` and the `getConfigDir` /
//! `getConfigFilePath` helpers in `src/collections.ts`, renamed to the `rmd`
//! namespace (`~/.config/rmd/`, `RMD_CONFIG_DIR`).

use std::path::PathBuf;

/// Return the user's home directory, mirroring TS `qmdHomedir()` precedence.
///
/// Order: `HOME` → `USERPROFILE` → `std::env::home_dir()` → `/tmp`.
/// On Windows, the `HOME` fallback matches Git-Bash / MSYS conventions.
pub fn rmd_homedir() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME")
        && !h.is_empty()
    {
        return h.into();
    }
    if let Some(h) = std::env::var_os("USERPROFILE")
        && !h.is_empty()
    {
        return h.into();
    }
    // `std::env::home_dir` was un-deprecated in Rust 1.86 with a corrected impl.
    #[allow(deprecated)]
    std::env::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"))
}

/// Resolve the directory that holds rmd's YAML config file(s).
///
/// Precedence: `RMD_CONFIG_DIR` > `XDG_CONFIG_HOME/rmd` > `~/.config/rmd`.
/// All checks are performed per-call so that test harnesses can mutate env
/// vars between operations.
pub fn config_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("RMD_CONFIG_DIR")
        && !d.is_empty()
    {
        return d.into();
    }
    if let Some(d) = std::env::var_os("XDG_CONFIG_HOME")
        && !d.is_empty()
    {
        return PathBuf::from(d).join("rmd");
    }
    rmd_homedir().join(".config").join("rmd")
}

/// Full path to `<config_dir>/<index_name>.yml`.
pub fn config_file_path(index_name: &str) -> PathBuf {
    config_dir().join(format!("{index_name}.yml"))
}
