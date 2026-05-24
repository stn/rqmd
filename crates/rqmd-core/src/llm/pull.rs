//! HuggingFace model download and GGUF validation.
//!
//! Mirrors `tobi/qmd/src/llm.ts` lines 298–435.
//!
//! Two intentional departures from the TS version:
//!
//! * The TS cache layout was flat under `~/.cache/qmd/models/` with
//!   per-file `<filename>.etag` sidecars and fuzzy-name lookups
//!   (`entry.name.includes(filename)`). The Rust port uses `hf-hub`'s
//!   native SHA-keyed snapshot layout under the same cache root, which
//!   is more robust but not on-disk-compatible with old qmd installs.
//!   `pull_models(refresh=true)` invalidates the snapshot by removing
//!   the per-repo directory.
//! * This module is sync. `hf-hub::api::sync` uses `ureq` under the
//!   hood. PR2's async `Llm` implementation wraps `pull_models` in
//!   `tokio::task::spawn_blocking`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Cache, Repo, RepoType};

use crate::llm::config::default_model_cache_dir;
use crate::llm::error::{Error, Result};
use crate::llm::types::{PullOptions, PullResult};

/// Parsed `hf:<org>/<repo>/<file>` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    /// `<org>/<repo>`
    pub repo: String,
    /// File path within the repo (may contain `/`).
    pub file: String,
}

/// Parse an `hf:org/repo/path/to/file.gguf` URI. Returns `None` for
/// anything else (caller may treat it as a local filesystem path).
pub fn parse_hf_uri(model: &str) -> Option<HfRef> {
    let without = model.strip_prefix("hf:")?;
    let parts: Vec<&str> = without.split('/').collect();
    if parts.len() < 3 {
        return None;
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let file = parts[2..].join("/");
    if repo.is_empty() || file.is_empty() {
        return None;
    }
    Some(HfRef { repo, file })
}

/// GGUF magic bytes (first 4 bytes of a valid GGUF file).
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Validate that `path` is a real GGUF file. On any failure the file is
/// `unlinkSync`-equivalent removed (mirrors TS so the next `pull_models`
/// call re-downloads cleanly) and a descriptive [`Error::InvalidGguf`]
/// is returned.
pub fn validate_gguf_file(path: &Path, model_uri: &str) -> Result<()> {
    if !path.exists() {
        return Ok(()); // downstream `pull` failure path handles "missing"
    }

    // Read first 512 bytes for magic + HTML sniff.
    let mut sniff = [0u8; 512];
    let n = {
        let mut f = fs::File::open(path).map_err(|e| Error::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let mut total = 0;
        loop {
            match f.read(&mut sniff[total..]) {
                Ok(0) => break,
                Ok(read) => {
                    total += read;
                    if total == sniff.len() {
                        break;
                    }
                }
                Err(e) => {
                    return Err(Error::Io {
                        path: path.to_owned(),
                        source: e,
                    });
                }
            }
        }
        total
    };

    let header = &sniff[..n.min(4)];
    if header == GGUF_MAGIC {
        return Ok(());
    }

    // Not GGUF. Build a useful error before deleting.
    let text = String::from_utf8_lossy(&sniff[..n]).to_lowercase();
    let looks_like_html = text.contains("<!doctype") || text.contains("<html");
    let got = String::from_utf8_lossy(header).into_owned();
    let size_kb = fs::metadata(path).map(|m| m.len() / 1024).unwrap_or(0);

    // Remove the bad file so the next pull re-downloads cleanly.
    let _ = fs::remove_file(path);

    let reason = if looks_like_html {
        format!(
            "downloaded file is an HTML page, not a GGUF model ({size_kb} KB). \
             Something is intercepting the download from huggingface.co (a proxy, \
             firewall, or captive portal). Model URI: {model_uri}. To work around: \
             set HF_ENDPOINT to a mirror, or set the model env var to a local path."
        )
    } else {
        format!(
            "expected GGUF magic \"GGUF\", got \"{got}\" ({size_kb} KB). \
             The file has been removed; retry to re-download. Model URI: {model_uri}."
        )
    };

    Err(Error::InvalidGguf {
        path: path.to_owned(),
        reason,
        looks_like_html,
    })
}

/// Download one or more models from HuggingFace (or resolve to an
/// existing local file when the URI is a plain path). Returns one
/// [`PullResult`] per input, in the same order.
///
/// Synchronous: callers in async contexts should wrap in
/// `tokio::task::spawn_blocking`.
pub fn pull_models(models: &[String], options: &PullOptions) -> Result<Vec<PullResult>> {
    let cache_dir = options
        .cache_dir
        .clone()
        .or_else(default_model_cache_dir)
        .ok_or_else(|| Error::Io {
            path: PathBuf::from("<model-cache-dir>"),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "could not determine a model cache directory",
            ),
        })?;
    fs::create_dir_all(&cache_dir).map_err(|e| Error::Io {
        path: cache_dir.clone(),
        source: e,
    })?;

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.clone())
        .with_progress(true)
        .build()
        .map_err(|e| Error::HfApi(e.to_string()))?;

    let mut results = Vec::with_capacity(models.len());
    for model_uri in models {
        results.push(pull_one(&api, &cache_dir, model_uri, options.refresh)?);
    }
    Ok(results)
}

fn pull_one(api: &Api, cache_dir: &Path, model_uri: &str, refresh: bool) -> Result<PullResult> {
    let (path, refreshed) = match parse_hf_uri(model_uri) {
        Some(hf_ref) => {
            let repo = Repo::new(hf_ref.repo.clone(), RepoType::Model);
            let repo_dir = cache_dir.join(repo.folder_name());

            // Only count as "refreshed" if there was an actual cached
            // snapshot to invalidate. An empty `models--org--repo` dir
            // from a previous failed download (or just `mkdir -p`)
            // shouldn't be reported as a stale-cache invalidation;
            // matches TS `refreshed = cached.length > 0`.
            let mut did_refresh = false;
            if refresh && repo_dir.exists() {
                let had_snapshots = repo_dir
                    .join("snapshots")
                    .read_dir()
                    .map(|mut entries| entries.next().is_some())
                    .unwrap_or(false);
                fs::remove_dir_all(&repo_dir).map_err(|e| Error::Io {
                    path: repo_dir.clone(),
                    source: e,
                })?;
                did_refresh = had_snapshots;
            }

            let path = api
                .model(hf_ref.repo.clone())
                .get(&hf_ref.file)
                .map_err(|e| Error::HfApi(format!("{model_uri}: {e}")))?;
            (path, did_refresh)
        }
        None => {
            // Treat as local filesystem path.
            let path = PathBuf::from(model_uri);
            if !path.exists() {
                return Err(Error::ModelNotFound(model_uri.to_owned()));
            }
            (path, false)
        }
    };

    validate_gguf_file(&path, model_uri)?;

    let size_bytes = fs::metadata(&path)
        .map_err(|e| Error::Io {
            path: path.clone(),
            source: e,
        })?
        .len();

    Ok(PullResult {
        model: model_uri.to_owned(),
        path,
        size_bytes,
        refreshed,
    })
}

// ============================================================================
// Non-mutating GGUF inspection (doctor diagnostics)
// ============================================================================
//
// Port of `tobi/qmd/src/llm.ts::inspectGgufFile` (origin/main). Unlike
// [`validate_gguf_file`] (which deletes a bad file and returns `Err` so the
// next `pull` re-downloads), these helpers **never mutate the filesystem** —
// `doctor` reports on all three active models and must not bail or delete.

/// What kind of file `inspect_gguf_file` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufKind {
    /// File does not exist.
    Missing,
    /// Valid GGUF (magic bytes match).
    Gguf,
    /// An HTML page (proxy / firewall / captive-portal response), not a model.
    Html,
    /// Present but not a valid GGUF (wrong magic, or unreadable).
    Invalid,
}

/// Result of inspecting a candidate GGUF file without mutating it.
#[derive(Debug, Clone)]
pub struct GgufInspection {
    pub kind: GgufKind,
    pub size_bytes: u64,
    /// Printable first-4-byte magic (`GGUF`, `<html`, …) or `0x<hex>`.
    pub magic: String,
    /// Human-readable detail mirroring qmd's wording.
    pub details: String,
}

impl GgufInspection {
    /// True only for a real GGUF file.
    pub fn valid(&self) -> bool {
        matches!(self.kind, GgufKind::Gguf)
    }

    /// True when the file exists on disk (any kind except `Missing`).
    pub fn exists(&self) -> bool {
        !matches!(self.kind, GgufKind::Missing)
    }
}

/// qmd `formatModelFileSize`: `${(bytes/1024).toFixed(0)} KB` (rounded).
fn format_model_file_size(size_bytes: u64) -> String {
    format!("{} KB", ((size_bytes as f64) / 1024.0).round() as u64)
}

/// qmd `printableMagic`: the 4 header bytes as text when all printable ASCII,
/// otherwise `0x<hex>`.
fn printable_magic(header: &[u8]) -> String {
    let printable = !header.is_empty()
        && header.len() <= 4
        && header.iter().all(|&b| (0x20..=0x7e).contains(&b));
    if printable {
        String::from_utf8_lossy(header).into_owned()
    } else {
        let hex: String = header.iter().map(|b| format!("{b:02x}")).collect();
        format!("0x{hex}")
    }
}

/// Inspect a potential GGUF model file **without mutating it**. Port of
/// qmd `inspectGgufFile` — used by `doctor`'s `model cache` check.
pub fn inspect_gguf_file(path: &Path) -> GgufInspection {
    if !path.exists() {
        return GgufInspection {
            kind: GgufKind::Missing,
            size_bytes: 0,
            magic: String::new(),
            details: "file does not exist".to_owned(),
        };
    }

    let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // Read up to the first 512 bytes (zero-padded) for magic + HTML sniff.
    let mut sniff = [0u8; 512];
    let n = {
        let mut f = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                return GgufInspection {
                    kind: GgufKind::Invalid,
                    size_bytes,
                    magic: String::new(),
                    details: format!("cannot read model file: {e}"),
                };
            }
        };
        let mut total = 0;
        loop {
            match f.read(&mut sniff[total..]) {
                Ok(0) => break,
                Ok(read) => {
                    total += read;
                    if total == sniff.len() {
                        break;
                    }
                }
                Err(e) => {
                    return GgufInspection {
                        kind: GgufKind::Invalid,
                        size_bytes,
                        magic: String::new(),
                        details: format!("cannot read model file: {e}"),
                    };
                }
            }
        }
        total
    };

    // qmd compares the (zero-padded) first 4 bytes.
    let header = &sniff[..4];
    if header == GGUF_MAGIC {
        return GgufInspection {
            kind: GgufKind::Gguf,
            size_bytes,
            magic: "GGUF".to_owned(),
            details: format!("valid GGUF ({})", format_model_file_size(size_bytes)),
        };
    }

    let magic = printable_magic(header);
    let text = String::from_utf8_lossy(&sniff[..n]).to_lowercase();
    if text.contains("<!doctype") || text.contains("<html") {
        return GgufInspection {
            kind: GgufKind::Html,
            size_bytes,
            magic,
            details: format!(
                "HTML page, not a GGUF model ({}); likely proxy/firewall/captive portal response",
                format_model_file_size(size_bytes)
            ),
        };
    }

    let details = format!(
        "not valid GGUF (expected magic \"GGUF\", got \"{magic}\", {})",
        format_model_file_size(size_bytes)
    );
    GgufInspection {
        kind: GgufKind::Invalid,
        size_bytes,
        magic,
        details,
    }
}

/// Result of [`inspect_cached_model`]: the cached path (when a valid GGUF is
/// present) plus any `<path>: <detail>` strings for cached-but-invalid files.
#[derive(Debug, Clone, Default)]
pub struct CachedModelInspection {
    pub path: Option<PathBuf>,
    pub invalid: Vec<String>,
}

/// Resolve a model URI to its cached file **without downloading** and inspect
/// it. Port of qmd `findCachedModelInspection`, adapted to rqmd's hf-hub
/// SHA-keyed snapshot cache layout (vs qmd's flat-dir fuzzy match): `hf:` URIs
/// are resolved via [`Cache::get`] (returns `None` when not cached, never
/// fetches); plain paths are inspected directly.
pub fn inspect_cached_model(model: &str) -> CachedModelInspection {
    let mut invalid = Vec::new();

    if let Some(hf) = parse_hf_uri(model) {
        let Some(cache_dir) = default_model_cache_dir() else {
            return CachedModelInspection {
                path: None,
                invalid,
            };
        };
        let cache = Cache::new(cache_dir);
        match cache.model(hf.repo).get(&hf.file) {
            Some(path) => {
                let inspection = inspect_gguf_file(&path);
                if inspection.valid() {
                    CachedModelInspection {
                        path: Some(path),
                        invalid,
                    }
                } else {
                    invalid.push(format!("{}: {}", path.display(), inspection.details));
                    CachedModelInspection {
                        path: None,
                        invalid,
                    }
                }
            }
            None => CachedModelInspection {
                path: None,
                invalid,
            },
        }
    } else {
        // Plain filesystem path.
        let path = PathBuf::from(model);
        let inspection = inspect_gguf_file(&path);
        if inspection.valid() {
            CachedModelInspection {
                path: Some(path),
                invalid,
            }
        } else {
            if inspection.exists() {
                invalid.push(format!("{}: {}", path.display(), inspection.details));
            }
            CachedModelInspection {
                path: None,
                invalid,
            }
        }
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(bytes: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn inspect_gguf_detects_valid_gguf() {
        let f = write_temp(b"GGUF\x00\x01\x02\x03 model data");
        let insp = inspect_gguf_file(f.path());
        assert_eq!(insp.kind, GgufKind::Gguf);
        assert!(insp.valid());
        assert_eq!(insp.magic, "GGUF");
        assert!(insp.details.contains("valid GGUF"));
        // Non-destructive: the file must still exist.
        assert!(f.path().exists());
    }

    #[test]
    fn inspect_gguf_detects_html() {
        let f = write_temp(b"<!DOCTYPE html><html><body>blocked</body></html>");
        let insp = inspect_gguf_file(f.path());
        assert_eq!(insp.kind, GgufKind::Html);
        assert!(!insp.valid());
        assert!(insp.details.contains("HTML page, not a GGUF model"));
        assert!(f.path().exists(), "inspection must not delete the file");
    }

    #[test]
    fn inspect_gguf_detects_invalid_magic() {
        let f = write_temp(b"ABCD not a model file");
        let insp = inspect_gguf_file(f.path());
        assert_eq!(insp.kind, GgufKind::Invalid);
        assert_eq!(insp.magic, "ABCD");
        assert!(insp.details.contains("not valid GGUF"));
        assert!(f.path().exists());
    }

    #[test]
    fn inspect_gguf_missing_file() {
        let insp = inspect_gguf_file(Path::new("/no/such/model.gguf"));
        assert_eq!(insp.kind, GgufKind::Missing);
        assert!(!insp.exists());
        assert_eq!(insp.details, "file does not exist");
    }

    #[test]
    fn inspect_cached_model_local_invalid_path_reports_detail() {
        let f = write_temp(b"<!doctype html>nope");
        let path = f.path().to_string_lossy().to_string();
        let result = inspect_cached_model(&path);
        assert!(result.path.is_none());
        assert_eq!(result.invalid.len(), 1);
        assert!(result.invalid[0].contains("HTML page, not a GGUF model"));
    }

    #[test]
    fn inspect_cached_model_local_valid_path() {
        let f = write_temp(b"GGUF\x00\x00\x00\x00");
        let path = f.path().to_string_lossy().to_string();
        let result = inspect_cached_model(&path);
        assert!(result.path.is_some());
        assert!(result.invalid.is_empty());
    }
}
