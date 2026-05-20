//! `qmd://` virtual paths.
//!
//! Port of the virtual-path utilities in `tobi/qmd`'s `src/store.ts`
//! (lines 569–724).

use std::path::{Path, PathBuf};

use rusqlite::Connection;

use super::{Error, Result};

/// A parsed `qmd://[index@]collection/path` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualPath {
    pub collection: String,
    pub path: String,
    pub index_name: Option<String>,
}

const SCHEME_QMD: &str = "qmd://";

/// Normalise `collection://path`, `qmd://collection/path`, and bare
/// `collection/path` inputs to the canonical `collection://path` shape.
pub fn normalize_virtual_path(input: &str) -> Result<String> {
    let vp = parse_virtual_path(input)?;
    Ok(format!("{}://{}", vp.collection, vp.path))
}

/// Parse a virtual path. Accepts:
///
/// * `qmd://[index@]collection/path`
/// * `collection://path`
pub fn parse_virtual_path(input: &str) -> Result<VirtualPath> {
    if let Some(rest) = input.strip_prefix(SCHEME_QMD) {
        let (index_name, body) = if let Some((idx, rest)) = rest.split_once('@') {
            (Some(idx.to_string()), rest)
        } else {
            (None, rest)
        };
        let (collection, path) = body
            .split_once('/')
            .ok_or_else(|| Error::InvalidVirtualPath(input.to_string()))?;
        return Ok(VirtualPath {
            collection: collection.to_string(),
            path: path.to_string(),
            index_name,
        });
    }

    if let Some((collection, path)) = input.split_once("://") {
        return Ok(VirtualPath {
            collection: collection.to_string(),
            path: path.to_string(),
            index_name: None,
        });
    }

    Err(Error::InvalidVirtualPath(input.to_string()))
}

/// Build a `qmd://[index@]collection/path` URI.
pub fn build_virtual_path(collection: &str, path: &str, index_name: Option<&str>) -> String {
    match index_name {
        Some(idx) => format!("{}{idx}@{collection}/{path}", SCHEME_QMD),
        None => format!("{}{collection}/{path}", SCHEME_QMD),
    }
}

/// Quick membership check — does `s` look like a virtual path?
pub fn is_virtual_path(s: &str) -> bool {
    s.starts_with(SCHEME_QMD) || s.contains("://")
}

/// Resolve a virtual path to its filesystem location by looking up the
/// owning collection's `path` and joining the relative portion.
pub fn resolve_virtual_path(conn: &Connection, virtual_path: &str) -> Result<Option<PathBuf>> {
    let vp = parse_virtual_path(virtual_path)?;
    let row: Option<String> = conn
        .query_row(
            "SELECT path FROM store_collections WHERE name = ?",
            rusqlite::params![vp.collection],
            |row| row.get(0),
        )
        .ok();
    Ok(row.map(|base| PathBuf::from(base).join(&vp.path)))
}

/// Inverse of [`resolve_virtual_path`]: find the collection whose root is
/// a prefix of `abs` and return `qmd://collection/<relative>`.
pub fn to_virtual_path(conn: &Connection, abs: &Path) -> Result<Option<String>> {
    let abs_str = abs.to_string_lossy().replace('\\', "/");

    let mut stmt = conn.prepare("SELECT name, path FROM store_collections")?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Longest-prefix wins.
    let mut best: Option<(String, String)> = None;
    for (name, base) in rows {
        let base_norm = base.replace('\\', "/");
        let with_slash = if base_norm.ends_with('/') {
            base_norm.clone()
        } else {
            format!("{}/", base_norm)
        };
        if abs_str == base_norm || abs_str.starts_with(&with_slash) {
            let candidate = best
                .as_ref()
                .map(|(_, b)| base_norm.len() > b.len())
                .unwrap_or(true);
            if candidate {
                let rel = if abs_str == base_norm {
                    String::new()
                } else {
                    abs_str[with_slash.len()..].to_string()
                };
                best = Some((build_virtual_path(&name, &rel, None), base_norm));
            }
        }
    }

    Ok(best.map(|(s, _)| s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_qmd_scheme() {
        let vp = parse_virtual_path("qmd://docs/readme.md").unwrap();
        assert_eq!(vp.collection, "docs");
        assert_eq!(vp.path, "readme.md");
        assert_eq!(vp.index_name, None);
    }

    #[test]
    fn parse_qmd_with_index() {
        let vp = parse_virtual_path("qmd://myindex@docs/readme.md").unwrap();
        assert_eq!(vp.index_name.as_deref(), Some("myindex"));
        assert_eq!(vp.collection, "docs");
        assert_eq!(vp.path, "readme.md");
    }

    #[test]
    fn parse_collection_scheme() {
        let vp = parse_virtual_path("docs://readme.md").unwrap();
        assert_eq!(vp.collection, "docs");
        assert_eq!(vp.path, "readme.md");
    }

    #[test]
    fn parse_invalid() {
        assert!(parse_virtual_path("hello").is_err());
    }

    #[test]
    fn build_round_trip() {
        let built = build_virtual_path("docs", "readme.md", None);
        let parsed = parse_virtual_path(&built).unwrap();
        assert_eq!(parsed.collection, "docs");
        assert_eq!(parsed.path, "readme.md");
    }

    #[test]
    fn is_virtual_detects_scheme() {
        assert!(is_virtual_path("qmd://a/b"));
        assert!(is_virtual_path("docs://a"));
        assert!(!is_virtual_path("/abs/path"));
    }
}
