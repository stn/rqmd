//! `qmd://` virtual paths.
//!
//! Port of the virtual-path utilities in `tobi/qmd`'s `src/store.ts`
//! (lines 586–657). Mirrors the TS string semantics:
//!
//! * canonical form `qmd://collection/path`,
//! * `//collection/path` shorthand (the `qmd:` prefix omitted),
//! * an optional `?index=` query parameter selecting an alternate index,
//! * redundant leading slashes are collapsed (`qmd:////x` → `qmd://x`),
//! * bare `collection/path`, absolute paths, `~/...`, and docids are NOT
//!   virtual paths and pass through [`normalize_virtual_path`] unchanged.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

use super::{Error, Result};

/// A parsed `qmd://collection/path[?index=...]` URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualPath {
    pub collection: String,
    pub path: String,
    pub index_name: Option<String>,
}

const SCHEME_QMD: &str = "qmd://";

/// Normalise a virtual path to the canonical `qmd://collection/path` shape.
///
/// Trims, expands the `//collection` shorthand and `qmd:` prefix to
/// `qmd://`, and collapses redundant leading slashes. Inputs that are not
/// virtual paths (bare `collection/path`, absolute paths, `~/...`, docids)
/// are returned trimmed but otherwise unchanged.
pub fn normalize_virtual_path(input: &str) -> String {
    let path = input.trim();

    // `qmd:` (with any number of trailing slashes) → `qmd://`.
    if let Some(rest) = path.strip_prefix("qmd:") {
        return format!("{}{}", SCHEME_QMD, rest.trim_start_matches('/'));
    }

    // `//collection/path` shorthand (missing `qmd:` prefix).
    if let Some(rest) = path.strip_prefix("//") {
        return format!("{}{}", SCHEME_QMD, rest.trim_start_matches('/'));
    }

    // Bare/absolute/`~`/docid: not a virtual path, leave as-is.
    path.to_string()
}

/// Parse a virtual path. Accepts `qmd://collection/path`, the
/// `//collection/path` shorthand, and an optional `?index=...` query.
/// Returns [`Error::InvalidVirtualPath`] for non-virtual inputs.
pub fn parse_virtual_path(input: &str) -> Result<VirtualPath> {
    let normalized = normalize_virtual_path(input);

    // Mirror TS `normalized.split("?")` destructured as `[pathPart, query]`:
    // the path is everything before the first `?`, the query is the segment
    // between the first and second `?` (anything after a second `?` is dropped).
    let mut segments = normalized.split('?');
    let path_part = segments.next().unwrap_or("");
    let query = segments.next().unwrap_or("");

    let rest = path_part
        .strip_prefix(SCHEME_QMD)
        .ok_or_else(|| Error::InvalidVirtualPath(input.to_string()))?;

    let (collection, path) = rest.split_once('/').unwrap_or((rest, ""));
    if collection.is_empty() {
        return Err(Error::InvalidVirtualPath(input.to_string()));
    }

    // `new URLSearchParams(query).get("index")?.trim() || undefined`.
    let index_name = form_urlencoded::parse(query.as_bytes())
        .find(|(k, _)| k == "index")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty());

    Ok(VirtualPath {
        collection: collection.to_string(),
        path: path.to_string(),
        index_name,
    })
}

/// Build a `qmd://collection/path[?index=...]` URI.
pub fn build_virtual_path(collection: &str, path: &str, index_name: Option<&str>) -> String {
    let base = format!("{SCHEME_QMD}{collection}/{path}");
    match index_name {
        Some(idx) => format!("{base}?index={}", encode_uri_component(idx)),
        None => base,
    }
}

/// Mirror of JS `encodeURIComponent`: percent-encode every byte except the
/// unreserved set `A-Za-z0-9-_.!~*'()` (uppercase hex, like the JS builtin).
fn encode_uri_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Encode a path for use in `qmd://` URIs, mirroring qmd's `encodeQmdPath`
/// (`src/mcp/server.ts:70-73`): percent-encode each `/`-delimited segment with
/// `encodeURIComponent`, preserving the slashes for readability. Used by the MCP
/// server when building resource URIs for `get` / `multi_get`.
pub fn encode_qmd_path(path: &str) -> String {
    path.split('/')
        .map(encode_uri_component)
        .collect::<Vec<_>>()
        .join("/")
}

/// Quick membership check — does `s` look like a virtual path?
pub fn is_virtual_path(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("qmd:") || t.starts_with("//")
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

    // ---- encode_qmd_path (port of mcp.test.ts:818-825) ----

    #[test]
    fn encode_qmd_path_preserves_slashes_encodes_special_chars() {
        assert_eq!(
            encode_qmd_path("External Podcast/2023 April - Interview.md"),
            "External%20Podcast/2023%20April%20-%20Interview.md"
        );
        // Plain path is unchanged.
        assert_eq!(encode_qmd_path("docs/readme.md"), "docs/readme.md");
        // No slash → single segment encoded.
        assert_eq!(encode_qmd_path("a b.md"), "a%20b.md");
    }

    // ---- normalize_virtual_path (port of store.test.ts:3590) ----

    #[test]
    fn normalize_passes_through_canonical_qmd() {
        assert_eq!(
            normalize_virtual_path("qmd://collection/path.md"),
            "qmd://collection/path.md"
        );
        assert_eq!(
            normalize_virtual_path("qmd://journals/2025-01-01.md"),
            "qmd://journals/2025-01-01.md"
        );
    }

    #[test]
    fn normalize_expands_double_slash_shorthand() {
        assert_eq!(
            normalize_virtual_path("//collection/path.md"),
            "qmd://collection/path.md"
        );
        assert_eq!(
            normalize_virtual_path("//journals/2025-01-01.md"),
            "qmd://journals/2025-01-01.md"
        );
    }

    #[test]
    fn normalize_collapses_extra_slashes() {
        assert_eq!(
            normalize_virtual_path("qmd:////collection/path.md"),
            "qmd://collection/path.md"
        );
        assert_eq!(
            normalize_virtual_path("qmd:///journals/2025-01-01.md"),
            "qmd://journals/2025-01-01.md"
        );
        assert_eq!(
            normalize_virtual_path("qmd:///////archive/file.md"),
            "qmd://archive/file.md"
        );
    }

    #[test]
    fn normalize_handles_collection_roots() {
        assert_eq!(
            normalize_virtual_path("qmd://collection/"),
            "qmd://collection/"
        );
        assert_eq!(
            normalize_virtual_path("qmd://collection"),
            "qmd://collection"
        );
        assert_eq!(normalize_virtual_path("//collection/"), "qmd://collection/");
    }

    #[test]
    fn normalize_preserves_bare_paths() {
        assert_eq!(
            normalize_virtual_path("collection/path.md"),
            "collection/path.md"
        );
        assert_eq!(
            normalize_virtual_path("journals/2025-01-01.md"),
            "journals/2025-01-01.md"
        );
    }

    #[test]
    fn normalize_preserves_absolute_paths() {
        assert_eq!(
            normalize_virtual_path("/Users/test/file.md"),
            "/Users/test/file.md"
        );
        assert_eq!(
            normalize_virtual_path("/absolute/path/file.md"),
            "/absolute/path/file.md"
        );
    }

    #[test]
    fn normalize_preserves_home_relative_paths() {
        assert_eq!(
            normalize_virtual_path("~/Documents/file.md"),
            "~/Documents/file.md"
        );
    }

    #[test]
    fn normalize_preserves_docids() {
        assert_eq!(normalize_virtual_path("#abc123"), "#abc123");
        assert_eq!(normalize_virtual_path("#def456"), "#def456");
    }

    #[test]
    fn normalize_trims_whitespace() {
        assert_eq!(
            normalize_virtual_path("  qmd://collection/path.md  "),
            "qmd://collection/path.md"
        );
        assert_eq!(
            normalize_virtual_path("  //collection/path.md  "),
            "qmd://collection/path.md"
        );
    }

    // ---- is_virtual_path (port of store.test.ts:3640) ----

    #[test]
    fn is_virtual_recognizes_qmd() {
        assert!(is_virtual_path("qmd://collection/path.md"));
        assert!(is_virtual_path("qmd://journals/2025-01-01.md"));
        assert!(is_virtual_path("qmd://collection"));
    }

    #[test]
    fn is_virtual_recognizes_double_slash() {
        assert!(is_virtual_path("//collection/path.md"));
        assert!(is_virtual_path("//journals/2025-01-01.md"));
    }

    #[test]
    fn is_virtual_rejects_bare_paths() {
        assert!(!is_virtual_path("collection/path.md"));
        assert!(!is_virtual_path("journals/2025-01-01.md"));
        assert!(!is_virtual_path("archive/subfolder/file.md"));
    }

    #[test]
    fn is_virtual_rejects_docids() {
        assert!(!is_virtual_path("#abc123"));
        assert!(!is_virtual_path("#def456"));
    }

    #[test]
    fn is_virtual_rejects_absolute_paths() {
        assert!(!is_virtual_path("/Users/test/file.md"));
        assert!(!is_virtual_path("/absolute/path/file.md"));
    }

    #[test]
    fn is_virtual_rejects_home_relative_paths() {
        assert!(!is_virtual_path("~/Documents/file.md"));
        assert!(!is_virtual_path("~/notes/journal.md"));
    }

    #[test]
    fn is_virtual_rejects_paths_without_slashes() {
        assert!(!is_virtual_path("file.md"));
        assert!(!is_virtual_path("document"));
    }

    // ---- parse_virtual_path (port of store.test.ts:3680) ----

    #[test]
    fn parse_standard_qmd_paths() {
        let vp = parse_virtual_path("qmd://collection/path.md").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "path.md");
        assert_eq!(vp.index_name, None);

        let vp = parse_virtual_path("qmd://journals/2025-01-01.md").unwrap();
        assert_eq!(vp.collection, "journals");
        assert_eq!(vp.path, "2025-01-01.md");
    }

    #[test]
    fn parse_nested_directories() {
        let vp = parse_virtual_path("qmd://archive/subfolder/file.md").unwrap();
        assert_eq!(vp.collection, "archive");
        assert_eq!(vp.path, "subfolder/file.md");
    }

    #[test]
    fn parse_collection_roots() {
        let vp = parse_virtual_path("qmd://collection/").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "");

        let vp = parse_virtual_path("qmd://collection").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "");
    }

    #[test]
    fn parse_double_slash_shorthand() {
        let vp = parse_virtual_path("//collection/path.md").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "path.md");
    }

    #[test]
    fn parse_extra_slashes() {
        let vp = parse_virtual_path("qmd:////collection/path.md").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "path.md");
    }

    #[test]
    fn parse_index_query_parameter() {
        let vp = parse_virtual_path("qmd://collection/path.md?index=docs-v2").unwrap();
        assert_eq!(vp.collection, "collection");
        assert_eq!(vp.path, "path.md");
        assert_eq!(vp.index_name.as_deref(), Some("docs-v2"));
    }

    #[test]
    fn parse_index_query_url_decoding() {
        // URLSearchParams: `+` → space and `%xx` are decoded, then trimmed.
        assert_eq!(
            parse_virtual_path("qmd://c/p?index=docs+v2")
                .unwrap()
                .index_name
                .as_deref(),
            Some("docs v2")
        );
        assert_eq!(
            parse_virtual_path("qmd://c/p?index=a%20b")
                .unwrap()
                .index_name
                .as_deref(),
            Some("a b")
        );
        // First `index` key wins; other keys ignored.
        assert_eq!(
            parse_virtual_path("qmd://c/p?foo=1&index=x&index=y")
                .unwrap()
                .index_name
                .as_deref(),
            Some("x")
        );
        // Empty / missing value → None.
        assert_eq!(
            parse_virtual_path("qmd://c/p?index=").unwrap().index_name,
            None
        );
        assert_eq!(
            parse_virtual_path("qmd://c/p?foo=1").unwrap().index_name,
            None
        );
    }

    #[test]
    fn parse_keeps_only_first_query_segment() {
        // TS `split("?")` destructuring keeps segment [1]; anything after a
        // second `?` is dropped.
        assert_eq!(
            parse_virtual_path("qmd://c/p?index=y?extra")
                .unwrap()
                .index_name
                .as_deref(),
            Some("y")
        );
    }

    #[test]
    fn build_encodes_index_and_round_trips_special_chars() {
        let built = build_virtual_path("docs", "x.md", Some("a b/c"));
        assert_eq!(built, "qmd://docs/x.md?index=a%20b%2Fc");
        assert_eq!(
            parse_virtual_path(&built).unwrap().index_name.as_deref(),
            Some("a b/c")
        );
    }

    #[test]
    fn parse_rejects_non_virtual_paths() {
        assert!(parse_virtual_path("/absolute/path.md").is_err());
        assert!(parse_virtual_path("~/home/path.md").is_err());
        assert!(parse_virtual_path("#docid").is_err());
        assert!(parse_virtual_path("file.md").is_err());
        assert!(parse_virtual_path("collection/path.md").is_err());
    }

    // ---- build_virtual_path round-trips ----

    #[test]
    fn build_round_trips() {
        let built = build_virtual_path("docs", "readme.md", None);
        assert_eq!(built, "qmd://docs/readme.md");
        let parsed = parse_virtual_path(&built).unwrap();
        assert_eq!(parsed.collection, "docs");
        assert_eq!(parsed.path, "readme.md");
        assert_eq!(parsed.index_name, None);

        let built = build_virtual_path("docs", "readme.md", Some("idx"));
        assert_eq!(built, "qmd://docs/readme.md?index=idx");
        let parsed = parse_virtual_path(&built).unwrap();
        assert_eq!(parsed.collection, "docs");
        assert_eq!(parsed.path, "readme.md");
        assert_eq!(parsed.index_name.as_deref(), Some("idx"));
    }
}
