//! Document IDs and "handelization" — path normalisation for matching.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 1811–1883 + 2531–2580.

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};

use super::{Error, Result};

/// Short docid: first 6 chars of a SHA-256 hex hash.
/// Mirrors `getDocid` (`store.ts:1811–1813`).
pub fn get_docid(hash: &str) -> String {
    hash.chars().take(6).collect()
}

/// Normalise a docid for comparison: trim → strip a matching pair of
/// surrounding quotes → strip a single leading `#`. Case is **preserved**
/// (TS does not lowercase here; the downstream `LIKE` is ASCII
/// case-insensitive anyway).
///
/// Mirrors `normalizeDocid` (`store.ts:2531`).
pub fn normalize_docid(docid: &str) -> String {
    let mut s = docid.trim();

    // Strip a matching pair of surrounding quotes (single or double).
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            s = &s[1..s.len() - 1];
        }
    }

    // Strip a single leading `#`.
    s.strip_prefix('#').unwrap_or(s).to_string()
}

/// Quick "does this look like a docid?" test: the normalised form is a hex
/// string of **6 or more** characters (case-insensitive).
///
/// Mirrors `isDocid` (`store.ts:2553`): `length >= 6 && /^[a-f0-9]+$/i`.
pub fn is_docid(input: &str) -> bool {
    let normalised = normalize_docid(input);
    normalised.len() >= 6 && normalised.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Reference returned by [`find_document_by_docid`].
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentRef {
    pub filepath: String,
    pub hash: String,
}

/// Look up an active document whose hash begins with `docid`.
pub fn find_document_by_docid(conn: &Connection, docid: &str) -> Result<Option<DocumentRef>> {
    let normalised = normalize_docid(docid);
    let pattern = format!("{normalised}%");
    let row = conn
        .query_row(
            r#"SELECT collection || '/' || path AS filepath, hash
               FROM documents
               WHERE active = 1 AND hash LIKE ?
               LIMIT 1"#,
            params![pattern],
            |row| {
                Ok(DocumentRef {
                    filepath: row.get::<_, String>(0)?,
                    hash: row.get::<_, String>(1)?,
                })
            },
        )
        .optional()?;
    Ok(row)
}

// Unicode-property regexes mirroring the TS `\p{...}` classes used by
// `handelize`. The `regex` crate ships Unicode tables by default.

/// Any "valid filename content" character: letter, number, emoji
/// (`\p{So}`), modifier symbol (`\p{Sk}`), or `$`. Mirrors the
/// `hasValidContent` test (`store.ts:1843`).
static VALID_CONTENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\p{L}\p{N}\p{So}\p{Sk}$]").expect("valid regex"));

/// Trailing extension used by the *validation* step: a dot followed by any
/// run of non-dot chars (`/\.[^.]+$/`, `store.ts:1842`).
static EXT_VALIDATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\.[^.]+$").expect("valid regex"));

/// Trailing extension preserved by the *transform* step: a dot followed by
/// ASCII alphanumerics, case-insensitive (`/(\.[a-z0-9]+)$/i`, `store.ts:1859`).
static EXT_TRANSFORM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\.[a-z0-9]+$").expect("valid regex"));

/// Runs of characters to collapse into a single `-` when cleaning a segment:
/// anything that is not a letter, number, or `$` (`/[^\p{L}\p{N}$]+/gu`).
static NON_KEEP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\p{L}\p{N}$]+").expect("valid regex"));

/// Emoji run for `emoji_to_hex`: `(?:\p{So}\p{Mn}?|\p{Sk})+` (`store.ts:1826`).
static EMOJI_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\p{So}\p{Mn}?|\p{Sk})+").expect("valid regex"));

/// Single emoji / modifier-symbol char (`\p{So}|\p{Sk}`), used to decide
/// which chars inside a matched run get hex-encoded (combining marks `\p{Mn}`
/// are dropped). Mirrors the inner filter at `store.ts:1828`.
static EMOJI_CHAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[\p{So}\p{Sk}]$").expect("valid regex"));

/// Convert emoji runs to dash-joined lowercase hex codepoints.
///
/// Matches `(?:\p{So}\p{Mn}?|\p{Sk})+` then hex-encodes only the `\p{So}` /
/// `\p{Sk}` chars in the run (combining marks are discarded). Codepoints use
/// `{:x}` (no zero-padding), e.g. `🐘` → `1f418`, `🐘🎉` → `1f418-1f389`.
/// Mirrors `emojiToHex` (`store.ts:1825–1831`).
fn emoji_to_hex(segment: &str) -> String {
    EMOJI_RUN
        .replace_all(segment, |caps: &regex::Captures<'_>| {
            let run = caps.get(0).unwrap().as_str();
            run.chars()
                .filter(|c| EMOJI_CHAR.is_match(c.encode_utf8(&mut [0u8; 4])))
                .map(|c| format!("{:x}", c as u32))
                .collect::<Vec<_>>()
                .join("-")
        })
        .into_owned()
}

/// Replace runs of non-(letter|digit|`$`) with `-`, then strip leading and
/// trailing dashes. Mirrors `.replace(/[^\p{L}\p{N}$]+/gu, '-').replace(/^-+|-+$/g, '')`.
fn clean_segment(segment: &str) -> String {
    NON_KEEP
        .replace_all(segment, "-")
        .trim_matches('-')
        .to_string()
}

/// Handelize a filename to a token-friendly form.
///
/// Full port of `handelize` (`store.ts:1833–1883`): triple underscore →
/// folder separator, emoji → hex codepoints, per-segment cleaning that keeps
/// only letters / numbers / `$` (collapsing other runs to `-`), with the
/// final segment's extension preserved.
///
/// Returns [`Error::InvalidPath`] (TS `throw`) when the input is empty /
/// whitespace-only, when the last segment has no valid filename content, or
/// when processing yields an empty string.
pub fn handelize(path: &str) -> Result<String> {
    if path.trim().is_empty() {
        return Err(Error::InvalidPath("path cannot be empty".into()));
    }

    // Validation uses the *original* path's last segment (before `___` → `/`),
    // with a generic `\.[^.]+$` extension stripped.
    let last_segment = path.split('/').rfind(|s| !s.is_empty()).unwrap_or("");
    let filename_without_ext = EXT_VALIDATE.replace(last_segment, "");
    if !VALID_CONTENT.is_match(&filename_without_ext) {
        return Err(Error::InvalidPath(format!(
            "path \"{path}\" has no valid filename content"
        )));
    }

    // Transform: `___` → `/`, then clean each segment.
    let canonical = path.replace("___", "/");
    let raw_segments: Vec<&str> = canonical.split('/').collect();
    let n = raw_segments.len();

    let mut out: Vec<String> = Vec::with_capacity(n);
    for (i, seg) in raw_segments.iter().enumerate() {
        let is_last = i + 1 == n;
        let seg = emoji_to_hex(seg);

        let cleaned = if is_last {
            if let Some(m) = EXT_TRANSFORM.find(&seg) {
                let ext = &seg[m.start()..];
                let name = &seg[..m.start()];
                format!("{}{}", clean_segment(name), ext)
            } else {
                clean_segment(&seg)
            }
        } else {
            clean_segment(&seg)
        };

        if !cleaned.is_empty() {
            out.push(cleaned);
        }
    }

    let joined = out.join("/");
    if joined.is_empty() {
        return Err(Error::InvalidPath(format!(
            "path \"{path}\" resulted in empty string after processing"
        )));
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docid_first_six_chars() {
        assert_eq!(get_docid("abcdef0123456789"), "abcdef");
    }

    // --- normalize_docid: ported from store.test.ts `describe("normalizeDocid")` ---

    #[test]
    fn normalize_strips_leading_hash() {
        assert_eq!(normalize_docid("#abc123"), "abc123");
        assert_eq!(normalize_docid("#def456"), "def456");
    }

    #[test]
    fn normalize_returns_bare_hex_unchanged() {
        assert_eq!(normalize_docid("abc123"), "abc123");
        assert_eq!(normalize_docid("def456"), "def456");
    }

    #[test]
    fn normalize_strips_surrounding_double_quotes() {
        assert_eq!(normalize_docid("\"#abc123\""), "abc123");
        assert_eq!(normalize_docid("\"abc123\""), "abc123");
    }

    #[test]
    fn normalize_strips_surrounding_single_quotes() {
        assert_eq!(normalize_docid("'#abc123'"), "abc123");
        assert_eq!(normalize_docid("'abc123'"), "abc123");
    }

    #[test]
    fn normalize_handles_quoted_docid_without_hash() {
        assert_eq!(normalize_docid("\"def456\""), "def456");
        assert_eq!(normalize_docid("'def456'"), "def456");
    }

    #[test]
    fn normalize_handles_whitespace() {
        assert_eq!(normalize_docid("  #abc123  "), "abc123");
        assert_eq!(normalize_docid("  abc123  "), "abc123");
    }

    #[test]
    fn normalize_preserves_uppercase_hex() {
        assert_eq!(normalize_docid("#ABC123"), "ABC123");
        assert_eq!(normalize_docid("\"ABC123\""), "ABC123");
    }

    #[test]
    fn normalize_does_not_strip_mismatched_quotes() {
        // `"abc123'` — opens with `"`, closes with `'` → no strip.
        assert_eq!(normalize_docid("\"abc123'"), "\"abc123'");
        // `'abc123"` — opens with `'`, closes with `"` → no strip.
        assert_eq!(normalize_docid("'abc123\""), "'abc123\"");
    }

    // --- is_docid: ported from store.test.ts `describe("isDocid")` ---

    #[test]
    fn is_docid_accepts_hash_format() {
        assert!(is_docid("#abc123"));
        assert!(is_docid("#def456"));
        assert!(is_docid("#ABCDEF"));
    }

    #[test]
    fn is_docid_accepts_bare_six_char_hex() {
        assert!(is_docid("abc123"));
        assert!(is_docid("def456"));
        assert!(is_docid("ABCDEF"));
    }

    #[test]
    fn is_docid_accepts_longer_hex() {
        assert!(is_docid("abc123def456"));
        assert!(is_docid("#abc123def456"));
    }

    #[test]
    fn is_docid_accepts_double_quoted() {
        assert!(is_docid("\"#abc123\""));
        assert!(is_docid("\"abc123\""));
    }

    #[test]
    fn is_docid_accepts_single_quoted() {
        assert!(is_docid("'#abc123'"));
        assert!(is_docid("'abc123'"));
    }

    #[test]
    fn is_docid_rejects_non_hex() {
        assert!(!is_docid("ghijkl"));
        assert!(!is_docid("#ghijkl"));
        assert!(!is_docid("abc12g"));
    }

    #[test]
    fn is_docid_rejects_shorter_than_six() {
        assert!(!is_docid("abc12"));
        assert!(!is_docid("#abc1"));
        assert!(!is_docid("'abc'"));
    }

    #[test]
    fn is_docid_rejects_empty() {
        assert!(!is_docid(""));
        assert!(!is_docid("#"));
        assert!(!is_docid("\"\""));
    }

    #[test]
    fn is_docid_rejects_file_paths() {
        assert!(!is_docid("/path/to/file.md"));
        assert!(!is_docid("path/to/file.md"));
        assert!(!is_docid("qmd://collection/file.md"));
    }

    #[test]
    fn is_docid_rejects_hex_with_extension() {
        assert!(!is_docid("abc123.md"));
    }

    // --- handelize: ported from store.helpers.unit.test.ts `describe("handelize")` ---

    fn h(input: &str) -> String {
        handelize(input).unwrap()
    }

    #[test]
    fn handelize_preserves_original_case() {
        assert_eq!(h("README.md"), "README.md");
        assert_eq!(h("MyFile.MD"), "MyFile.MD");
    }

    #[test]
    fn handelize_preserves_folder_structure() {
        assert_eq!(h("a/b/c/d.md"), "a/b/c/d.md");
        assert_eq!(h("docs/api/README.md"), "docs/api/README.md");
    }

    #[test]
    fn handelize_replaces_non_word_chars_with_dash() {
        assert_eq!(h("hello world.md"), "hello-world.md");
        assert_eq!(h("file (1).md"), "file-1.md");
        assert_eq!(h("foo@bar#baz.md"), "foo-bar-baz.md");
    }

    #[test]
    fn handelize_collapses_multiple_special_chars() {
        assert_eq!(h("hello   world.md"), "hello-world.md");
        assert_eq!(h("foo---bar.md"), "foo-bar.md");
        assert_eq!(h("a  -  b.md"), "a-b.md");
    }

    #[test]
    fn handelize_removes_leading_trailing_dashes_per_segment() {
        assert_eq!(h("-hello-.md"), "hello.md");
        assert_eq!(h("--test--.md"), "test.md");
        assert_eq!(h("a/-b-/c.md"), "a/b/c.md");
    }

    #[test]
    fn handelize_triple_underscore_to_slash() {
        assert_eq!(h("foo___bar.md"), "foo/bar.md");
        assert_eq!(h("notes___2025___january.md"), "notes/2025/january.md");
        assert_eq!(h("a/b___c/d.md"), "a/b/c/d.md");
    }

    #[test]
    fn handelize_complex_meeting_notes() {
        let complex =
            "Money Movement Licensing Review - 2025／11／19 10:25 EST - Notes by Gemini.md";
        let result = h(complex);
        assert_eq!(
            result,
            "Money-Movement-Licensing-Review-2025-11-19-10-25-EST-Notes-by-Gemini.md"
        );
        assert!(!result.contains(' '));
        assert!(!result.contains('／'));
        assert!(!result.contains(':'));
    }

    #[test]
    fn handelize_unicode_characters() {
        assert_eq!(h("日本語.md"), "日本語.md");
        assert_eq!(h("Зоны и проекты.md"), "Зоны-и-проекты.md");
        assert_eq!(h("café-notes.md"), "café-notes.md");
        assert_eq!(h("naïve.md"), "naïve.md");
        assert_eq!(h("日本語-notes.md"), "日本語-notes.md");
    }

    #[test]
    fn handelize_emoji_filenames_issue_302() {
        // Emoji-only filenames convert to hex codepoints.
        assert_eq!(h("🐘.md"), "1f418.md");
        assert_eq!(h("🎉.md"), "1f389.md");
        // Emoji mixed with text.
        assert_eq!(h("notes 🐘.md"), "notes-1f418.md");
        assert_eq!(h("🐘 elephant.md"), "1f418-elephant.md");
        // Multiple emoji.
        assert_eq!(h("🐘🎉.md"), "1f418-1f389.md");
        // Emoji in directory names.
        assert_eq!(h("🐘/notes.md"), "1f418/notes.md");
    }

    #[test]
    fn handelize_dates_and_times() {
        assert_eq!(h("meeting-2025-01-15.md"), "meeting-2025-01-15.md");
        assert_eq!(h("notes 2025/01/15.md"), "notes-2025/01/15.md");
        assert_eq!(h("call_10:30_AM.md"), "call-10-30-AM.md");
    }

    #[test]
    fn handelize_special_project_naming() {
        assert_eq!(h("PROJECT_ABC_v2.0.md"), "PROJECT-ABC-v2-0.md");
        assert_eq!(h("[WIP] Feature Request.md"), "WIP-Feature-Request.md");
        assert_eq!(h("(DRAFT) Proposal v1.md"), "DRAFT-Proposal-v1.md");
    }

    #[test]
    fn handelize_symbol_only_route_filenames() {
        assert_eq!(h("routes/api/auth/$.ts"), "routes/api/auth/$.ts");
        assert_eq!(h("app/routes/$id.tsx"), "app/routes/$id.tsx");
    }

    #[test]
    fn handelize_filters_out_empty_segments() {
        assert_eq!(h("a//b/c.md"), "a/b/c.md");
        assert_eq!(h("/a/b/"), "a/b");
        assert_eq!(h("///test///"), "test");
    }

    #[test]
    fn handelize_throws_for_invalid_inputs() {
        assert!(handelize("").is_err());
        assert!(handelize("   ").is_err());
        assert!(handelize(".md").is_err());
        assert!(handelize("...").is_err());
        assert!(handelize("___").is_err());
    }

    #[test]
    fn handelize_minimal_valid_inputs() {
        assert_eq!(h("a"), "a");
        assert_eq!(h("1"), "1");
        assert_eq!(h("a.md"), "a.md");
    }
}
