//! Document IDs and "handelization" — path normalisation for matching.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 1811–1883 + 2531–2580.

use rusqlite::{params, Connection, OptionalExtension};

use super::Result;

/// Short docid: first 6 chars of a SHA-256 hex hash.
/// Mirrors `getDocid` (`store.ts:1811–1813`).
pub fn get_docid(hash: &str) -> String {
    hash.chars().take(6).collect()
}

/// Normalise a docid for comparison: lowercase + strip leading `#`.
/// Mirrors `normalizeDocid` (`store.ts:2531`).
pub fn normalize_docid(docid: &str) -> String {
    let trimmed = docid.trim().trim_start_matches('#');
    trimmed.to_ascii_lowercase()
}

/// Quick "does this look like a docid?" test: 6 lowercase-hex chars.
/// Mirrors `isDocid` (`store.ts:2553`).
pub fn is_docid(input: &str) -> bool {
    let normalised = normalize_docid(input);
    normalised.len() == 6 && normalised.bytes().all(|b| b.is_ascii_hexdigit())
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

/// Handelize a filename to a token-friendly form. Simpler than the TS
/// version (which handles emoji-to-hex and Unicode categories via `\p{...}`);
/// for the Rust port we keep ASCII-letters/digits/underscore as-is, collapse
/// runs of other characters into a single `-`, and preserve `/` boundaries.
///
/// Mirrors `handelize` (`store.ts:1833–1883`) for the common ASCII case.
/// Returns `None` for empty / whitespace-only inputs.
pub fn handelize(path: &str) -> Option<String> {
    if path.trim().is_empty() {
        return None;
    }

    // Triple underscore -> folder separator.
    let canonical = path.replace("___", "/");

    let segments: Vec<&str> = canonical.split('/').filter(|s| !s.is_empty()).collect();
    let n = segments.len();

    let mut out: Vec<String> = Vec::with_capacity(n);
    for (i, seg) in segments.iter().enumerate() {
        let is_last = i + 1 == n;

        let (stem, ext) = if is_last {
            match seg.rsplit_once('.') {
                Some((s, e)) if !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()) => {
                    (s, format!(".{e}"))
                }
                _ => (*seg, String::new()),
            }
        } else {
            (*seg, String::new())
        };

        let cleaned = clean_segment(stem);
        if cleaned.is_empty() && ext.is_empty() {
            continue;
        }
        out.push(format!("{cleaned}{ext}"));
    }

    let joined = out.join("/");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Replace runs of non-(letter|digit|`$`) characters with `-`, and strip
/// leading/trailing dashes.
fn clean_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut prev_dash = true; // suppress leading dash
    for ch in segment.chars() {
        let keep = ch.is_alphanumeric() || ch == '$';
        if keep {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docid_first_six_chars() {
        assert_eq!(get_docid("abcdef0123456789"), "abcdef");
    }

    #[test]
    fn normalize_strips_hash_prefix() {
        assert_eq!(normalize_docid("#ABC123"), "abc123");
        assert_eq!(normalize_docid("  AbC123  "), "abc123");
    }

    #[test]
    fn is_docid_accepts_six_hex() {
        assert!(is_docid("abc123"));
        assert!(is_docid("#ABC123"));
        assert!(!is_docid("xyz123"));
        assert!(!is_docid("abc12"));
        assert!(!is_docid("abc1234"));
    }

    #[test]
    fn handelize_simple_path() {
        assert_eq!(
            handelize("docs/Hello World.md").as_deref(),
            Some("docs/Hello-World.md")
        );
    }

    #[test]
    fn handelize_triple_underscore_to_slash() {
        assert_eq!(handelize("a___b___c.md").as_deref(), Some("a/b/c.md"));
    }

    #[test]
    fn handelize_empty_rejected() {
        assert!(handelize("").is_none());
        assert!(handelize("   ").is_none());
    }
}
