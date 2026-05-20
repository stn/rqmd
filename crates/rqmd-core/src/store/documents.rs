//! `content` + `documents` tables: CRUD, title extraction, FTS sync.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 2150–2355 + `rebuildDocumentFTS`
//! (2201–2221). Content-addressable storage: the `content` table holds
//! `(hash, body)` pairs, the `documents` table maps `(collection, path)` to
//! a hash. Triggers keep `documents_fts` in sync for writes that go through
//! `documents` directly, but the indexing path here calls
//! [`rebuild_document_fts`] to ensure CJK normalisation runs before the
//! tokeniser sees the text.

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use super::Result;
use super::schema::normalize_cjk_for_fts;

// ============================================================================
// Hashing & title extraction
// ============================================================================

/// SHA-256 of `content`, lowercase hex. Mirrors `hashContent`
/// (`store.ts:2150–2154`).
pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

static MD_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^##?\s+(.+)$").expect("valid regex"));
static MD_HEADING_NEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^##\s+(.+)$").expect("valid regex"));
static ORG_TITLE: LazyLock<Regex> = LazyLock::new(|| {
    regex::RegexBuilder::new(r"(?m)^#\+TITLE:\s*(.+)$")
        .case_insensitive(true)
        .build()
        .expect("valid regex")
});
static ORG_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\*+\s+(.+)$").expect("valid regex"));

/// Extract a title from `content` based on `filename` extension.
/// Mirrors `extractTitle` (`store.ts:2178–2186`).
pub fn extract_title(content: &str, filename: &str) -> String {
    let lower = filename.to_ascii_lowercase();
    let ext = lower.rsplit_once('.').map(|(_, e)| format!(".{e}"));

    if let Some(ext) = ext.as_deref() {
        match ext {
            ".md" => {
                if let Some(m) = MD_HEADING.captures(content) {
                    let title = m.get(1).map(|m| m.as_str().trim()).unwrap_or("");
                    if (title == "📝 Notes" || title == "Notes")
                        && let Some(next) = MD_HEADING_NEXT.captures(content)
                        && let Some(t) = next.get(1)
                    {
                        return t.as_str().trim().to_string();
                    }
                    if !title.is_empty() {
                        return title.to_string();
                    }
                }
            }
            ".org" => {
                if let Some(m) = ORG_TITLE.captures(content)
                    && let Some(t) = m.get(1)
                {
                    return t.as_str().trim().to_string();
                }
                if let Some(m) = ORG_HEADING.captures(content)
                    && let Some(t) = m.get(1)
                {
                    return t.as_str().trim().to_string();
                }
            }
            _ => {}
        }
    }

    // Fallback: filename without extension, last path segment.
    let stem = match filename.rsplit_once('.') {
        Some((stem, _)) => stem,
        None => filename,
    };
    stem.rsplit_once('/')
        .map(|(_, last)| last)
        .unwrap_or(stem)
        .to_string()
}

// ============================================================================
// CRUD
// ============================================================================

/// Active document row returned by [`find_active_document`] etc.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveDocument {
    pub id: i64,
    pub hash: String,
    pub title: String,
}

/// Insert content if not already present (content-addressable).
pub fn insert_content(conn: &Connection, hash: &str, content: &str, created_at: &str) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)",
        params![hash, content, created_at],
    )?;
    Ok(())
}

/// Insert (or upsert) a document. Rebuilds the FTS row with CJK normalisation.
pub fn insert_document(
    conn: &Connection,
    collection: &str,
    path: &str,
    title: &str,
    hash: &str,
    created_at: &str,
    modified_at: &str,
) -> Result<i64> {
    conn.execute(
        r#"INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
           VALUES (?, ?, ?, ?, ?, ?, 1)
           ON CONFLICT(collection, path) DO UPDATE SET
               title = excluded.title,
               hash = excluded.hash,
               modified_at = excluded.modified_at,
               active = 1"#,
        params![collection, path, title, hash, created_at, modified_at],
    )?;

    let id: i64 = conn.query_row(
        "SELECT id FROM documents WHERE collection = ? AND path = ?",
        params![collection, path],
        |row| row.get(0),
    )?;
    rebuild_document_fts(conn, id)?;
    Ok(id)
}

pub fn find_active_document(
    conn: &Connection,
    collection: &str,
    path: &str,
) -> Result<Option<ActiveDocument>> {
    let row = conn
        .query_row(
            "SELECT id, hash, title FROM documents WHERE collection = ? AND path = ? AND active = 1",
            params![collection, path],
            |row| {
                Ok(ActiveDocument {
                    id: row.get(0)?,
                    hash: row.get(1)?,
                    title: row.get(2)?,
                })
            },
        )
        .optional()?;
    Ok(row)
}

pub fn find_or_migrate_legacy_document(
    conn: &mut Connection,
    collection: &str,
    path: &str,
) -> Result<Option<ActiveDocument>> {
    if let Some(doc) = find_active_document(conn, collection, path)? {
        return Ok(Some(doc));
    }

    let legacy_id: Option<i64> = conn
        .query_row(
            r#"SELECT id FROM documents
               WHERE collection = ? AND path COLLATE NOCASE = ? AND active = 1
               ORDER BY id LIMIT 1"#,
            params![collection, path],
            |row| row.get(0),
        )
        .optional()?;
    let Some(legacy_id) = legacy_id else {
        return Ok(None);
    };

    let tx = conn.transaction()?;
    let changes = tx.execute(
        "UPDATE OR IGNORE documents SET path = ? WHERE id = ? AND active = 1",
        params![path, legacy_id],
    )?;
    if changes == 0 {
        return Ok(None);
    }
    rebuild_document_fts(&tx, legacy_id)?;
    tx.commit()?;

    find_active_document(conn, collection, path)
}

pub fn update_document_title(
    conn: &Connection,
    document_id: i64,
    title: &str,
    modified_at: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET title = ?, modified_at = ? WHERE id = ?",
        params![title, modified_at, document_id],
    )?;
    rebuild_document_fts(conn, document_id)?;
    Ok(())
}

pub fn update_document(
    conn: &Connection,
    document_id: i64,
    title: &str,
    hash: &str,
    modified_at: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?",
        params![title, hash, modified_at, document_id],
    )?;
    rebuild_document_fts(conn, document_id)?;
    Ok(())
}

pub fn deactivate_document(conn: &Connection, collection: &str, path: &str) -> Result<()> {
    conn.execute(
        "UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1",
        params![collection, path],
    )?;
    Ok(())
}

pub fn get_active_document_paths(conn: &Connection, collection: &str) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT path FROM documents WHERE collection = ? AND active = 1")?;
    let rows = stmt
        .query_map(params![collection], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(rows)
}

/// Rebuild a single FTS row, applying CJK normalisation. Used after every
/// insert/update so the tokeniser sees spaced CJK runs.
pub fn rebuild_document_fts(conn: &Connection, document_id: i64) -> Result<()> {
    let row: Option<(i64, String, String, String, String)> = conn
        .query_row(
            r#"SELECT d.id, d.collection, d.path, d.title, content.doc AS body
               FROM documents d
               JOIN content ON content.hash = d.hash
               WHERE d.id = ? AND d.active = 1"#,
            params![document_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;

    conn.execute("DELETE FROM documents_fts WHERE rowid = ?", params![document_id])?;

    if let Some((id, collection, path, title, body)) = row {
        let filepath = normalize_cjk_for_fts(&format!("{collection}/{path}"));
        conn.execute(
            "INSERT INTO documents_fts(rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
            params![
                id,
                filepath,
                normalize_cjk_for_fts(&title),
                normalize_cjk_for_fts(&body),
            ],
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_content_is_stable() {
        assert_eq!(
            hash_content("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn extract_title_markdown_heading() {
        let title = extract_title("# Hello World\n\nbody", "x.md");
        assert_eq!(title, "Hello World");
    }

    #[test]
    fn extract_title_markdown_skip_notes() {
        let title = extract_title("# 📝 Notes\n\n## Real Title\nbody", "x.md");
        assert_eq!(title, "Real Title");
    }

    #[test]
    fn extract_title_fallback_to_filename() {
        let title = extract_title("no heading", "foo/bar.md");
        assert_eq!(title, "bar");
    }

    #[test]
    fn extract_title_org() {
        let title = extract_title("#+TITLE: From Property\n* heading", "x.org");
        assert_eq!(title, "From Property");
    }
}
