//! SQLite schema initialisation, FTS5 helpers, and CJK normalisation.
//!
//! Port of the schema-related portion of `tobi/qmd`'s `src/store.ts`
//! (`initializeDatabase` 806–960, `normalizeCjkForFTS` 740–763,
//! `sanitizeFTS5Term` 3016–…, `rebuildFTSForCjkNormalization` 764–804).

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::Connection;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use super::Result;

const FTS_CJK_NORMALIZED_VERSION: &str = "1";

static CJK_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]+").expect("valid regex")
});

static CJK_CHAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]").expect("valid regex")
});

/// Decompose CJK runs by inserting spaces between every character.
///
/// FTS5's `unicode61` tokeniser does not segment CJK text, so we space
/// every character so exact CJK queries can be translated into phrase
/// queries while Latin text keeps the default tokenisation.
/// Mirrors `normalizeCjkForFTS` (`store.ts:748–750`).
pub fn normalize_cjk_for_fts(text: &str) -> String {
    CJK_RUN
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let run = caps.get(0).unwrap().as_str();
            let mut out = String::with_capacity(run.len() * 2 + 2);
            out.push(' ');
            let mut first = true;
            for ch in run.chars() {
                if !first {
                    out.push(' ');
                }
                out.push(ch);
                first = false;
            }
            out.push(' ');
            out
        })
        .into_owned()
}

/// Return whether `text` contains any CJK character.
pub fn contains_cjk(text: &str) -> bool {
    CJK_CHAR.is_match(text)
}

/// Sanitise a single FTS5 search term to safe *content*: keep Unicode
/// letters (`\p{L}`), numbers (`\p{N}`), `'`, and `_`; drop everything else;
/// lowercase.
///
/// Mirrors TS `sanitizeFTS5Term` (`store.ts:3016–3018`):
/// `term.replace(/[^\p{L}\p{N}'_]/gu, '').toLowerCase()`. The keep set is
/// decided by Unicode *general category* (via `unicode-properties`) rather
/// than `char::is_alphanumeric()` — the latter also matches
/// `Other_Alphabetic` combining marks (e.g. Indic vowel signs), which `\p{L}`
/// excludes, so it would silently diverge from the TS filter.
///
/// Because every FTS5 special character is removed here, callers can safely
/// wrap the result in `"..."` themselves (see
/// [`crate::store::search::build_fts5_query`]) with no quote-escaping needed.
/// There is intentionally **no** "quoting" variant — TS qmd has none, and an
/// earlier quote-based helper diverged from it.
pub fn sanitize_fts5_term(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        let keep = ch == '\''
            || ch == '_'
            || matches!(
                ch.general_category_group(),
                GeneralCategoryGroup::Letter | GeneralCategoryGroup::Number
            );
        if keep {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// Sanitise an FTS5 phrase: CJK-normalise, then map each whitespace-delimited
/// token through [`sanitize_fts5_term`] and re-join with single spaces.
///
/// Mirrors TS `sanitizeFTS5Phrase` (`store.ts:756–762`).
pub fn sanitize_fts5_phrase(phrase: &str) -> String {
    let normalised = normalize_cjk_for_fts(phrase);
    normalised
        .split_whitespace()
        .map(sanitize_fts5_term)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Validate a lexical (FTS) query. Currently a stub that accepts any
/// non-empty query — exists so future SQL syntax checks live in one place.
pub fn validate_lex_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(super::Error::InvalidQuery("empty query".into()));
    }
    Ok(())
}

/// Validate a semantic (vector) query. Stub for forward compatibility.
pub fn validate_semantic_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(super::Error::InvalidQuery("empty query".into()));
    }
    Ok(())
}

// ============================================================================
// Schema initialisation
// ============================================================================

/// Apply the full schema (tables, indexes, triggers, virtual tables).
/// Idempotent — every statement uses `IF NOT EXISTS` / `IF EXISTS` so
/// reopening a store is a no-op except for the legacy table drops and the
/// CJK-FTS rebuild marker. Mirrors `initializeDatabase` (`store.ts:806–960`).
pub fn initialize(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(SCHEMA_DDL)?;
    upgrade_content_vectors_if_needed(conn)?;
    rebuild_fts_for_cjk_normalization(conn)?;
    Ok(())
}

const SCHEMA_DDL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Drop legacy tables that are now managed in YAML.
DROP TABLE IF EXISTS path_contexts;
DROP TABLE IF EXISTS collections;

-- Content-addressable storage — source of truth for document bodies.
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Documents — file-system layer mapping (collection, path) -> content hash.
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
    UNIQUE(collection, path)
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
CREATE INDEX IF NOT EXISTS idx_documents_hash       ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_path       ON documents(path, active);

-- Cache for LLM API responses (expandQuery / rerank results).
CREATE TABLE IF NOT EXISTS llm_cache (
    hash TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Per-chunk embedding metadata. The actual vectors live in vectors_vec.
CREATE TABLE IF NOT EXISTS content_vectors (
    hash TEXT NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    pos INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (hash, seq)
);

-- Self-contained collection config (mirrors YAML but DB is source of truth).
CREATE TABLE IF NOT EXISTS store_collections (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    pattern TEXT NOT NULL DEFAULT '**/*.md',
    ignore_patterns TEXT,
    include_by_default INTEGER DEFAULT 1,
    update_command TEXT,
    context TEXT
);

-- Key-value metadata (config_hash, fts_cjk_normalized_version, global_context).
CREATE TABLE IF NOT EXISTS store_config (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- FTS5 over (filepath, title, body) for BM25 lexical search.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    filepath, title, body,
    tokenize='porter unicode61'
);

-- Keep FTS in sync for callers that write directly to `documents`.
-- Indexing paths normalise CJK in TypeScript/Rust before insert.
DROP TRIGGER IF EXISTS documents_ai;
CREATE TRIGGER documents_ai AFTER INSERT ON documents
WHEN new.active = 1
BEGIN
    INSERT INTO documents_fts(rowid, filepath, title, body)
    SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
    WHERE new.active = 1;
END;

DROP TRIGGER IF EXISTS documents_ad;
CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id;
END;

DROP TRIGGER IF EXISTS documents_au;
CREATE TRIGGER documents_au AFTER UPDATE ON documents
BEGIN
    DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;

    INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
    SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
    WHERE new.active = 1;
END;
"#;

/// `content_vectors.seq` was added later in qmd; if an existing DB lacks the
/// column, drop and recreate both `content_vectors` and `vectors_vec`.
/// Mirrors the `cvInfo`/`hasSeqColumn` check at `store.ts:865–869`.
fn upgrade_content_vectors_if_needed(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(content_vectors)")?;
    let cols: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);

    if !cols.is_empty() && !cols.iter().any(|c| c == "seq") {
        conn.execute("DROP TABLE IF EXISTS content_vectors", [])?;
        conn.execute("DROP TABLE IF EXISTS vectors_vec", [])?;
        conn.execute(
            r#"CREATE TABLE content_vectors (
                hash TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                pos INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 1,
                embedded_at TEXT NOT NULL,
                PRIMARY KEY (hash, seq)
            )"#,
            [],
        )?;
    }

    Ok(())
}

/// Rebuild the FTS index with CJK normalisation if the marker version has
/// changed. Mirrors `rebuildFTSForCjkNormalization` (`store.ts:764–804`).
fn rebuild_fts_for_cjk_normalization(conn: &mut Connection) -> Result<()> {
    let current: Option<String> = conn
        .query_row(
            "SELECT value FROM store_config WHERE key = 'fts_cjk_normalized_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .ok();
    if current.as_deref() == Some(FTS_CJK_NORMALIZED_VERSION) {
        return Ok(());
    }

    // Try the bulk delete; if FTS5 shadow tables are in a wonky state, drop
    // and recreate documents_fts from scratch.
    if conn
        .execute("DELETE FROM documents_fts WHERE rowid >= 0", [])
        .is_err()
    {
        conn.execute("DROP TABLE IF EXISTS documents_fts", [])?;
        conn.execute(
            "CREATE VIRTUAL TABLE documents_fts USING fts5(filepath, title, body, tokenize='porter unicode61')",
            [],
        )?;
    }

    let rows: Vec<(i64, String, String, String, String)> = {
        let mut stmt = conn.prepare(
            "SELECT d.id, d.collection, d.path, d.title, content.doc AS body
             FROM documents d
             JOIN content ON content.hash = d.hash
             WHERE d.active = 1",
        )?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    let tx = conn.transaction()?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO documents_fts(rowid, filepath, title, body) VALUES (?, ?, ?, ?)",
        )?;
        for (id, collection, path, title, body) in &rows {
            let filepath = normalize_cjk_for_fts(&format!("{collection}/{path}"));
            insert.execute(rusqlite::params![
                id,
                filepath,
                normalize_cjk_for_fts(title),
                normalize_cjk_for_fts(body),
            ])?;
        }
    }
    tx.commit()?;

    conn.execute(
        "INSERT OR REPLACE INTO store_config(key, value) VALUES ('fts_cjk_normalized_version', ?)",
        rusqlite::params![FTS_CJK_NORMALIZED_VERSION],
    )?;

    Ok(())
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_normalisation_spaces_runs() {
        assert_eq!(normalize_cjk_for_fts("hello"), "hello");
        assert_eq!(normalize_cjk_for_fts("日本語"), " 日 本 語 ");
        assert_eq!(
            normalize_cjk_for_fts("hello 日本語 world"),
            "hello  日 本 語  world"
        );
    }

    #[test]
    fn contains_cjk_detects_mixed_text() {
        assert!(!contains_cjk("hello"));
        assert!(contains_cjk("日本"));
        assert!(contains_cjk("hello 日本"));
    }

    // --- sanitize_fts5_term: ported from store.helpers.unit.test.ts
    //     `describe("sanitizeFTS5Term")` (content filter, mirrors TS exactly) ---

    #[test]
    fn sanitize_fts5_term_preserves_snake_case_underscores() {
        assert_eq!(sanitize_fts5_term("my_variable"), "my_variable");
        assert_eq!(sanitize_fts5_term("MAX_RETRIES"), "max_retries");
        assert_eq!(sanitize_fts5_term("__init__"), "__init__");
    }

    #[test]
    fn sanitize_fts5_term_preserves_alphanumeric() {
        assert_eq!(sanitize_fts5_term("hello123"), "hello123");
        assert_eq!(sanitize_fts5_term("test"), "test");
    }

    #[test]
    fn sanitize_fts5_term_preserves_apostrophes() {
        assert_eq!(sanitize_fts5_term("don't"), "don't");
        assert_eq!(sanitize_fts5_term("it's"), "it's");
    }

    #[test]
    fn sanitize_fts5_term_strips_other_punctuation() {
        assert_eq!(sanitize_fts5_term("hello!"), "hello");
        assert_eq!(sanitize_fts5_term("test@value"), "testvalue");
        assert_eq!(sanitize_fts5_term("a.b"), "ab");
    }

    #[test]
    fn sanitize_fts5_term_lowercases_output() {
        assert_eq!(sanitize_fts5_term("Hello"), "hello");
        assert_eq!(sanitize_fts5_term("MY_VAR"), "my_var");
    }

    #[test]
    fn sanitize_fts5_term_handles_unicode_letters_and_numbers() {
        assert_eq!(sanitize_fts5_term("café"), "café");
        assert_eq!(sanitize_fts5_term("日本語"), "日本語");
    }

    #[test]
    fn sanitize_fts5_term_drops_combining_marks_like_ts() {
        // `\p{L}\p{N}` excludes combining marks that `char::is_alphanumeric()`
        // would keep (they have the `Other_Alphabetic` property). U+093E is a
        // Devanagari vowel sign (general category Mc) — TS strips it.
        assert_eq!(sanitize_fts5_term("a\u{093E}"), "a");
    }
}
