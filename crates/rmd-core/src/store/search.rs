//! Document result types and FTS5 lexical search.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 1795–1969 (types) and
//! 3016–3246 (`sanitizeFTS5Term` / `validateLexQuery` / `searchFTS` plus
//! the FTS5 query parser `buildFTS5Query`).
//!
//! The vector-search half (`searchVec`, `getEmbedding`) is LLM-using and
//! deliberately out of scope this pass.

use rusqlite::{types::Value, Connection};

use super::context::get_context_for_file;
use super::docid::get_docid;
use super::schema::{contains_cjk, sanitize_fts5_phrase};
use super::{Error, Result};

// ============================================================================
// Result types (TS lines 1795–1969)
// ============================================================================

/// Source of a search hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSource {
    Fts,
    Vec,
}

/// Unified document result. `body` is optional — callers can defer loading
/// it via [`super::lookup::get_document_body`].
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentResult {
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub context: Option<String>,
    pub hash: String,
    pub docid: String,
    pub collection_name: String,
    pub modified_at: String,
    pub body_length: usize,
    pub body: Option<String>,
}

/// A search hit. Adds score + source on top of [`DocumentResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub doc: DocumentResult,
    pub score: f64,
    pub source: SearchSource,
    pub chunk_pos: Option<i64>,
}

/// Simplified ranked result used by RRF and reranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedResult {
    pub file: String,
    pub display_path: String,
    pub title: String,
    pub body: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentNotFound {
    pub query: String,
    pub similar_files: Vec<String>,
}

/// Result of a multi-get operation. `Skipped` carries minimal metadata
/// (filepath + display_path) and a reason; `Found` carries the full doc.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiGetResult {
    Found(DocumentResult),
    Skipped {
        filepath: String,
        display_path: String,
        skip_reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionInfo {
    pub name: String,
    pub path: Option<String>,
    pub pattern: Option<String>,
    pub documents: i64,
    pub last_updated: String,
}

// ============================================================================
// FTS5 query parsing (TS `buildFTS5Query`, lines 3061–3151)
// ============================================================================

/// Build an FTS5 query from a free-text input. Returns `None` if there is
/// nothing positive to search for (FTS5 `NOT` is a binary operator).
pub(crate) fn build_fts5_query(query: &str) -> Option<String> {
    let s = query.trim();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        let negated = bytes[i] == b'-';
        if negated {
            i += 1;
        }

        if i < bytes.len() && bytes[i] == b'"' {
            let start = i + 1;
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let phrase = &s[start..i];
            if i < bytes.len() {
                i += 1; // skip closing quote
            }
            let sanitised = sanitize_fts5_phrase(phrase);
            if !sanitised.is_empty() {
                let q = format!("\"{sanitised}\"");
                if negated {
                    negative.push(q);
                } else {
                    positive.push(q);
                }
            }
        } else {
            let start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'"' {
                i += 1;
            }
            let term = &s[start..i];

            if is_hyphenated_token(term) {
                let sanitised = sanitize_hyphenated(term);
                if !sanitised.is_empty() {
                    let q = format!("\"{sanitised}\"");
                    if negated {
                        negative.push(q);
                    } else {
                        positive.push(q);
                    }
                }
            } else if contains_cjk(term) {
                let sanitised = sanitize_fts5_phrase(term);
                if !sanitised.is_empty() {
                    let q = format!("\"{sanitised}\"");
                    if negated {
                        negative.push(q);
                    } else {
                        positive.push(q);
                    }
                }
            } else {
                let sanitised = sanitise_term_alnum(term);
                if !sanitised.is_empty() {
                    let q = format!("\"{sanitised}\"*");
                    if negated {
                        negative.push(q);
                    } else {
                        positive.push(q);
                    }
                }
            }
        }
    }

    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" AND ");
    for neg in &negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

fn is_hyphenated_token(term: &str) -> bool {
    if !term.contains('-') {
        return false;
    }
    let bytes = term.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];
    if !first.is_ascii_alphanumeric() || !last.is_ascii_alphanumeric() {
        return false;
    }
    term.chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '\'')
}

fn sanitize_hyphenated(term: &str) -> String {
    term.split('-')
        .map(sanitise_term_alnum)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Stricter term sanitiser that mirrors TS `sanitizeFTS5Term`
/// (`store.ts:3016–3018`): keep letters/digits/`'`/`_`, lowercase the rest.
/// `super::schema::sanitize_fts5_term` is the *quoting* helper; this one is
/// the *content* filter.
fn sanitise_term_alnum(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        if ch.is_alphanumeric() || ch == '\'' || ch == '_' {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

// Re-export the quoting helper under the original TS name for callers
// that want to escape a single term verbatim.
pub use super::schema::sanitize_fts5_term as quote_fts5_term;

// ============================================================================
// validateLexQuery / validateSemanticQuery (refined from schema.rs stubs)
// ============================================================================

/// Mirrors TS `validateLexQuery` (`store.ts:3166–3175`).
pub fn validate_lex_query(query: &str) -> Result<()> {
    if query.contains('\r') || query.contains('\n') {
        return Err(Error::InvalidQuery(
            "Lex queries must be a single line. Remove newline characters or split into separate lex: lines.".into(),
        ));
    }
    let quotes = query.bytes().filter(|&b| b == b'"').count();
    if quotes % 2 == 1 {
        return Err(Error::InvalidQuery(
            "Lex query has an unmatched double quote (\"). Add the closing quote or remove it."
                .into(),
        ));
    }
    Ok(())
}

/// Mirrors TS `validateSemanticQuery` (`store.ts:3157–3164`).
pub fn validate_semantic_query(query: &str) -> Result<()> {
    let bytes = query.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'-' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            let next = bytes.get(i + 1).copied().unwrap_or(0);
            if next.is_ascii_alphanumeric() || next == b'_' || next == b'"' {
                return Err(Error::InvalidQuery(
                    "Negation (-term) is not supported in vec/hyde queries. Use lex for exclusions.".into(),
                ));
            }
        }
    }
    Ok(())
}

// ============================================================================
// searchFTS (TS lines 3177–3246)
// ============================================================================

/// Run a BM25 lexical search via FTS5. Returns hits ordered by relevance
/// (higher score = better).
pub fn search_fts(
    conn: &Connection,
    query: &str,
    limit: Option<usize>,
    collection_name: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let Some(fts_query) = build_fts5_query(query) else {
        return Ok(Vec::new());
    };

    let limit = limit.unwrap_or(20);
    let fts_limit = if collection_name.is_some() {
        limit * 10
    } else {
        limit
    };

    let mut sql = format!(
        r#"WITH fts_matches AS (
               SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) AS bm25_score
               FROM documents_fts
               WHERE documents_fts MATCH ?
               ORDER BY bm25_score ASC
               LIMIT {fts_limit}
           )
           SELECT
               'qmd://' || d.collection || '/' || d.path AS filepath,
               d.collection || '/' || d.path AS display_path,
               d.title,
               content.doc AS body,
               d.hash,
               fm.bm25_score
           FROM fts_matches fm
           JOIN documents d ON d.id = fm.rowid
           JOIN content ON content.hash = d.hash
           WHERE d.active = 1"#,
    );

    let mut bind: Vec<Value> = vec![Value::Text(fts_query)];
    if let Some(c) = collection_name {
        sql.push_str(" AND d.collection = ?");
        bind.push(Value::Text(c.to_string()));
    }
    sql.push_str(" ORDER BY fm.bm25_score ASC LIMIT ?");
    bind.push(Value::Integer(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(bind.iter()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>();

    let mut out = Vec::with_capacity(rows.len());
    for (filepath, display_path, title, body, hash, bm25) in rows {
        let abs = bm25.abs();
        let score = abs / (1.0 + abs);
        let collection_name = filepath
            .strip_prefix("qmd://")
            .and_then(|s| s.split_once('/').map(|(c, _)| c.to_string()))
            .unwrap_or_default();
        let context = get_context_for_file(conn, &filepath).ok().flatten();
        let body_length = body.len();
        out.push(SearchResult {
            doc: DocumentResult {
                filepath,
                display_path,
                title,
                context,
                hash: hash.clone(),
                docid: get_docid(&hash),
                collection_name,
                modified_at: String::new(),
                body_length,
                body: Some(body),
            },
            score,
            source: SearchSource::Fts,
            chunk_pos: None,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_lex_rejects_newlines() {
        assert!(validate_lex_query("a\nb").is_err());
        assert!(validate_lex_query("ok").is_ok());
    }

    #[test]
    fn validate_lex_rejects_unbalanced_quotes() {
        assert!(validate_lex_query("\"open").is_err());
        assert!(validate_lex_query("\"closed\"").is_ok());
    }

    #[test]
    fn validate_semantic_rejects_negation() {
        assert!(validate_semantic_query("-bad").is_err());
        assert!(validate_semantic_query("hello -bad").is_err());
        // Hyphenated mid-word is fine.
        assert!(validate_semantic_query("real-time updates").is_ok());
    }

    #[test]
    fn build_fts5_plain_terms() {
        assert_eq!(build_fts5_query("hello").as_deref(), Some("\"hello\"*"));
        assert_eq!(
            build_fts5_query("hello world").as_deref(),
            Some("\"hello\"* AND \"world\"*")
        );
    }

    #[test]
    fn build_fts5_negation_needs_positive() {
        assert!(build_fts5_query("-only").is_none());
        assert_eq!(
            build_fts5_query("good -bad").as_deref(),
            Some("\"good\"* NOT \"bad\"*")
        );
    }

    #[test]
    fn build_fts5_phrase() {
        assert_eq!(
            build_fts5_query("\"machine learning\"").as_deref(),
            Some("\"machine learning\"")
        );
    }

    #[test]
    fn build_fts5_hyphenated() {
        let q = build_fts5_query("multi-agent").unwrap();
        assert_eq!(q, "\"multi agent\"");
    }
}
