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
use super::schema::{contains_cjk, sanitize_fts5_phrase, sanitize_fts5_term};
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
                let sanitised = sanitize_fts5_term(term);
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
        .map(sanitize_fts5_term)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

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

    /// Display string of a validator error, for `.contains(...)` parity with
    /// the TS `.toContain(...)` assertions.
    fn err_msg(r: Result<()>) -> String {
        r.unwrap_err().to_string()
    }

    // =========================================================================
    // validateSemanticQuery — ported from structured-search.test.ts
    // `describe("validateSemanticQuery")` (lines 357-431). TS returns null on
    // accept / a string containing "Negation" on reject; we mirror with
    // is_ok() / err message contains "Negation".
    // =========================================================================

    #[test]
    fn semantic_accepts_plain_natural_language() {
        assert!(validate_semantic_query("how does error handling work").is_ok());
        assert!(validate_semantic_query("what is the CAP theorem").is_ok());
    }

    #[test]
    fn semantic_rejects_negation_at_start() {
        assert!(err_msg(validate_semantic_query("-redis connection pooling")).contains("Negation"));
    }

    #[test]
    fn semantic_rejects_negation_after_space() {
        assert!(err_msg(validate_semantic_query("performance -sports")).contains("Negation"));
    }

    #[test]
    fn semantic_rejects_negated_quoted_phrase() {
        assert!(err_msg(validate_semantic_query("-\"exact phrase\"")).contains("Negation"));
    }

    #[test]
    fn semantic_rejects_multiple_negations() {
        assert!(
            err_msg(validate_semantic_query("error handling -java -python")).contains("Negation")
        );
    }

    #[test]
    fn semantic_rejects_negation_after_leading_whitespace() {
        assert!(err_msg(validate_semantic_query("  -term at start")).contains("Negation"));
    }

    #[test]
    fn semantic_rejects_negation_after_tab() {
        assert!(err_msg(validate_semantic_query("foo\t-bar")).contains("Negation"));
    }

    #[test]
    fn semantic_accepts_hyphenated_compound_words() {
        assert!(validate_semantic_query("long-lived server shared across clients").is_ok());
        assert!(validate_semantic_query("real-time voice processing pipeline").is_ok());
        assert!(validate_semantic_query("how does the rate-limiter handle burst traffic").is_ok());
        assert!(validate_semantic_query("self-hosted deployment options").is_ok());
        assert!(validate_semantic_query("multi-client session architecture").is_ok());
        assert!(validate_semantic_query("cross-platform compatibility").is_ok());
        assert!(validate_semantic_query("non-blocking I/O model").is_ok());
        assert!(validate_semantic_query("in-memory caching strategy").is_ok());
        assert!(validate_semantic_query("write-ahead log for crash recovery").is_ok());
        assert!(validate_semantic_query("copy-on-write semantics").is_ok());
    }

    #[test]
    fn semantic_accepts_multiple_hyphens_in_a_phrase() {
        assert!(validate_semantic_query("state-of-the-art embedding models").is_ok());
        assert!(validate_semantic_query("end-to-end testing").is_ok());
        assert!(validate_semantic_query("man-in-the-middle attack prevention").is_ok());
    }

    #[test]
    fn semantic_accepts_multiple_hyphenated_words_in_one_query() {
        assert!(validate_semantic_query("built-in vs add-on features").is_ok());
    }

    #[test]
    fn semantic_accepts_short_hyphenated_terms() {
        assert!(validate_semantic_query("A-B testing for ML models").is_ok());
        assert!(validate_semantic_query("e-commerce platform").is_ok());
    }

    #[test]
    fn semantic_accepts_bare_hyphen_without_word_character() {
        assert!(validate_semantic_query("-").is_ok());
    }

    #[test]
    fn semantic_accepts_hyde_style_hypothetical_answers() {
        assert!(validate_semantic_query(
            "The CAP theorem states that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance."
        )
        .is_ok());
    }

    #[test]
    fn semantic_accepts_hyde_with_hyphenated_words() {
        assert!(validate_semantic_query(
            "HTTP transport runs a single long-lived daemon shared across all clients, avoiding per-session model re-loading."
        )
        .is_ok());
    }

    // =========================================================================
    // validateLexQuery — ported from structured-search.test.ts
    // `describe("validateLexQuery")` (lines 433-445).
    // =========================================================================

    #[test]
    fn lex_accepts_basic_query() {
        assert!(validate_lex_query("auth token").is_ok());
    }

    #[test]
    fn lex_rejects_newline() {
        assert!(err_msg(validate_lex_query("foo\nbar")).contains("single line"));
    }

    #[test]
    fn lex_rejects_unmatched_quote() {
        assert!(err_msg(validate_lex_query("\"unfinished")).contains("unmatched"));
    }

    // =========================================================================
    // buildFTS5Query — ported from structured-search.test.ts
    // `describe("buildFTS5Query (lex parser)")` (lines 452-594).
    // =========================================================================

    #[test]
    fn build_fts5_plain_terms_and() {
        assert_eq!(
            build_fts5_query("foo bar").as_deref(),
            Some("\"foo\"* AND \"bar\"*")
        );
    }

    #[test]
    fn build_fts5_single_term() {
        assert_eq!(
            build_fts5_query("performance").as_deref(),
            Some("\"performance\"*")
        );
    }

    #[test]
    fn build_fts5_quoted_phrase_exact() {
        assert_eq!(
            build_fts5_query("\"machine learning\"").as_deref(),
            Some("\"machine learning\"")
        );
    }

    #[test]
    fn build_fts5_quoted_phrase_mixed_case_sanitized() {
        assert_eq!(
            build_fts5_query("\"C++ performance\"").as_deref(),
            Some("\"c performance\"")
        );
    }

    #[test]
    fn build_fts5_negation_of_term() {
        assert_eq!(
            build_fts5_query("performance -sports").as_deref(),
            Some("\"performance\"* NOT \"sports\"*")
        );
    }

    #[test]
    fn build_fts5_negation_of_phrase() {
        assert_eq!(
            build_fts5_query("performance -\"sports athlete\"").as_deref(),
            Some("\"performance\"* NOT \"sports athlete\"")
        );
    }

    #[test]
    fn build_fts5_multiple_negations() {
        assert_eq!(
            build_fts5_query("performance -sports -athlete").as_deref(),
            Some("\"performance\"* NOT \"sports\"* NOT \"athlete\"*")
        );
    }

    #[test]
    fn build_fts5_quoted_positive_plus_negation() {
        assert_eq!(
            build_fts5_query("\"machine learning\" -sports -athlete").as_deref(),
            Some("\"machine learning\" NOT \"sports\"* NOT \"athlete\"*")
        );
    }

    #[test]
    fn build_fts5_intent_aware_cpp_example() {
        let result = build_fts5_query("\"C++ performance\" optimization -sports -athlete").unwrap();
        assert!(result.contains("NOT \"sports\"*"));
        assert!(result.contains("NOT \"athlete\"*"));
        assert!(result.contains("\"optimization\"*"));
    }

    #[test]
    fn build_fts5_only_negations_is_none() {
        assert!(build_fts5_query("-sports -athlete").is_none());
    }

    #[test]
    fn build_fts5_empty_is_none() {
        assert!(build_fts5_query("").is_none());
        assert!(build_fts5_query("   ").is_none());
    }

    #[test]
    fn build_fts5_special_chars_stripped() {
        assert_eq!(
            build_fts5_query("hello!world").as_deref(),
            Some("\"helloworld\"*")
        );
    }

    #[test]
    fn build_fts5_hyphenated_term_phrase() {
        assert_eq!(
            build_fts5_query("multi-agent").as_deref(),
            Some("\"multi agent\"")
        );
    }

    #[test]
    fn build_fts5_hyphenated_identifier_phrase() {
        assert_eq!(
            build_fts5_query("DEC-0054").as_deref(),
            Some("\"dec 0054\"")
        );
    }

    #[test]
    fn build_fts5_hyphenated_model_name_phrase() {
        assert_eq!(build_fts5_query("gpt-4").as_deref(), Some("\"gpt 4\""));
    }

    #[test]
    fn build_fts5_multi_hyphen_phrase() {
        assert_eq!(
            build_fts5_query("foo-bar-baz").as_deref(),
            Some("\"foo bar baz\"")
        );
    }

    #[test]
    fn build_fts5_hyphenated_mixed_with_plain() {
        assert_eq!(
            build_fts5_query("multi-agent memory").as_deref(),
            Some("\"multi agent\" AND \"memory\"*")
        );
    }

    #[test]
    fn build_fts5_negation_alongside_hyphenated() {
        assert_eq!(
            build_fts5_query("multi-agent -sports").as_deref(),
            Some("\"multi agent\" NOT \"sports\"*")
        );
    }

    #[test]
    fn build_fts5_negated_hyphenated_term() {
        assert_eq!(
            build_fts5_query("performance -multi-agent").as_deref(),
            Some("\"performance\"* NOT \"multi agent\"")
        );
    }

    #[test]
    fn build_fts5_plain_negation_not_confused_with_hyphen() {
        assert_eq!(
            build_fts5_query("performance -sports").as_deref(),
            Some("\"performance\"* NOT \"sports\"*")
        );
    }
}
