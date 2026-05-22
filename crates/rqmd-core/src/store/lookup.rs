//! Document lookup: by name, by docid, by glob, and similar-file search.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 2566–2617 + 3720–3983.

use globset::Glob;
use rusqlite::{params, Connection, OptionalExtension};

use super::context::get_context_for_file;
use super::docid::{find_document_by_docid, get_docid, is_docid};
use super::path::homedir;
use super::search::{DocumentNotFound, DocumentResult, MultiGetResult};
use super::store_config::get_store_collections;
use super::DEFAULT_MULTI_GET_MAX_BYTES;
use super::{Error, Result};

// ============================================================================
// Outcomes
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum FindDocumentOutcome {
    Found(DocumentResult),
    NotFound(DocumentNotFound),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FindDocumentsResult {
    pub docs: Vec<MultiGetResult>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FindDocumentOptions {
    pub include_body: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct FindDocumentsOptions {
    pub include_body: bool,
    pub max_bytes: usize,
}

impl Default for FindDocumentsOptions {
    fn default() -> Self {
        Self {
            include_body: false,
            max_bytes: DEFAULT_MULTI_GET_MAX_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileMatch {
    pub filepath: String,
    pub display_path: String,
    pub body_length: usize,
}

// ============================================================================
// findDocument
// ============================================================================

struct DocRow {
    virtual_path: String,
    display_path: String,
    title: String,
    hash: String,
    collection: String,
    modified_at: String,
    body_length: i64,
    body: Option<String>,
}

fn map_doc_row(row: &rusqlite::Row<'_>, include_body: bool) -> rusqlite::Result<DocRow> {
    Ok(DocRow {
        virtual_path: row.get(0)?,
        display_path: row.get(1)?,
        title: row.get(2)?,
        hash: row.get(3)?,
        collection: row.get(4)?,
        modified_at: row.get(5)?,
        body_length: row.get(6)?,
        body: if include_body {
            Some(row.get(7)?)
        } else {
            None
        },
    })
}

fn doc_select_cols(include_body: bool) -> &'static str {
    if include_body {
        "'qmd://' || d.collection || '/' || d.path AS virtual_path,
         d.collection || '/' || d.path AS display_path,
         d.title, d.hash, d.collection, d.modified_at,
         LENGTH(content.doc) AS body_length,
         content.doc AS body"
    } else {
        "'qmd://' || d.collection || '/' || d.path AS virtual_path,
         d.collection || '/' || d.path AS display_path,
         d.title, d.hash, d.collection, d.modified_at,
         LENGTH(content.doc) AS body_length"
    }
}

pub fn find_document(
    conn: &Connection,
    filename: &str,
    options: FindDocumentOptions,
) -> Result<FindDocumentOutcome> {
    let original = filename.to_string();

    // Strip trailing `:NN` line suffix.
    let mut filepath = filename.to_string();
    if let Some(idx) = filepath.rfind(':') {
        let suffix = &filepath[idx + 1..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            filepath.truncate(idx);
        }
    }

    // Docid shortcut.
    if is_docid(&filepath) {
        match find_document_by_docid(conn, &filepath)? {
            Some(d) => filepath = d.filepath,
            None => {
                return Ok(FindDocumentOutcome::NotFound(DocumentNotFound {
                    query: original,
                    similar_files: Vec::new(),
                }));
            }
        }
    }

    // `~/` expansion.
    if let Some(rest) = filepath.strip_prefix("~/") {
        filepath = format!("{}/{rest}", homedir().display());
    }

    let cols = doc_select_cols(options.include_body);

    // 1. Exact virtual-path match.
    let row = conn
        .query_row(
            &format!(
                "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
                 WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1"
            ),
            params![filepath],
            |r| map_doc_row(r, options.include_body),
        )
        .optional()?;

    // 2. Suffix LIKE match.
    let row = match row {
        Some(r) => Some(r),
        None => conn
            .query_row(
                &format!(
                    "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
                     WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
                     LIMIT 1"
                ),
                params![format!("%{filepath}")],
                |r| map_doc_row(r, options.include_body),
            )
            .optional()?,
    };

    // 3. Match by absolute path against known collection roots.
    let row = match row {
        Some(r) => Some(r),
        None if !filepath.starts_with("qmd://") => {
            // Normalize separators so absolute Windows paths (backslashes) match
            // collection roots stored with the OS separator. Mirrors qmd's
            // `normalizePathSeparators` (store.ts:383); compare normalized forms
            // only — stored data is untouched, and the derived `rel` uses `/`
            // like the handelized `documents.path`.
            let norm_filepath = filepath.replace('\\', "/");
            let mut found = None;
            for coll in get_store_collections(conn)? {
                let norm_coll = coll.path.replace('\\', "/");
                let rel =
                    if let Some(stripped) = norm_filepath.strip_prefix(&format!("{norm_coll}/")) {
                        Some(stripped.to_string())
                    } else if !norm_filepath.starts_with('/') {
                        Some(norm_filepath.clone())
                    } else {
                        None
                    };
                if let Some(rel) = rel {
                    let r = conn
                        .query_row(
                            &format!(
                                "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
                                 WHERE d.collection = ? AND d.path = ? AND d.active = 1"
                            ),
                            params![coll.name, rel],
                            |r| map_doc_row(r, options.include_body),
                        )
                        .optional()?;
                    if let Some(r) = r {
                        found = Some(r);
                        break;
                    }
                }
            }
            found
        }
        None => None,
    };

    let Some(doc) = row else {
        let similar = find_similar_files(conn, &filepath, Some(5), Some(5))?;
        return Ok(FindDocumentOutcome::NotFound(DocumentNotFound {
            query: original,
            similar_files: similar,
        }));
    };

    let virtual_path = if doc.virtual_path.is_empty() {
        format!("qmd://{}/{}", doc.collection, doc.display_path)
    } else {
        doc.virtual_path
    };
    let context = get_context_for_file(conn, &virtual_path)?;

    Ok(FindDocumentOutcome::Found(DocumentResult {
        filepath: virtual_path,
        display_path: doc.display_path,
        title: doc.title,
        context,
        hash: doc.hash.clone(),
        docid: get_docid(&doc.hash),
        collection_name: doc.collection,
        modified_at: doc.modified_at,
        body_length: doc.body_length as usize,
        body: doc.body,
    }))
}

// ============================================================================
// getDocumentBody
// ============================================================================

pub fn get_document_body(
    conn: &Connection,
    filepath: &str,
    from_line: Option<usize>,
    max_lines: Option<usize>,
) -> Result<Option<String>> {
    let mut row: Option<String> = None;

    if filepath.starts_with("qmd://") {
        row = conn
            .query_row(
                "SELECT content.doc FROM documents d JOIN content ON content.hash = d.hash
                 WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1",
                params![filepath],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
    }

    if row.is_none() {
        for coll in get_store_collections(conn)? {
            let prefix = format!("{}/", coll.path);
            if let Some(rel) = filepath.strip_prefix(&prefix) {
                row = conn
                    .query_row(
                        "SELECT content.doc FROM documents d JOIN content ON content.hash = d.hash
                         WHERE d.collection = ? AND d.path = ? AND d.active = 1",
                        params![coll.name, rel],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?;
                if row.is_some() {
                    break;
                }
            }
        }
    }

    let Some(body) = row else { return Ok(None) };

    if from_line.is_some() || max_lines.is_some() {
        let lines: Vec<&str> = body.split('\n').collect();
        let start = from_line.unwrap_or(1).saturating_sub(1);
        let end = match max_lines {
            Some(n) => (start + n).min(lines.len()),
            None => lines.len(),
        };
        if start >= lines.len() {
            return Ok(Some(String::new()));
        }
        return Ok(Some(lines[start..end].join("\n")));
    }

    Ok(Some(body))
}

// ============================================================================
// findDocuments
// ============================================================================

pub fn find_documents(
    conn: &Connection,
    pattern: &str,
    options: FindDocumentsOptions,
) -> Result<FindDocumentsResult> {
    let mut errors: Vec<String> = Vec::new();
    let mut rows: Vec<DocRow> = Vec::new();

    let is_comma_separated = pattern.contains(',')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('{');
    let cols = doc_select_cols(options.include_body);

    if is_comma_separated {
        let names: Vec<&str> = pattern
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        for name in &names {
            let mut row = conn
                .query_row(
                    &format!(
                        "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
                         WHERE 'qmd://' || d.collection || '/' || d.path = ? AND d.active = 1"
                    ),
                    params![name],
                    |r| map_doc_row(r, options.include_body),
                )
                .optional()?;
            if row.is_none() {
                row = conn
                    .query_row(
                        &format!(
                            "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
                             WHERE 'qmd://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
                             LIMIT 1"
                        ),
                        params![format!("%{name}")],
                        |r| map_doc_row(r, options.include_body),
                    )
                    .optional()?;
            }
            if let Some(r) = row {
                rows.push(r);
            } else {
                let similar = find_similar_files(conn, name, Some(5), Some(3))?;
                let mut msg = format!("File not found: {name}");
                if !similar.is_empty() {
                    msg.push_str(&format!(" (did you mean: {}?)", similar.join(", ")));
                }
                errors.push(msg);
            }
        }
    } else {
        let matches = match_files_by_glob(conn, pattern)?;
        if matches.is_empty() {
            errors.push(format!("No files matched pattern: {pattern}"));
            return Ok(FindDocumentsResult {
                docs: vec![],
                errors,
            });
        }
        let virtual_paths: Vec<String> = matches.iter().map(|m| m.filepath.clone()).collect();
        let placeholders = vec!["?"; virtual_paths.len()].join(",");
        let sql = format!(
            "SELECT {cols} FROM documents d JOIN content ON content.hash = d.hash
             WHERE 'qmd://' || d.collection || '/' || d.path IN ({placeholders})
               AND d.active = 1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt
            .query_map(rusqlite::params_from_iter(virtual_paths.iter()), |r| {
                map_doc_row(r, options.include_body)
            })?
            .filter_map(|r| r.ok());
        rows.extend(mapped);
    }

    let mut docs: Vec<MultiGetResult> = Vec::new();
    for row in rows {
        let virtual_path = if row.virtual_path.is_empty() {
            format!("qmd://{}/{}", row.collection, row.display_path)
        } else {
            row.virtual_path
        };
        let context = get_context_for_file(conn, &virtual_path)?;

        if (row.body_length as usize) > options.max_bytes {
            docs.push(MultiGetResult::Skipped {
                filepath: virtual_path,
                display_path: row.display_path,
                // Round to nearest KB to match qmd's `Math.round(bytes / 1024)`
                // (store.ts:3960). `f64::round` rounds half away from zero, which
                // equals JS `Math.round` for these non-negative byte counts.
                skip_reason: format!(
                    "File too large ({}KB > {}KB)",
                    (row.body_length as f64 / 1024.0).round() as i64,
                    (options.max_bytes as f64 / 1024.0).round() as i64
                ),
            });
            continue;
        }

        let title = if row.title.is_empty() {
            row.display_path
                .rsplit_once('/')
                .map(|(_, last)| last.to_string())
                .unwrap_or_else(|| row.display_path.clone())
        } else {
            row.title
        };

        docs.push(MultiGetResult::Found(DocumentResult {
            filepath: virtual_path,
            display_path: row.display_path,
            title,
            context,
            hash: row.hash.clone(),
            docid: get_docid(&row.hash),
            collection_name: row.collection,
            modified_at: row.modified_at,
            body_length: row.body_length as usize,
            body: row.body,
        }));
    }

    Ok(FindDocumentsResult { docs, errors })
}

// ============================================================================
// findSimilarFiles (Levenshtein)
// ============================================================================

pub fn find_similar_files(
    conn: &Connection,
    query: &str,
    max_distance: Option<usize>,
    limit: Option<usize>,
) -> Result<Vec<String>> {
    let max_distance = max_distance.unwrap_or(3);
    let limit = limit.unwrap_or(5);

    let mut stmt = conn.prepare("SELECT d.path FROM documents d WHERE d.active = 1")?;
    let paths: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    let q = query.to_ascii_lowercase();
    let mut scored: Vec<(String, usize)> = paths
        .into_iter()
        .map(|p| {
            let d = levenshtein(&p.to_ascii_lowercase(), &q);
            (p, d)
        })
        .filter(|(_, d)| *d <= max_distance)
        .collect();
    scored.sort_by_key(|(_, d)| *d);
    scored.truncate(limit);
    Ok(scored.into_iter().map(|(p, _)| p).collect())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

// ============================================================================
// matchFilesByGlob
// ============================================================================

pub fn match_files_by_glob(conn: &Connection, pattern: &str) -> Result<Vec<FileMatch>> {
    let glob = Glob::new(pattern)
        .map_err(|e| Error::InvalidGlob(format!("{pattern}: {e}")))?
        .compile_matcher();

    let mut stmt = conn.prepare(
        "SELECT 'qmd://' || d.collection || '/' || d.path AS virtual_path,
                LENGTH(content.doc) AS body_length,
                d.path,
                d.collection
         FROM documents d
         JOIN content ON content.hash = d.hash
         WHERE d.active = 1",
    )?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?
        .filter_map(|r| r.ok());

    let mut out: Vec<FileMatch> = Vec::new();
    for (virtual_path, body_length, path, collection) in rows {
        let combined = format!("{collection}/{path}");
        if glob.is_match(&virtual_path) || glob.is_match(&path) || glob.is_match(&combined) {
            out.push(FileMatch {
                filepath: virtual_path,
                display_path: path,
                body_length: body_length as usize,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }
}
