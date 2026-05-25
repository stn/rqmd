//! Output formatting for `get` and `multi-get`.
//!
//! Maps to qmd's `src/cli/formatter.ts` (the CLI/JSON/CSV/MD/XML/files
//! branches in `multiGet`, `qmd.ts:1262–1336`, plus the single-document
//! formatters `documentTo*` / `formatDocument`, `formatter.ts:314–376`).
//!
//! Each `fmt_*` returns the complete stdout text (including trailing
//! newlines) and the thin `write_*` wrappers just `print!` it — this keeps a
//! single source of truth that is unit-testable.

use rqmd_core::store::search::{DocumentResult, MultiGetResult};
use serde_json::{Value, json};

use crate::cli::FormatFlags;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Cli,
    Json,
    Csv,
    Md,
    Xml,
    Files,
}

impl From<&FormatFlags> for OutputFormat {
    fn from(f: &FormatFlags) -> Self {
        if f.json {
            OutputFormat::Json
        } else if f.csv {
            OutputFormat::Csv
        } else if f.md {
            OutputFormat::Md
        } else if f.xml {
            OutputFormat::Xml
        } else if f.files {
            OutputFormat::Files
        } else {
            OutputFormat::Cli
        }
    }
}

pub fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Apply a max-lines truncation, appending the `[... truncated N more lines]`
/// marker qmd uses (qmd.ts:1241–1247).
pub fn truncate_lines(body: &str, max_lines: Option<usize>) -> String {
    let Some(max) = max_lines else {
        return body.to_string();
    };
    let lines: Vec<&str> = body.split('\n').collect();
    if lines.len() <= max {
        return body.to_string();
    }
    let kept: String = lines[..max].join("\n");
    let extra = lines.len() - max;
    format!("{kept}\n\n[... truncated {extra} more lines]")
}

// ============================================================================
// multi-get formatters
// ============================================================================

/// Write `multi-get` results in the requested format.
pub fn write_multi_get(results: &[MultiGetResult], max_lines: Option<usize>, format: OutputFormat) {
    print!("{}", fmt_multi_get(results, max_lines, format));
}

/// Format `multi-get` results to a string (dispatcher over the per-format
/// `fmt_*` helpers).
pub fn fmt_multi_get(
    results: &[MultiGetResult],
    max_lines: Option<usize>,
    format: OutputFormat,
) -> String {
    match format {
        OutputFormat::Json => fmt_json(results, max_lines),
        OutputFormat::Csv => fmt_csv(results, max_lines),
        OutputFormat::Files => fmt_files(results),
        OutputFormat::Md => fmt_md(results, max_lines),
        OutputFormat::Xml => fmt_xml(results, max_lines),
        OutputFormat::Cli => fmt_cli(results, max_lines),
    }
}

fn title_of(doc: &DocumentResult) -> String {
    if !doc.title.is_empty() {
        doc.title.clone()
    } else {
        doc.display_path
            .rsplit_once('/')
            .map(|(_, last)| last.to_string())
            .unwrap_or_else(|| doc.display_path.clone())
    }
}

fn skipped_title(display_path: &str) -> String {
    display_path
        .rsplit_once('/')
        .map(|(_, l)| l.to_string())
        .unwrap_or_else(|| display_path.to_string())
}

fn fmt_json(results: &[MultiGetResult], max_lines: Option<usize>) -> String {
    let arr: Vec<Value> = results
        .iter()
        .map(|r| match r {
            MultiGetResult::Found(doc) => {
                let mut obj = json!({
                    "file": doc.filepath,
                    "title": title_of(doc),
                    "body": truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines),
                });
                if let Some(ctx) = &doc.context {
                    obj["context"] = json!(ctx);
                }
                obj
            }
            MultiGetResult::Skipped {
                filepath,
                display_path,
                skip_reason,
            } => json!({
                "file": filepath,
                "title": skipped_title(display_path),
                "skipped": true,
                "reason": skip_reason,
            }),
        })
        .collect();
    let s = serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string());
    format!("{s}\n")
}

fn fmt_csv(results: &[MultiGetResult], max_lines: Option<usize>) -> String {
    let mut out = String::from("file,title,context,skipped,body\n");
    for r in results {
        let row = match r {
            MultiGetResult::Found(doc) => [
                escape_csv(&doc.filepath),
                escape_csv(&title_of(doc)),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                "false".to_string(),
                escape_csv(&truncate_lines(
                    doc.body.as_deref().unwrap_or(""),
                    max_lines,
                )),
            ],
            MultiGetResult::Skipped {
                filepath,
                display_path,
                skip_reason,
            } => [
                escape_csv(filepath),
                escape_csv(&skipped_title(display_path)),
                String::new(),
                "true".to_string(),
                escape_csv(skip_reason),
            ],
        };
        out.push_str(&row.join(","));
        out.push('\n');
    }
    out
}

fn fmt_files(results: &[MultiGetResult]) -> String {
    let mut out = String::new();
    for r in results {
        match r {
            MultiGetResult::Found(doc) => {
                if let Some(ctx) = &doc.context {
                    let esc = ctx.replace('"', "\"\"");
                    out.push_str(&format!("{},\"{esc}\"\n", doc.filepath));
                } else {
                    out.push_str(&format!("{}\n", doc.filepath));
                }
            }
            MultiGetResult::Skipped { filepath, .. } => {
                out.push_str(&format!("{filepath},[SKIPPED]\n"));
            }
        }
    }
    out
}

fn fmt_md(results: &[MultiGetResult], max_lines: Option<usize>) -> String {
    let mut out = String::new();
    for r in results {
        match r {
            MultiGetResult::Found(doc) => {
                out.push_str(&format!("## {}\n\n", doc.filepath));
                let title = title_of(doc);
                if title != doc.filepath {
                    out.push_str(&format!("**Title:** {title}\n\n"));
                }
                if let Some(ctx) = &doc.context {
                    out.push_str(&format!("**Context:** {ctx}\n\n"));
                }
                out.push_str("```\n");
                out.push_str(&format!(
                    "{}\n",
                    truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines)
                ));
                out.push_str("```\n\n");
            }
            MultiGetResult::Skipped {
                filepath,
                skip_reason,
                ..
            } => {
                out.push_str(&format!("## {filepath}\n\n"));
                out.push_str(&format!("> {skip_reason}\n\n"));
            }
        }
    }
    out
}

fn fmt_xml(results: &[MultiGetResult], max_lines: Option<usize>) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<documents>\n");
    for r in results {
        out.push_str("  <document>\n");
        match r {
            MultiGetResult::Found(doc) => {
                out.push_str(&format!("    <file>{}</file>\n", escape_xml(&doc.filepath)));
                out.push_str(&format!(
                    "    <title>{}</title>\n",
                    escape_xml(&title_of(doc))
                ));
                if let Some(ctx) = &doc.context {
                    out.push_str(&format!("    <context>{}</context>\n", escape_xml(ctx)));
                }
                out.push_str(&format!(
                    "    <body>{}</body>\n",
                    escape_xml(&truncate_lines(
                        doc.body.as_deref().unwrap_or(""),
                        max_lines
                    ))
                ));
            }
            MultiGetResult::Skipped {
                filepath,
                skip_reason,
                ..
            } => {
                out.push_str(&format!("    <file>{}</file>\n", escape_xml(filepath)));
                out.push_str("    <skipped>true</skipped>\n");
                out.push_str(&format!(
                    "    <reason>{}</reason>\n",
                    escape_xml(skip_reason)
                ));
            }
        }
        out.push_str("  </document>\n");
    }
    out.push_str("</documents>\n");
    out
}

fn fmt_cli(results: &[MultiGetResult], max_lines: Option<usize>) -> String {
    let bar = "=".repeat(60);
    let mut out = String::new();
    for r in results {
        match r {
            MultiGetResult::Found(doc) => {
                out.push_str(&format!("\n{bar}\n"));
                out.push_str(&format!("File: {}\n", doc.filepath));
                out.push_str(&format!("{bar}\n\n"));
                if let Some(ctx) = &doc.context {
                    out.push_str(&format!("Folder Context: {ctx}\n---\n\n"));
                }
                out.push_str(&format!(
                    "{}\n",
                    truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines)
                ));
            }
            MultiGetResult::Skipped {
                filepath,
                skip_reason,
                ..
            } => {
                out.push_str(&format!("\n{bar}\n"));
                out.push_str(&format!("File: {filepath}\n"));
                out.push_str(&format!("{bar}\n\n"));
                out.push_str(&format!("[SKIPPED: {skip_reason}]\n"));
            }
        }
    }
    out
}

// ============================================================================
// single-document formatters
//
// Maps to qmd `documentToJson` / `documentToMarkdown` / `documentToXml` /
// `formatDocument` (`formatter.ts:314–376`). JSON keys and XML element names
// follow qmd's camelCase (`modifiedAt`, `bodyLength`).
//
// As in qmd (where `formatDocument` has no production caller — only tests),
// these are unwired: `rqmd get` mirrors qmd's plain CLI output and exposes no
// format flags. Kept + tested for parity; `#[allow(dead_code)]` until a caller
// (e.g. a future structured `get`) needs them.
// ============================================================================

#[allow(dead_code)]
pub fn document_to_json(doc: &DocumentResult) -> String {
    let mut obj = json!({
        "file": doc.display_path,
        "title": doc.title,
        "hash": doc.hash,
        "modifiedAt": doc.modified_at,
        "bodyLength": doc.body_length,
    });
    if let Some(ctx) = &doc.context {
        obj["context"] = json!(ctx);
    }
    if let Some(body) = &doc.body {
        obj["body"] = json!(body);
    }
    serde_json::to_string_pretty(&obj).unwrap_or_else(|_| "{}".to_string())
}

#[allow(dead_code)]
pub fn document_to_md(doc: &DocumentResult) -> String {
    let heading = if doc.title.is_empty() {
        doc.display_path.as_str()
    } else {
        doc.title.as_str()
    };
    let mut md = format!("# {heading}\n\n");
    if let Some(ctx) = &doc.context {
        md.push_str(&format!("**Context:** {ctx}\n\n"));
    }
    md.push_str(&format!("**File:** {}\n", doc.display_path));
    md.push_str(&format!("**Modified:** {}\n\n", doc.modified_at));
    if let Some(body) = &doc.body {
        md.push_str(&format!("---\n\n{body}\n"));
    }
    md
}

#[allow(dead_code)]
pub fn document_to_xml(doc: &DocumentResult) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<document>\n");
    xml.push_str(&format!(
        "  <file>{}</file>\n",
        escape_xml(&doc.display_path)
    ));
    xml.push_str(&format!("  <title>{}</title>\n", escape_xml(&doc.title)));
    if let Some(ctx) = &doc.context {
        xml.push_str(&format!("  <context>{}</context>\n", escape_xml(ctx)));
    }
    xml.push_str(&format!("  <hash>{}</hash>\n", escape_xml(&doc.hash)));
    xml.push_str(&format!(
        "  <modifiedAt>{}</modifiedAt>\n",
        escape_xml(&doc.modified_at)
    ));
    xml.push_str(&format!("  <bodyLength>{}</bodyLength>\n", doc.body_length));
    if let Some(body) = &doc.body {
        xml.push_str(&format!("  <body>{}</body>\n", escape_xml(body)));
    }
    xml.push_str("</document>");
    xml
}

/// Format a single document to the requested format. `json`/`md`/`xml` are
/// handled; everything else falls back to markdown (qmd `formatDocument`
/// default, `formatter.ts:364–376`).
#[allow(dead_code)]
pub fn format_document(doc: &DocumentResult, format: OutputFormat) -> String {
    match format {
        OutputFormat::Json => document_to_json(doc),
        OutputFormat::Xml => document_to_xml(doc),
        _ => document_to_md(doc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CONTEXT: &str = "Internal engineering keynotes from company summit events";

    fn make_doc(context: Option<&str>) -> DocumentResult {
        DocumentResult {
            filepath: "qmd://archive/summit/keynote.md".to_string(),
            display_path: "archive/summit/keynote.md".to_string(),
            title: "Summit Keynote".to_string(),
            context: context.map(String::from),
            hash: "dc5590abcdef".to_string(),
            docid: "dc5590".to_string(),
            collection_name: "archive".to_string(),
            modified_at: "2024-01-01T00:00:00Z".to_string(),
            body_length: 100,
            body: Some(
                "---\ntitle: Summit Keynote\n---\n\nThis is the keynote content.".to_string(),
            ),
        }
    }

    fn make_found(context: Option<&str>) -> MultiGetResult {
        MultiGetResult::Found(make_doc(context))
    }

    // ---- multi-get: context in every format (qmd group C) ----

    #[test]
    fn multi_get_json_includes_context() {
        let out = fmt_json(&[make_found(Some(TEST_CONTEXT))], None);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed[0]["context"], TEST_CONTEXT);
    }

    #[test]
    fn multi_get_csv_includes_context() {
        let out = fmt_csv(&[make_found(Some(TEST_CONTEXT))], None);
        assert!(out.lines().next().unwrap().contains("context"));
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn multi_get_files_includes_context() {
        let out = fmt_files(&[make_found(Some(TEST_CONTEXT))]);
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn multi_get_md_includes_context() {
        let out = fmt_md(&[make_found(Some(TEST_CONTEXT))], None);
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn multi_get_xml_includes_context() {
        let out = fmt_xml(&[make_found(Some(TEST_CONTEXT))], None);
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn format_documents_json_includes_context() {
        let out = fmt_multi_get(&[make_found(Some(TEST_CONTEXT))], None, OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed[0]["context"], TEST_CONTEXT);
    }

    #[test]
    fn format_documents_md_includes_context() {
        let out = fmt_multi_get(&[make_found(Some(TEST_CONTEXT))], None, OutputFormat::Md);
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn format_documents_xml_includes_context() {
        let out = fmt_multi_get(&[make_found(Some(TEST_CONTEXT))], None, OutputFormat::Xml);
        assert!(out.contains(TEST_CONTEXT));
    }

    // ---- single document: context in every format (qmd group D) ----

    #[test]
    fn single_doc_json_includes_context() {
        let out = document_to_json(&make_doc(Some(TEST_CONTEXT)));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["context"], TEST_CONTEXT);
    }

    #[test]
    fn single_doc_md_includes_context() {
        assert!(document_to_md(&make_doc(Some(TEST_CONTEXT))).contains(TEST_CONTEXT));
    }

    #[test]
    fn single_doc_xml_includes_context() {
        assert!(document_to_xml(&make_doc(Some(TEST_CONTEXT))).contains(TEST_CONTEXT));
    }

    #[test]
    fn format_document_json_includes_context() {
        let out = format_document(&make_doc(Some(TEST_CONTEXT)), OutputFormat::Json);
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["context"], TEST_CONTEXT);
    }

    #[test]
    fn format_document_md_includes_context() {
        let out = format_document(&make_doc(Some(TEST_CONTEXT)), OutputFormat::Md);
        assert!(out.contains(TEST_CONTEXT));
    }

    #[test]
    fn format_document_xml_includes_context() {
        let out = format_document(&make_doc(Some(TEST_CONTEXT)), OutputFormat::Xml);
        assert!(out.contains(TEST_CONTEXT));
    }

    // ---- single document: context omitted when null (qmd group E) ----

    #[test]
    fn single_doc_json_omits_context_when_none() {
        let out = document_to_json(&make_doc(None));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed.get("context").is_none());
    }

    #[test]
    fn single_doc_md_omits_context_line_when_none() {
        assert!(!document_to_md(&make_doc(None)).contains("Context:"));
    }

    #[test]
    fn single_doc_xml_omits_context_element_when_none() {
        assert!(!document_to_xml(&make_doc(None)).contains("<context>"));
    }

    // ---- hardening beyond qmd: camelCase keys + heading fallback ----

    #[test]
    fn single_doc_json_uses_camel_case_keys() {
        let out = document_to_json(&make_doc(Some(TEST_CONTEXT)));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed.get("modifiedAt").is_some());
        assert!(parsed.get("bodyLength").is_some());
    }

    #[test]
    fn single_doc_md_heading_falls_back_to_display_path_when_title_empty() {
        let mut doc = make_doc(Some(TEST_CONTEXT));
        doc.title = String::new();
        // single-doc formatters keep the bare `display_path` (qmd `documentTo*`).
        assert!(document_to_md(&doc).starts_with("# archive/summit/keynote.md\n"));
    }
}
