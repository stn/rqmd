//! Output formatting for `get` and `multi-get`.
//!
//! Maps to qmd's `src/cli/formatter.ts` (the CLI/JSON/CSV/MD/XML/files
//! branches in `multiGet`, `qmd.ts:1262–1336`).

use rqmd_core::store::search::{DocumentResult, MultiGetResult};
use serde_json::{json, Value};

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

/// Write `multi-get` results in the requested format.
pub fn write_multi_get(results: &[MultiGetResult], max_lines: Option<usize>, format: OutputFormat) {
    match format {
        OutputFormat::Json => write_json(results, max_lines),
        OutputFormat::Csv => write_csv(results, max_lines),
        OutputFormat::Files => write_files(results),
        OutputFormat::Md => write_md(results, max_lines),
        OutputFormat::Xml => write_xml(results, max_lines),
        OutputFormat::Cli => write_cli(results, max_lines),
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

fn write_json(results: &[MultiGetResult], max_lines: Option<usize>) {
    let arr: Vec<Value> = results
        .iter()
        .map(|r| match r {
            MultiGetResult::Found(doc) => {
                let mut obj = json!({
                    "file": doc.display_path,
                    "title": title_of(doc),
                    "body": truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines),
                });
                if let Some(ctx) = &doc.context {
                    obj["context"] = json!(ctx);
                }
                obj
            }
            MultiGetResult::Skipped { filepath: _, display_path, skip_reason } => json!({
                "file": display_path,
                "title": display_path.rsplit_once('/').map(|(_, l)| l.to_string()).unwrap_or_else(|| display_path.clone()),
                "skipped": true,
                "reason": skip_reason,
            }),
        })
        .collect();
    let s = serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string());
    println!("{s}");
}

fn write_csv(results: &[MultiGetResult], max_lines: Option<usize>) {
    println!("file,title,context,skipped,body");
    for r in results {
        let row = match r {
            MultiGetResult::Found(doc) => [
                escape_csv(&doc.display_path),
                escape_csv(&title_of(doc)),
                escape_csv(doc.context.as_deref().unwrap_or("")),
                "false".to_string(),
                escape_csv(&truncate_lines(
                    doc.body.as_deref().unwrap_or(""),
                    max_lines,
                )),
            ],
            MultiGetResult::Skipped {
                display_path,
                skip_reason,
                ..
            } => [
                escape_csv(display_path),
                escape_csv(
                    &display_path
                        .rsplit_once('/')
                        .map(|(_, l)| l.to_string())
                        .unwrap_or_else(|| display_path.clone()),
                ),
                String::new(),
                "true".to_string(),
                escape_csv(skip_reason),
            ],
        };
        println!("{}", row.join(","));
    }
}

fn write_files(results: &[MultiGetResult]) {
    for r in results {
        match r {
            MultiGetResult::Found(doc) => {
                if let Some(ctx) = &doc.context {
                    let esc = ctx.replace('"', "\"\"");
                    println!("{},\"{esc}\"", doc.display_path);
                } else {
                    println!("{}", doc.display_path);
                }
            }
            MultiGetResult::Skipped { display_path, .. } => {
                println!("{display_path},[SKIPPED]");
            }
        }
    }
}

fn write_md(results: &[MultiGetResult], max_lines: Option<usize>) {
    for r in results {
        match r {
            MultiGetResult::Found(doc) => {
                println!("## {}\n", doc.display_path);
                let title = title_of(doc);
                if title != doc.display_path {
                    println!("**Title:** {title}\n");
                }
                if let Some(ctx) = &doc.context {
                    println!("**Context:** {ctx}\n");
                }
                println!("```");
                println!(
                    "{}",
                    truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines)
                );
                println!("```\n");
            }
            MultiGetResult::Skipped {
                display_path,
                skip_reason,
                ..
            } => {
                println!("## {display_path}\n");
                println!("> {skip_reason}\n");
            }
        }
    }
}

fn write_xml(results: &[MultiGetResult], max_lines: Option<usize>) {
    println!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    println!("<documents>");
    for r in results {
        println!("  <document>");
        match r {
            MultiGetResult::Found(doc) => {
                println!("    <file>{}</file>", escape_xml(&doc.display_path));
                println!("    <title>{}</title>", escape_xml(&title_of(doc)));
                if let Some(ctx) = &doc.context {
                    println!("    <context>{}</context>", escape_xml(ctx));
                }
                println!(
                    "    <body>{}</body>",
                    escape_xml(&truncate_lines(
                        doc.body.as_deref().unwrap_or(""),
                        max_lines
                    ))
                );
            }
            MultiGetResult::Skipped {
                display_path,
                skip_reason,
                ..
            } => {
                println!("    <file>{}</file>", escape_xml(display_path));
                println!("    <skipped>true</skipped>");
                println!("    <reason>{}</reason>", escape_xml(skip_reason));
            }
        }
        println!("  </document>");
    }
    println!("</documents>");
}

fn write_cli(results: &[MultiGetResult], max_lines: Option<usize>) {
    for r in results {
        let bar = "=".repeat(60);
        match r {
            MultiGetResult::Found(doc) => {
                println!("\n{bar}");
                println!("File: {}", doc.display_path);
                println!("{bar}\n");
                if let Some(ctx) = &doc.context {
                    println!("Folder Context: {ctx}\n---\n");
                }
                println!(
                    "{}",
                    truncate_lines(doc.body.as_deref().unwrap_or(""), max_lines)
                );
            }
            MultiGetResult::Skipped {
                display_path,
                skip_reason,
                ..
            } => {
                println!("\n{bar}");
                println!("File: {display_path}");
                println!("{bar}\n");
                println!("[SKIPPED: {skip_reason}]");
            }
        }
    }
}
