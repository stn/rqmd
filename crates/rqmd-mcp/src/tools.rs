//! MCP tool schemas, handlers, and result DTOs.
//!
//! Port of the four tools registered in `tobi/qmd/src/mcp/server.ts`
//! (`query`, `get`, `multi_get`, `status`) plus the `formatSearchSummary` /
//! `buildInstructions` helpers. Client-visible wording is rqmd-ified (see the
//! plan): the server identifies as `rqmd`, instructions say "Run `rqmd embed`",
//! the SKIPPED hint references the real tool name `get`, and the `status` text
//! reads "rqmd index". The `qmd://` URI scheme is kept (structural).

use rmcp::ErrorData as McpError;
use rmcp::model::{CallToolResult, Content, ReadResourceResult, ResourceContents};
use rmcp::model::{JsonObject, object};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use rqmd_core::store::snippet::{add_line_numbers, extract_snippet};
use rqmd_core::store::virtual_path::encode_qmd_path;
use rqmd_core::{
    CollectionInfo, ExpandedQuery, ExpandedQueryType, FindDocumentOutcome, IndexStatus,
    MultiGetResult, RqmdStore, SearchOptions,
};

// ============================================================================
// Result DTOs (structuredContent + REST JSON)
// ============================================================================

/// A single search hit, mirroring qmd's `SearchResultItem`
/// (`server.ts:39-47`). Field order and `context: string | null` (always
/// present, serialized as `null` when absent) match the TS type exactly.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResultItem {
    /// Short docid with leading `#` (e.g. `#abc123`).
    pub docid: String,
    /// Collection-relative display path (NOT a `qmd://` URI — matches qmd).
    pub file: String,
    pub title: String,
    /// Relevance, rounded to 2 decimals like qmd.
    pub score: f64,
    /// Always emitted; `null` when the document has no context.
    pub context: Option<String>,
    /// 1-indexed absolute line of the best match in the source markdown.
    pub line: usize,
    /// Snippet with `add_line_numbers` applied (anchored at `line`).
    pub snippet: String,
}

/// `status` structuredContent, mirroring qmd's `StatusResult` (camelCase keys).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusResultDto {
    pub total_documents: i64,
    pub needs_embedding: i64,
    pub has_vector_index: bool,
    pub collections: Vec<CollectionDto>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectionDto {
    pub name: String,
    /// `null` when unknown (matches qmd's `path: string | null`).
    pub path: Option<String>,
    pub pattern: Option<String>,
    pub documents: i64,
    pub last_updated: String,
}

impl StatusResultDto {
    fn from_status(s: &IndexStatus) -> Self {
        Self {
            total_documents: s.total_documents,
            needs_embedding: s.needs_embedding,
            has_vector_index: s.has_vector_index,
            collections: s.collections.iter().map(CollectionDto::from_info).collect(),
        }
    }
}

impl CollectionDto {
    fn from_info(c: &CollectionInfo) -> Self {
        Self {
            name: c.name.clone(),
            path: c.path.clone(),
            pattern: c.pattern.clone(),
            documents: c.documents,
            last_updated: c.last_updated.clone(),
        }
    }
}

// ============================================================================
// Tool input argument types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct SubSearch {
    #[serde(rename = "type")]
    pub type_: String,
    pub query: String,
}

/// Shared by the `query` MCP tool and the REST `POST /query` endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryArgs {
    pub searches: Vec<SubSearch>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
    pub candidate_limit: Option<usize>,
    pub collections: Option<Vec<String>>,
    pub intent: Option<String>,
    pub rerank: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetArgs {
    file: String,
    from_line: Option<usize>,
    max_lines: Option<usize>,
    #[serde(default)]
    line_numbers: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MultiGetArgs {
    pattern: String,
    max_lines: Option<usize>,
    max_bytes: Option<usize>,
    #[serde(default)]
    line_numbers: bool,
}

/// Default `maxBytes` for `multi_get` (qmd: `10 * 1024`).
const DEFAULT_MULTI_GET_MAX_BYTES: usize = 10 * 1024;

/// Resolve the effective `maxBytes`, mirroring qmd's `maxBytes || DEFAULT`
/// (`server.ts:448`): a missing value *or* an explicit `0` (falsy in JS) both
/// fall back to the default rather than skipping every file.
fn resolve_max_bytes(max_bytes: Option<usize>) -> usize {
    max_bytes
        .filter(|&b| b != 0)
        .unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES)
}

// ============================================================================
// Tool metadata (names, titles, descriptions, schemas)
// ============================================================================

pub const TOOL_QUERY: &str = "query";
pub const TOOL_GET: &str = "get";
pub const TOOL_MULTI_GET: &str = "multi_get";
pub const TOOL_STATUS: &str = "status";

/// The long `query` tool description (verbatim from `server.ts:241-299`).
pub const QUERY_DESCRIPTION: &str = r#"Search the knowledge base using a query document — one or more typed sub-queries combined for best recall.

Each result includes a `line` field with the absolute 1-indexed line of the best match in the source markdown. To read more context around a hit, call `get(file, fromLine = max(1, line - 20), maxLines = 80, lineNumbers = true)`.

## Query Types

**lex** — BM25 keyword search. Fast, exact, no LLM needed.
Full lex syntax:
- `term` — prefix match ("perf" matches "performance")
- `"exact phrase"` — phrase must appear verbatim
- `-term` or `-"phrase"` — exclude documents containing this

Good lex examples:
- `"connection pool" timeout -redis`
- `"machine learning" -sports -athlete`
- `handleError async typescript`

**vec** — Semantic vector search. Write a natural language question. Finds documents by meaning, not exact words.
- `how does the rate limiter handle burst traffic?`
- `what is the tradeoff between consistency and availability?`

**hyde** — Hypothetical document. Write 50-100 words that look like the answer. Often the most powerful for nuanced topics.
- `The rate limiter uses a token bucket algorithm. When a client exceeds 100 req/min, subsequent requests return 429 until the window resets.`

## Strategy

Combine types for best results. First sub-query gets 2× weight — put your strongest signal first.

| Goal | Approach |
|------|----------|
| Know exact term/name | `lex` only |
| Concept search | `vec` only |
| Best recall | `lex` + `vec` |
| Complex/nuanced | `lex` + `vec` + `hyde` |
| Unknown vocabulary | Use a standalone natural-language query (no typed lines) so the server can auto-expand it |

## Examples

Simple lookup:
```json
[{ "type": "lex", "query": "CAP theorem" }]
```

Best recall on a technical topic:
```json
[
  { "type": "lex", "query": "\"connection pool\" timeout -redis" },
  { "type": "vec", "query": "why do database connections time out under load" },
  { "type": "hyde", "query": "Connection pool exhaustion occurs when all connections are in use and new requests must wait. This typically happens under high concurrency when queries run longer than expected." }
]
```

Intent-aware lex (C++ performance, not sports):
```json
[
  { "type": "lex", "query": "\"C++ performance\" optimization -sports -athlete" },
  { "type": "vec", "query": "how to optimize C++ program performance" }
]
```"#;

pub const GET_DESCRIPTION: &str = "Retrieve the full content of a document by its file path or docid. Use paths or docids (#abc123) from search results. Suggests similar files if not found.";

pub const MULTI_GET_DESCRIPTION: &str = "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.";

// rqmd-ified from qmd's "Show the status of the QMD index: …".
pub const STATUS_DESCRIPTION: &str =
    "Show the status of the rqmd index: collections, document counts, and health information.";

/// `query` input schema (semantic equivalent of the zod schema at
/// `server.ts:301-317`).
pub fn query_input_schema() -> JsonObject {
    object(json!({
        "type": "object",
        "properties": {
            "searches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["lex", "vec", "hyde"],
                            "description": "lex = BM25 keywords (supports \"phrase\" and -negation); vec = semantic question; hyde = hypothetical answer passage"
                        },
                        "query": {
                            "type": "string",
                            "description": "The query text. For lex: use keywords, \"quoted phrases\", and -negation. For vec: natural language question. For hyde: 50-100 word answer passage."
                        }
                    },
                    "required": ["type", "query"],
                    "additionalProperties": false
                },
                "minItems": 1,
                "maxItems": 10,
                "description": "Typed sub-queries to execute (lex/vec/hyde). First gets 2x weight."
            },
            "limit": { "type": "number", "default": 10, "description": "Max results (default: 10)" },
            "minScore": { "type": "number", "default": 0, "description": "Min relevance 0-1 (default: 0)" },
            "candidateLimit": { "type": "number", "description": "Maximum candidates to rerank (default: 40, lower = faster but may miss results)" },
            "collections": { "type": "array", "items": { "type": "string" }, "description": "Filter to collections (OR match)" },
            "intent": { "type": "string", "description": "Background context to disambiguate the query. Example: query='performance', intent='web page load times and Core Web Vitals'. Does not search on its own." },
            "rerank": { "type": "boolean", "default": true, "description": "Rerank results using LLM (default: true). Set to false for faster results on CPU-only machines." }
        },
        "required": ["searches"],
        "additionalProperties": false
    }))
}

pub fn get_input_schema() -> JsonObject {
    object(json!({
        "type": "object",
        "properties": {
            "file": { "type": "string", "description": "File path or docid from search results (e.g., 'pages/meeting.md', '#abc123', or 'pages/meeting.md:100' to start at line 100)" },
            "fromLine": { "type": "number", "description": "Start from this line number (1-indexed)" },
            "maxLines": { "type": "number", "description": "Maximum number of lines to return" },
            "lineNumbers": { "type": "boolean", "default": false, "description": "Add line numbers to output (format: 'N: content')" }
        },
        "required": ["file"],
        "additionalProperties": false
    }))
}

pub fn multi_get_input_schema() -> JsonObject {
    object(json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Glob pattern or comma-separated list of file paths" },
            "maxLines": { "type": "number", "description": "Maximum lines per file" },
            "maxBytes": { "type": "number", "default": 10240, "description": "Skip files larger than this (default: 10240 = 10KB)" },
            "lineNumbers": { "type": "boolean", "default": false, "description": "Add line numbers to output (format: 'N: content')" }
        },
        "required": ["pattern"],
        "additionalProperties": false
    }))
}

pub fn status_input_schema() -> JsonObject {
    object(json!({ "type": "object", "properties": {} }))
}

// ============================================================================
// query
// ============================================================================

/// Run a structured search and shape the hits into [`SearchResultItem`]s.
/// Shared by the `query` tool and the REST `/query` (`/search`) endpoint —
/// mirrors `server.ts:319-356` and the REST handler at `server.ts:684-719`.
pub async fn run_query(
    store: &RqmdStore,
    args: &QueryArgs,
) -> Result<Vec<SearchResultItem>, McpError> {
    let queries: Vec<ExpandedQuery> = args
        .searches
        .iter()
        .map(|s| {
            Ok(ExpandedQuery {
                type_: parse_query_type(&s.type_)?,
                query: s.query.clone(),
                line: None,
            })
        })
        .collect::<Result<_, McpError>>()?;

    // Default collections when none specified (`collections ?? defaultCollectionNames`).
    let effective: Vec<String> = match &args.collections {
        Some(c) => c.clone(),
        None => store
            .default_collection_names()
            .map_err(|e| McpError::internal_error(e.to_string(), None))?,
    };

    let results = store
        .search(SearchOptions {
            queries: Some(queries),
            collections: (!effective.is_empty()).then_some(effective),
            limit: Some(args.limit.unwrap_or(10)),
            min_score: Some(args.min_score.unwrap_or(0.0)),
            candidate_limit: args.candidate_limit,
            rerank: Some(args.rerank.unwrap_or(true)),
            intent: args.intent.clone(),
            ..Default::default()
        })
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

    // Snippet query: first lex, else first vec, else first, else "".
    let primary = args
        .searches
        .iter()
        .find(|s| s.type_ == "lex")
        .or_else(|| args.searches.iter().find(|s| s.type_ == "vec"))
        .or_else(|| args.searches.first())
        .map(|s| s.query.as_str())
        .unwrap_or("");
    let intent = args.intent.as_deref();

    let items = results
        .iter()
        .map(|r| {
            let snip = extract_snippet(
                &r.body,
                primary,
                Some(300),
                Some(r.best_chunk_pos),
                Some(r.best_chunk.len()),
                intent,
            );
            SearchResultItem {
                docid: format!("#{}", r.docid),
                file: r.display_path.clone(),
                title: r.title.clone(),
                score: (r.score * 100.0).round() / 100.0,
                context: r.context.clone(),
                line: snip.line,
                snippet: add_line_numbers(&snip.snippet, Some(snip.line)),
            }
        })
        .collect();
    Ok(items)
}

/// Human-readable summary, port of `formatSearchSummary` (`server.ts:78-87`).
pub fn format_search_summary(results: &[SearchResultItem], query: &str) -> String {
    if results.is_empty() {
        return format!("No results found for \"{query}\"");
    }
    let mut lines = vec![format!(
        "Found {} result{} for \"{}\":\n",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        query
    )];
    for r in results {
        lines.push(format!(
            "{} {}% {} - {}",
            r.docid,
            (r.score * 100.0).round() as i64,
            r.file,
            r.title
        ));
    }
    lines.join("\n")
}

pub async fn handle_query(store: &RqmdStore, args: Value) -> Result<CallToolResult, McpError> {
    let args: QueryArgs = serde_json::from_value(args)
        .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}"), None))?;
    let primary = args
        .searches
        .iter()
        .find(|s| s.type_ == "lex")
        .or_else(|| args.searches.iter().find(|s| s.type_ == "vec"))
        .or_else(|| args.searches.first())
        .map(|s| s.query.clone())
        .unwrap_or_default();
    let items = run_query(store, &args).await?;
    let text = format_search_summary(&items, &primary);
    let structured = json!({ "results": items });
    let mut result = CallToolResult::success(vec![Content::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

fn parse_query_type(s: &str) -> Result<ExpandedQueryType, McpError> {
    match s {
        "lex" => Ok(ExpandedQueryType::Lex),
        "vec" => Ok(ExpandedQueryType::Vec),
        "hyde" => Ok(ExpandedQueryType::Hyde),
        other => Err(McpError::invalid_params(
            format!("invalid search type: {other} (expected lex, vec, or hyde)"),
            None,
        )),
    }
}

// ============================================================================
// get
// ============================================================================

pub async fn handle_get(store: &RqmdStore, args: Value) -> Result<CallToolResult, McpError> {
    let args: GetArgs = serde_json::from_value(args)
        .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}"), None))?;

    // Support a trailing `:N` line suffix when `fromLine` isn't given
    // (`server.ts:383-390`).
    let mut from_line = args.from_line;
    let mut lookup = args.file.clone();
    if from_line.is_none()
        && let Some((head, tail)) = lookup.rsplit_once(':')
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit())
    {
        // All-digit tail: parse, clamping overflow to usize::MAX so an absurd
        // line number still strips the suffix and resolves to an out-of-range
        // start (parity with qmd's `Math.max(1, parseInt(...))`).
        from_line = Some(tail.parse::<usize>().unwrap_or(usize::MAX));
        lookup = head.to_string();
    }
    from_line = from_line.map(|n| n.max(1));

    let found = match store
        .get(&lookup, false)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?
    {
        FindDocumentOutcome::Found(d) => d,
        FindDocumentOutcome::NotFound(nf) => {
            let mut msg = format!("Document not found: {}", args.file);
            if !nf.similar_files.is_empty() {
                msg.push_str("\n\nDid you mean one of these?\n");
                msg.push_str(
                    &nf.similar_files
                        .iter()
                        .map(|s| format!("  - {s}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }
    };

    let body = store
        .get_document_body(&found.filepath, from_line, args.max_lines)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?
        .unwrap_or_default();
    let mut text = body;
    if args.line_numbers {
        text = add_line_numbers(&text, Some(from_line.unwrap_or(1)));
    }
    if let Some(ctx) = &found.context {
        text = format!("<!-- Context: {ctx} -->\n\n{text}");
    }

    // qmd's `get` also sets non-spec `name`/`title` on the embedded resource
    // (server.ts:418-423). Those are absent here on purpose: rmcp's
    // `ResourceContents` is a strict spec enum (uri/mimeType/text/_meta only) with
    // no name/title and no extension seam, and a compliant client strips them
    // anyway. Sanctioned parity gap (like the `rqmd` server name); the data is
    // recoverable from `uri` (encodes `display_path`) + the document title, and is
    // covered at the function level by rqmd-core's `store_lookup` tests.
    let uri = format!("qmd://{}", encode_qmd_path(&found.display_path));
    let rc = ResourceContents::text(text, uri).with_mime_type("text/markdown");
    Ok(CallToolResult::success(vec![Content::resource(rc)]))
}

// ============================================================================
// multi_get
// ============================================================================

pub async fn handle_multi_get(store: &RqmdStore, args: Value) -> Result<CallToolResult, McpError> {
    let args: MultiGetArgs = serde_json::from_value(args)
        .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}"), None))?;
    let max_bytes = resolve_max_bytes(args.max_bytes);

    let bundle = store
        .multi_get(&args.pattern, true, Some(max_bytes))
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

    if bundle.docs.is_empty() && bundle.errors.is_empty() {
        return Ok(CallToolResult::error(vec![Content::text(format!(
            "No files matched pattern: {}",
            args.pattern
        ))]));
    }

    let mut content: Vec<Content> = Vec::new();
    if !bundle.errors.is_empty() {
        content.push(Content::text(format!(
            "Errors:\n{}",
            bundle.errors.join("\n")
        )));
    }

    for doc in bundle.docs {
        match doc {
            MultiGetResult::Skipped {
                display_path,
                skip_reason,
                ..
            } => {
                content.push(Content::text(format!(
                    "[SKIPPED: {display_path} - {skip_reason}. Use 'get' with file=\"{display_path}\" to retrieve.]"
                )));
            }
            MultiGetResult::Found(d) => {
                let mut text = d.body.clone().unwrap_or_default();
                if let Some(max) = args.max_lines {
                    let lines: Vec<&str> = text.split('\n').collect();
                    let kept = lines[..max.min(lines.len())].join("\n");
                    text = if lines.len() > max {
                        format!("{kept}\n\n[... truncated {} more lines]", lines.len() - max)
                    } else {
                        kept
                    };
                }
                if args.line_numbers {
                    text = add_line_numbers(&text, None);
                }
                if let Some(ctx) = &d.context {
                    text = format!("<!-- Context: {ctx} -->\n\n{text}");
                }
                let uri = format!("qmd://{}", encode_qmd_path(&d.display_path));
                let rc = ResourceContents::text(text, uri).with_mime_type("text/markdown");
                content.push(Content::resource(rc));
            }
        }
    }

    Ok(CallToolResult::success(content))
}

// ============================================================================
// status
// ============================================================================

pub async fn handle_status(store: &RqmdStore) -> Result<CallToolResult, McpError> {
    let status = store
        .status()
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    let text = format_status_summary(&status);
    let structured = serde_json::to_value(StatusResultDto::from_status(&status))
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    let mut result = CallToolResult::success(vec![Content::text(text)]);
    result.structured_content = Some(structured);
    Ok(result)
}

/// Port of the `status` text summary (`server.ts:518-527`), rqmd-ified header.
pub fn format_status_summary(status: &IndexStatus) -> String {
    let mut lines = vec![
        "RQMD Index Status:".to_string(),
        format!("  Total documents: {}", status.total_documents),
        format!("  Needs embedding: {}", status.needs_embedding),
        format!(
            "  Vector index: {}",
            if status.has_vector_index { "yes" } else { "no" }
        ),
        format!("  Collections: {}", status.collections.len()),
    ];
    for col in &status.collections {
        // `${col.path}` renders "null" in JS when path is null.
        let path = col.path.as_deref().unwrap_or("null");
        lines.push(format!(
            "    - {}: {} ({} docs)",
            col.name, path, col.documents
        ));
    }
    lines.join("\n")
}

// ============================================================================
// resource read (qmd://{+path})
// ============================================================================

/// Read a document for the `qmd://{+path}` resource. `decoded` is the already
/// url-decoded path, `uri` the original request URI (echoed in the result).
/// Port of the resource handler at `server.ts:193-219` (default line numbers).
pub fn handle_read_resource(
    store: &RqmdStore,
    decoded: &str,
    uri: &str,
) -> Result<ReadResourceResult, McpError> {
    let contents = match store
        .get(decoded, true)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?
    {
        FindDocumentOutcome::Found(doc) => {
            let mut text = add_line_numbers(doc.body.as_deref().unwrap_or(""), None);
            if let Some(ctx) = &doc.context {
                text = format!("<!-- Context: {ctx} -->\n\n{text}");
            }
            vec![ResourceContents::text(text, uri.to_string()).with_mime_type("text/markdown")]
        }
        FindDocumentOutcome::NotFound(_) => {
            vec![ResourceContents::text(
                format!("Document not found: {decoded}"),
                uri.to_string(),
            )]
        }
    };
    Ok(ReadResourceResult::new(contents))
}

// ============================================================================
// Instructions (initialize response)
// ============================================================================

/// Build the dynamic server instructions, port of `buildInstructions`
/// (`server.ts:108-165`) with rqmd-ified wording. Errors reading status fall
/// back to `None` (the server still starts). Computed once at server creation
/// (qmd calls `buildInstructions` once in `createMcpServer`), so this takes a
/// plain `&RqmdStore` rather than the shared `Mutex`-wrapped handle.
pub fn build_instructions(store: &RqmdStore) -> Option<String> {
    let status = store.status().ok()?;
    let global_ctx = store.get_global_context().ok().flatten();
    let mut lines: Vec<String> = Vec::new();

    lines.push(format!(
        "RQMD is your local search engine over {} markdown documents.",
        status.total_documents
    ));
    if let Some(ctx) = &global_ctx {
        lines.push(format!("Context: {ctx}"));
    }

    if !status.collections.is_empty() {
        lines.push(String::new());
        let names = status
            .collections
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!(
            "Collections (scope with `collection` parameter): {names}"
        ));
        lines.push(
            "Call the `status` tool for collection descriptions, paths, and per-collection doc counts."
                .to_string(),
        );
    }

    if !status.has_vector_index {
        lines.push(String::new());
        lines.push(
            "Note: No vector embeddings yet. Run `rqmd embed` to enable semantic search (vec/hyde)."
                .to_string(),
        );
    } else if status.needs_embedding > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Note: {} documents need embedding. Run `rqmd embed` to update.",
            status.needs_embedding
        ));
    }

    lines.push(String::new());
    lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):".to_string());
    lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)".to_string());
    lines.push("  - type:'vec' — semantic vector search (meaning-based)".to_string());
    lines.push(
        "  - type:'hyde' — hypothetical document (write what the answer looks like)".to_string(),
    );
    lines.push(String::new());
    lines.push(
        "  Always provide `intent` on every search call to disambiguate and improve snippets."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("Examples:".to_string());
    lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]".to_string());
    lines.push(
        "  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]".to_string(),
    );
    lines.push(
        "  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]"
            .to_string(),
    );
    lines.push(
        "  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'"
            .to_string(),
    );
    lines.push(String::new());
    lines.push("Retrieval:".to_string());
    lines.push(
        "  - `get` — single document by path or docid (#abc123). Supports line offset (`file.md:100`)."
            .to_string(),
    );
    lines.push(
        "  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`) or comma-separated list."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("Tips:".to_string());
    lines.push("  - File paths in results are relative to their collection.".to_string());
    lines.push("  - Use `minScore: 0.5` to filter low-confidence results.".to_string());
    lines.push("  - Results include a `context` field describing the content type.".to_string());

    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(
        docid: &str,
        file: &str,
        title: &str,
        score: f64,
        ctx: Option<&str>,
    ) -> SearchResultItem {
        SearchResultItem {
            docid: docid.to_string(),
            file: file.to_string(),
            title: title.to_string(),
            score,
            context: ctx.map(String::from),
            line: 1,
            snippet: "snip".to_string(),
        }
    }

    #[test]
    fn summary_empty() {
        assert_eq!(
            format_search_summary(&[], "cap theorem"),
            "No results found for \"cap theorem\""
        );
    }

    #[test]
    fn summary_singular_and_rows() {
        let items = vec![item("#abc", "docs/a.md", "A", 0.95, None)];
        let s = format_search_summary(&items, "auth");
        assert_eq!(s, "Found 1 result for \"auth\":\n\n#abc 95% docs/a.md - A");
    }

    #[test]
    fn summary_plural() {
        let items = vec![
            item("#a", "x.md", "X", 0.5, None),
            item("#b", "y.md", "Y", 0.4, None),
        ];
        let s = format_search_summary(&items, "q");
        assert!(s.starts_with("Found 2 results for \"q\":\n"));
    }

    #[test]
    fn item_serializes_context_null_in_order() {
        let v = serde_json::to_value(item("#a", "x.md", "X", 0.5, None)).unwrap();
        // context present and null.
        assert!(v.as_object().unwrap().contains_key("context"));
        assert!(v["context"].is_null());
        // field order preserved (serde_json preserve_order): docid first.
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "docid", "file", "title", "score", "context", "line", "snippet"
            ]
        );
    }

    #[test]
    fn item_serializes_context_value() {
        let v = serde_json::to_value(item("#a", "x.md", "X", 0.5, Some("ctx"))).unwrap();
        assert_eq!(v["context"], "ctx");
    }

    #[test]
    fn status_dto_is_camel_case() {
        let dto = StatusResultDto {
            total_documents: 5,
            needs_embedding: 1,
            has_vector_index: true,
            collections: vec![CollectionDto {
                name: "docs".into(),
                path: Some("/p".into()),
                pattern: Some("**/*.md".into()),
                documents: 5,
                last_updated: "2024".into(),
            }],
        };
        let v = serde_json::to_value(&dto).unwrap();
        assert_eq!(v["totalDocuments"], 5);
        assert_eq!(v["needsEmbedding"], 1);
        assert_eq!(v["hasVectorIndex"], true);
        assert_eq!(v["collections"][0]["lastUpdated"], "2024");
    }

    #[test]
    fn schemas_have_expected_shape() {
        let q = query_input_schema();
        assert_eq!(q["required"], json!(["searches"]));
        assert_eq!(q["properties"]["limit"]["default"], 10);
        assert_eq!(q["properties"]["rerank"]["default"], true);
        assert_eq!(
            q["properties"]["searches"]["items"]["properties"]["type"]["enum"],
            json!(["lex", "vec", "hyde"])
        );
        assert_eq!(q["properties"]["searches"]["minItems"], 1);
        assert_eq!(q["properties"]["searches"]["maxItems"], 10);

        let g = get_input_schema();
        assert_eq!(g["required"], json!(["file"]));
        assert_eq!(g["properties"]["lineNumbers"]["default"], false);

        let m = multi_get_input_schema();
        assert_eq!(m["required"], json!(["pattern"]));
        assert_eq!(m["properties"]["maxBytes"]["default"], 10240);
    }

    #[test]
    fn resolve_max_bytes_treats_zero_and_none_as_default() {
        assert_eq!(resolve_max_bytes(None), DEFAULT_MULTI_GET_MAX_BYTES);
        // Explicit 0 is falsy in qmd's `maxBytes || DEFAULT` → default.
        assert_eq!(resolve_max_bytes(Some(0)), DEFAULT_MULTI_GET_MAX_BYTES);
        assert_eq!(resolve_max_bytes(Some(1)), 1);
        assert_eq!(resolve_max_bytes(Some(2048)), 2048);
    }

    #[test]
    fn parse_query_type_maps_and_rejects() {
        assert_eq!(parse_query_type("lex").unwrap(), ExpandedQueryType::Lex);
        assert_eq!(parse_query_type("vec").unwrap(), ExpandedQueryType::Vec);
        assert_eq!(parse_query_type("hyde").unwrap(), ExpandedQueryType::Hyde);
        assert!(parse_query_type("bogus").is_err());
    }
}
