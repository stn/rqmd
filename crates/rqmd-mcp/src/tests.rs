//! Integration tests: drive the MCP server end-to-end.
//!
//! Three flavours: (1) `stdio_protocol_roundtrip` wires an in-process `rmcp`
//! client to [`QmdMcpServer`] over a `tokio::io::duplex` pair; (2)
//! `http_health_query_and_404` exercises the axum router via `tower`'s `oneshot`
//! (no sockets/TLS); (3) `http_mcp_streamable_roundtrip` runs the *real* `/mcp`
//! Streamable HTTP endpoint over an ephemeral TCP port with rmcp's HTTP client —
//! the closest port of qmd's "MCP HTTP Transport" block. All three assert the
//! qmd parity points (server name, tool list, and the absolute-line snippet from
//! `tobi/qmd/test/mcp.test.ts:1094-1117`). The matching real-stdio roundtrip
//! against the spawned `rqmd mcp` binary lives in `rqmd/tests/mcp_stdio.rs`.
//!
//! All assertions here are LLM-free: `status`/`get`/`read_resource` are pure
//! SQL, and the `query` cases use `lex` sub-queries with `rerank: false`, which
//! `structured_search` resolves without loading a model (see store_ops).

use std::path::Path;

use rmcp::model::{CallToolResult, ReadResourceResult};
use serde_json::{Value, json};

use rqmd_core::store::documents::{insert_content, insert_document};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::{AddCollectionOptions, RqmdStore, RqmdStoreOptions, UpdateOptions};

use crate::server::QmdMcpServer;

/// Documents seeded by [`seed_store`]: the 6 on-disk fixtures plus the
/// raw-inserted spaces-path doc. Pinned so `status` tests assert an exact count.
const SEEDED_DOCS: i64 = 7;

/// Seed a temp index mirroring qmd's `seedTestData` plus the absolute-line
/// fixture. Collection `docs` → display paths like `docs/readme.md`.
async fn seed_store(dir: &Path) -> RqmdStore {
    let fixtures = dir.join("docs");
    std::fs::create_dir_all(fixtures.join("meetings")).unwrap();
    std::fs::write(
        fixtures.join("readme.md"),
        "# Project README\n\nThis is the main readme file for the project.\n\nIt contains important information about setup and usage.",
    )
    .unwrap();
    std::fs::write(
        fixtures.join("api.md"),
        "# API Documentation\n\nThis document describes the REST API endpoints.\n\n## Authentication\n\nUse Bearer tokens for auth.",
    )
    .unwrap();
    std::fs::write(
        fixtures.join("meetings/meeting-2024-01.md"),
        "# January Meeting Notes\n\nDiscussed Q1 goals and roadmap.",
    )
    .unwrap();
    std::fs::write(
        fixtures.join("meetings/meeting-2024-02.md"),
        "# February Meeting Notes\n\nFollowed up on Q1 progress.",
    )
    .unwrap();
    std::fs::write(
        fixtures.join("large-file.md"),
        format!("# Large Document\n\n{}", "Lorem ipsum ".repeat(2000)),
    )
    .unwrap();
    // 300 pad lines push the marker past the first chunk boundary; the marker
    // sits on absolute line 301.
    let pad = "Pad line for chunk boundary coverage\n";
    let abs = format!(
        "{}{}{}",
        pad.repeat(300),
        "UNIQUE_KEYWORD_XYZ marker\n",
        pad.repeat(20)
    );
    std::fs::write(fixtures.join("absolute-line-fixture.md"), abs).unwrap();

    let mut store = RqmdStore::open(RqmdStoreOptions {
        db_path: dir.join("index.sqlite"),
        ..Default::default()
    })
    .unwrap();
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: fixtures.to_string_lossy().into_owned(),
                pattern: None,
                ignore: None,
            },
        )
        .unwrap();
    store.update(UpdateOptions::default()).await.unwrap();

    // Per-path context (qmd config: collection "docs", context "/meetings").
    assert!(
        store
            .add_context("docs", "/meetings", "Meeting notes and transcripts")
            .unwrap(),
        "add_context should find the docs collection"
    );

    // Spaces-in-path doc, inserted raw to bypass `handelize` (which slugifies
    // " " → "-"). Mirrors qmd's direct DB insert at mcp.test.ts:736-745 — a
    // spaces path can't survive real indexing in either project.
    let now = now_rfc3339();
    let s = store.internal();
    s.with_connection(|c| {
        insert_content(
            c,
            "hash_spaces",
            "# Podcast Episode\n\nInterview content here.",
            &now,
        )
    })
    .unwrap();
    s.with_connection(|c| {
        insert_document(
            c,
            "docs",
            "External Podcast/2023 April - Interview.md",
            "Podcast Episode",
            "hash_spaces",
            &now,
            &now,
        )
    })
    .unwrap();

    store
}

fn call(name: &str, args: Value) -> rmcp::model::CallToolRequestParams {
    serde_json::from_value(json!({ "name": name, "arguments": args })).unwrap()
}

// ============================================================================
// In-process client harness for the granular per-tool tests
// ============================================================================
//
// These mirror qmd's `mcp.test.ts` "MCP Server" describe block 1:1 but drive
// everything through the real `rmcp` client (the existing roundtrips above only
// smoke-test a subset). The function-level equivalents live in rqmd-core
// (`store_search_fts.rs`, `store_lookup.rs`, `status.rs`, …); having both lets a
// regression be pinned to the core function vs the MCP plumbing.
//
// LLM-free: every `query` here uses `lex` + `rerank:false`, and
// `get`/`multi_get`/`status`/resource reads are pure SQL — no model is loaded.
// The vec/hyde/rerank scenarios live in `tests/mcp_llm.rs`.

type Client = rmcp::service::RunningService<rmcp::service::RoleClient, ()>;
type Server = rmcp::service::RunningService<rmcp::service::RoleServer, QmdMcpServer>;

/// An in-process client+server pair over a duplex pipe, seeded by `seed_store`.
/// Holds the tempdir and both `RunningService` ends alive for the test.
struct Mcp {
    tmp: tempfile::TempDir,
    client: Client,
    _server: Server,
}

impl Mcp {
    /// Deterministic teardown (the worker thread is also reclaimed on drop, so
    /// this isn't strictly required — it just avoids the cancel-on-drop debug log).
    async fn shutdown(self) {
        self.client.cancel().await.ok();
    }
}

/// Seed a fresh store and wire an in-process rmcp client to its MCP server.
async fn connect() -> Mcp {
    use rmcp::{serve_client, serve_server};
    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_store(tmp.path()).await);
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let (srv, cli) = tokio::join!(serve_server(server, server_io), serve_client((), client_io));
    Mcp {
        tmp,
        _server: srv.expect("server initialized"),
        client: cli.expect("client initialized"),
    }
}

/// Run a `lex` query with reranking off (LLM-free).
async fn query_lex(mcp: &Mcp, q: &str) -> CallToolResult {
    mcp.client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "lex", "query": q }], "rerank": false }),
        ))
        .await
        .expect("query")
}

/// Read a `qmd://` resource by URI.
async fn read(mcp: &Mcp, uri: &str) -> ReadResourceResult {
    let params: rmcp::model::ReadResourceRequestParams =
        serde_json::from_value(json!({ "uri": uri })).unwrap();
    mcp.client
        .read_resource(params)
        .await
        .expect("read resource")
}

/// `structuredContent.results` as a JSON array.
fn structured_results(r: &CallToolResult) -> Vec<Value> {
    r.structured_content.as_ref().expect("structuredContent")["results"]
        .as_array()
        .expect("results array")
        .clone()
}

/// `content[i]` serialized to JSON.
fn block(r: &CallToolResult, i: usize) -> Value {
    serde_json::to_value(&r.content[i]).unwrap()
}

/// Concatenated text of every `type:"text"` content block.
fn text_blocks(r: &CallToolResult) -> String {
    r.content
        .iter()
        .filter_map(|c| {
            let v = serde_json::to_value(c).ok()?;
            (v["type"] == "text").then(|| v["text"].as_str().unwrap_or_default().to_string())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// All `type:"resource"` content blocks, serialized to JSON.
fn resource_blocks(r: &CallToolResult) -> Vec<Value> {
    r.content
        .iter()
        .filter_map(|c| {
            let v = serde_json::to_value(c).ok()?;
            (v["type"] == "resource").then_some(v)
        })
        .collect()
}

/// Text of the first resource-read content block.
fn resource_text(r: &ReadResourceResult) -> String {
    serde_json::to_value(&r.contents[0]).unwrap()["text"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn stdio_protocol_roundtrip() {
    use rmcp::{serve_client, serve_server};

    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_store(tmp.path()).await);

    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let (srv, cli) = tokio::join!(serve_server(server, server_io), serve_client((), client_io),);
    let _server = srv.expect("server initialized");
    let client = cli.expect("client initialized");

    // --- initialize: server identity is `rqmd` (divergence from qmd's `qmd`) ---
    let info = client.peer_info().expect("peer info");
    assert_eq!(info.server_info.name, "rqmd");
    assert!(
        info.instructions
            .as_deref()
            .unwrap_or_default()
            .contains("RQMD is your local search engine"),
        "instructions should be rqmd-ified"
    );

    // --- tools/list ⊇ {query, get, multi_get, status} ---
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    for expected in ["query", "get", "multi_get", "status"] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }

    // --- resource template qmd://{+path} ---
    let templates = client
        .list_resource_templates(None)
        .await
        .expect("list resource templates");
    assert_eq!(
        templates.resource_templates[0].uri_template,
        "qmd://{+path}"
    );

    // --- status (pure SQL) ---
    let status = client
        .call_tool(call("status", json!({})))
        .await
        .expect("status");
    let sc = status.structured_content.expect("status structuredContent");
    assert!(sc["totalDocuments"].as_i64().unwrap() >= 5);
    assert_eq!(sc["hasVectorIndex"], false); // no embeddings generated

    // --- get (pure SQL): a single resource block with the document body ---
    let got = client
        .call_tool(call("get", json!({ "file": "readme.md" })))
        .await
        .expect("get");
    assert_eq!(got.is_error, Some(false));
    let block = serde_json::to_value(&got.content[0]).unwrap();
    assert_eq!(block["type"], "resource");
    assert!(
        block["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("Project README")
    );

    // --- read_resource via the qmd:// template ---
    let read_params: rmcp::model::ReadResourceRequestParams =
        serde_json::from_value(json!({ "uri": "qmd://docs/readme.md" })).unwrap();
    let read = client
        .read_resource(read_params)
        .await
        .expect("read resource");
    let rc = serde_json::to_value(&read.contents[0]).unwrap();
    assert!(rc["text"].as_str().unwrap().contains("Project README"));

    // --- query (lex + rerank:false): absolute source line, not chunk-local ---
    let q = client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "lex", "query": "UNIQUE_KEYWORD_XYZ" }], "rerank": false }),
        ))
        .await
        .expect("query");
    // Text content first (formatSearchSummary), structured results alongside.
    let text_block = serde_json::to_value(&q.content[0]).unwrap();
    assert_eq!(text_block["type"], "text");
    let results = q.structured_content.expect("query structuredContent");
    let results = results["results"].as_array().expect("results array");
    let hit = results
        .iter()
        .find(|r| r["file"] == "docs/absolute-line-fixture.md")
        .expect("absolute-line hit");
    assert_eq!(hit["line"], 301);
    let first_line = hit["snippet"].as_str().unwrap().lines().next().unwrap();
    // Mirrors qmd's /^\d+: @@ -3\d\d,/ — line-numbered diff-style snippet header.
    assert!(
        first_line.contains(": @@ -3"),
        "unexpected snippet header: {first_line}"
    );

    client.cancel().await.ok();
}

#[tokio::test(flavor = "multi_thread")]
async fn http_health_query_and_404() {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_store(tmp.path()).await);
    let app = crate::http::router(server);

    // GET /health -> 200 { status: "ok", uptime: <number> }
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["status"], "ok");
    assert!(v["uptime"].is_number());

    // GET /other -> 404
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/other")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // POST /query -> { results: [...] }
    let body =
        serde_json::to_vec(&json!({ "searches": [{ "type": "lex", "query": "readme" }] })).unwrap();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/query")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["results"].is_array());

    // POST /query without `searches` -> 400 with the qmd error message.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/query")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"], "Missing required field: searches (array)");
}

/// End-to-end over the *real* `/mcp` Streamable HTTP endpoint: bind an ephemeral
/// TCP port, run `axum::serve`, and drive it with rmcp's HTTP client transport
/// (sessions + SSE framing — no in-process shortcut). Ports qmd's "MCP HTTP
/// Transport" block (`tobi/qmd/test/mcp.test.ts:1023-1117`), adapted: the server
/// identifies as `rqmd` (not `qmd`) and the framing is SSE rather than direct
/// JSON (transparent to a compliant client).
#[tokio::test(flavor = "multi_thread")]
async fn http_mcp_streamable_roundtrip() {
    use rmcp::serve_client;
    use rmcp::transport::StreamableHttpClientTransport;

    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_store(tmp.path()).await);
    let app = crate::http::router(server);

    // Bind before spawning so the OS accept queue exists before the client
    // connects — no connect race even if the serve task hasn't been polled yet.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        // Runs until the test aborts the task; ignore the result so an abort at
        // an await point can't surface as a spurious panic.
        let _ = axum::serve(listener, app).await;
    });

    let transport = StreamableHttpClientTransport::from_uri(format!("http://{addr}/mcp"));
    let client = serve_client((), transport).await.expect("client connected");

    // --- initialize: server identity is `rqmd` (divergence from qmd's `qmd`) ---
    let info = client.peer_info().expect("peer info");
    assert_eq!(info.server_info.name, "rqmd");

    // --- tools/list ⊇ {query, get, multi_get, status} ---
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    for expected in ["query", "get", "multi_get", "status"] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }

    // --- status (pure SQL) ---
    let status = client
        .call_tool(call("status", json!({})))
        .await
        .expect("status");
    let sc = status.structured_content.expect("status structuredContent");
    assert!(sc["totalDocuments"].as_i64().unwrap() >= 5);

    // --- get (pure SQL): a single resource block with the document body ---
    let got = client
        .call_tool(call("get", json!({ "file": "readme.md" })))
        .await
        .expect("get");
    assert_eq!(got.is_error, Some(false));
    let block = serde_json::to_value(&got.content[0]).unwrap();
    assert_eq!(block["type"], "resource");
    assert!(
        block["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("Project README")
    );

    // --- query (lex + rerank:false): absolute source line, not chunk-local ---
    let q = client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "lex", "query": "UNIQUE_KEYWORD_XYZ" }], "rerank": false }),
        ))
        .await
        .expect("query");
    let results = q.structured_content.expect("query structuredContent");
    let results = results["results"].as_array().expect("results array");
    let hit = results
        .iter()
        .find(|r| r["file"] == "docs/absolute-line-fixture.md")
        .expect("absolute-line hit");
    assert_eq!(hit["line"], 301);
    let first_line = hit["snippet"].as_str().unwrap().lines().next().unwrap();
    // Mirrors qmd's /^\d+: @@ -3\d\d,/ — line-numbered diff-style snippet header.
    assert!(
        first_line.contains(": @@ -3"),
        "unexpected snippet header: {first_line}"
    );

    client.cancel().await.ok();
    server_task.abort();
}

// ============================================================================
// query tool — searchFTS + edge cases (qmd mcp.test.ts:285-320, 770-796)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_returns_results_for_matching_query() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "readme").await;
    let res = structured_results(&r);
    assert!(!res.is_empty());
    assert_eq!(res[0]["file"], "docs/readme.md");
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_returns_empty_for_non_matching() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "xyznonexistent").await;
    assert!(structured_results(&r).is_empty());
    assert!(text_blocks(&r).contains("No results found"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_respects_limit() {
    let mcp = connect().await;
    let r = mcp
        .client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "lex", "query": "meeting" }], "limit": 1, "rerank": false }),
        ))
        .await
        .expect("query");
    assert_eq!(structured_results(&r).len(), 1);
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_results_have_structured_shape() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "readme").await;
    let res = structured_results(&r);
    assert!(!res.is_empty());
    let item = &res[0];
    assert!(item["docid"].is_string());
    assert!(item["file"].is_string());
    assert!(item["title"].is_string());
    // 0..=1 holds only because rerank is off (skip-rerank sets score = 1.0/rank).
    let score = item["score"].as_f64().expect("score number");
    assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    assert!(item["line"].is_number());
    assert!(item["snippet"].is_string());
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_handles_empty_query() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "").await;
    assert!(structured_results(&r).is_empty());
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_handles_special_chars() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "project's").await;
    let _ = structured_results(&r); // array, no panic
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_handles_unicode() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "文档").await;
    let _ = structured_results(&r);
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_handles_very_long_query() {
    let mcp = connect().await;
    let long = "documentation ".repeat(100);
    let r = query_lex(&mcp, &long).await;
    let _ = structured_results(&r);
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn query_handles_stopwords() {
    let mcp = connect().await;
    let r = query_lex(&mcp, "the and or").await;
    let _ = structured_results(&r);
    mcp.shutdown().await;
}

// ============================================================================
// get tool (qmd mcp.test.ts:441-512)
// ============================================================================

async fn get(mcp: &Mcp, args: Value) -> CallToolResult {
    mcp.client.call_tool(call("get", args)).await.expect("get")
}

#[tokio::test(flavor = "multi_thread")]
async fn get_by_display_path() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "docs/readme.md" })).await;
    assert_eq!(r.is_error, Some(false));
    let b = block(&r, 0);
    assert_eq!(b["type"], "resource");
    assert!(
        b["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("Project README")
    );
    assert_eq!(b["resource"]["uri"], "qmd://docs/readme.md");
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_by_collection_relative_path() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "readme.md" })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("Project README")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_by_partial_path() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "api.md" })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("API Documentation")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_by_absolute_path() {
    let mcp = connect().await;
    let abs = mcp.tmp.path().join("docs").join("api.md");
    let r = get(&mcp, json!({ "file": abs.to_string_lossy() })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("API Documentation")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_not_found() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "nonexistent.md" })).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text_blocks(&r).to_lowercase().contains("not found"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_suggests_similar() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "readm.md" })).await; // typo
    assert_eq!(r.is_error, Some(true));
    let t = text_blocks(&r);
    assert!(t.contains("Did you mean"), "missing suggestion header: {t}");
    assert!(t.contains("readme.md"), "missing suggested file: {t}");
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_supports_line_suffix() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "readme.md:2" })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        !block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("# Project README")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_supports_from_line() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "readme.md", "fromLine": 3 })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        !block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("# Project README")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_supports_max_lines() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "api.md", "maxLines": 3 })).await;
    assert_eq!(r.is_error, Some(false));
    let text = block(&r, 0)["resource"]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        text.lines().count() <= 3,
        "expected <= 3 lines, got:\n{text}"
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn get_includes_context() {
    let mcp = connect().await;
    let r = get(&mcp, json!({ "file": "meetings/meeting-2024-01.md" })).await;
    assert_eq!(r.is_error, Some(false));
    assert!(
        block(&r, 0)["resource"]["text"]
            .as_str()
            .unwrap()
            .starts_with("<!-- Context: Meeting notes and transcripts -->")
    );
    mcp.shutdown().await;
}

// ============================================================================
// multi_get tool (qmd mcp.test.ts:518-580)
// ============================================================================

async fn multi_get(mcp: &Mcp, args: Value) -> CallToolResult {
    mcp.client
        .call_tool(call("multi_get", args))
        .await
        .expect("multi_get")
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_by_glob() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "meetings/*.md" })).await;
    assert_eq!(resource_blocks(&r).len(), 2);
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_by_comma_list() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "readme.md, api.md" })).await;
    assert_eq!(resource_blocks(&r).len(), 2);
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_errors_for_missing_in_comma_list() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "readme.md, nonexistent.md" })).await;
    assert_eq!(resource_blocks(&r).len(), 1);
    assert!(text_blocks(&r).to_lowercase().contains("not found"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_skips_large_files() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "*.md", "maxBytes": 1000 })).await;
    assert!(
        text_blocks(&r).contains("[SKIPPED: docs/large-file.md"),
        "expected large-file to be skipped:\n{}",
        text_blocks(&r)
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_respects_max_lines() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "readme.md", "maxLines": 2 })).await;
    let blocks = resource_blocks(&r);
    assert_eq!(blocks.len(), 1);
    let text = blocks[0]["resource"]["text"].as_str().unwrap();
    assert!(text.contains("# Project README"));
    assert!(
        !text.contains("This is the main readme"),
        "body past line 2 not truncated:\n{text}"
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_non_matching_glob() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "nonexistent/*.md" })).await;
    // qmd surfaces this as a "No files matched" message (function-level returns
    // it in `errors`); assert the text rather than the isError flag.
    assert!(text_blocks(&r).contains("No files matched"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_get_includes_context() {
    let mcp = connect().await;
    let r = multi_get(&mcp, json!({ "pattern": "meetings/meeting-2024-01.md" })).await;
    let blocks = resource_blocks(&r);
    assert_eq!(blocks.len(), 1);
    assert!(
        blocks[0]["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("<!-- Context:")
    );
    mcp.shutdown().await;
}

// ============================================================================
// status tool (qmd mcp.test.ts:586-600, adapted: rqmd seeds no embeddings)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn status_returns_index_status() {
    let mcp = connect().await;
    let r = mcp
        .client
        .call_tool(call("status", json!({})))
        .await
        .expect("status");
    let sc = r.structured_content.as_ref().expect("structuredContent");
    assert_eq!(sc["totalDocuments"].as_i64().unwrap(), SEEDED_DOCS);
    assert_eq!(sc["hasVectorIndex"], false);
    let cols = sc["collections"].as_array().unwrap();
    assert_eq!(cols.len(), 1);
    assert!(
        cols[0]["path"]
            .as_str()
            .unwrap()
            .replace('\\', "/")
            .ends_with("docs")
    );
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn status_shows_documents_needing_embedding() {
    let mcp = connect().await;
    let r = mcp
        .client
        .call_tool(call("status", json!({})))
        .await
        .expect("status");
    let sc = r.structured_content.as_ref().expect("structuredContent");
    // No embeddings seeded, so every distinct content hash is pending.
    let needs = sc["needsEmbedding"].as_i64().unwrap();
    assert_eq!(needs, SEEDED_DOCS);
    assert!(needs >= 1);
    mcp.shutdown().await;
}

// ============================================================================
// qmd:// resource via read_resource (qmd mcp.test.ts:606-763)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn resource_read_by_display_path() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://docs/readme.md").await;
    assert!(resource_text(&r).contains("Project README"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_read_collection_relative() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://readme.md").await;
    assert!(resource_text(&r).contains("Project README"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_read_url_encoded() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://meetings%2Fmeeting-2024-01.md").await;
    assert!(resource_text(&r).contains("January Meeting"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_read_suffix_match() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://meeting-2024-01.md").await;
    assert!(resource_text(&r).contains("January Meeting"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_not_found() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://nonexistent.md").await;
    assert!(resource_text(&r).contains("Document not found"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_includes_context() {
    let mcp = connect().await;
    let r = read(&mcp, "qmd://meetings/meeting-2024-01.md").await;
    assert!(resource_text(&r).contains("<!-- Context:"));
    mcp.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_url_encoded_with_spaces() {
    let mcp = connect().await;
    let r = read(
        &mcp,
        "qmd://External%20Podcast%2F2023%20April%20-%20Interview.md",
    )
    .await;
    assert!(resource_text(&r).contains("Podcast Episode"));
    mcp.shutdown().await;
}

// ============================================================================
// MCP spec compliance via real calls (qmd mcp.test.ts:817-889)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn spec_get_resource_uri_is_encoded() {
    let mcp = connect().await;
    // encodeQmdPath through MCP: slashes preserved, spaces → %20, '-'/'.' kept.
    let r = get(
        &mcp,
        json!({ "file": "External Podcast/2023 April - Interview.md" }),
    )
    .await;
    assert_eq!(r.is_error, Some(false));
    assert_eq!(
        block(&r, 0)["resource"]["uri"],
        "qmd://docs/External%20Podcast/2023%20April%20-%20Interview.md"
    );
    mcp.shutdown().await;
}
