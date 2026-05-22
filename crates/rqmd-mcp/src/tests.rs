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
//! against the spawned `rqmd mcp` binary lives in `rqmd-cli/tests/mcp_stdio.rs`.
//!
//! All assertions here are LLM-free: `status`/`get`/`read_resource` are pure
//! SQL, and the `query` cases use `lex` sub-queries with `rerank: false`, which
//! `structured_search` resolves without loading a model (see store_ops).

use std::path::Path;

use serde_json::{Value, json};

use rqmd_core::{AddCollectionOptions, RqmdStore, RqmdStoreOptions, UpdateOptions};

use crate::server::QmdMcpServer;

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
    let abs = format!("{}{}{}", pad.repeat(300), "UNIQUE_KEYWORD_XYZ marker\n", pad.repeat(20));
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
    store
}

fn call(name: &str, args: Value) -> rmcp::model::CallToolRequestParams {
    serde_json::from_value(json!({ "name": name, "arguments": args })).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn stdio_protocol_roundtrip() {
    use rmcp::{serve_client, serve_server};

    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_store(tmp.path()).await);

    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let (srv, cli) = tokio::join!(
        serve_server(server, server_io),
        serve_client((), client_io),
    );
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
    let status = client.call_tool(call("status", json!({}))).await.expect("status");
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
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
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
        .oneshot(Request::builder().uri("/other").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // POST /query -> { results: [...] }
    let body = serde_json::to_vec(&json!({ "searches": [{ "type": "lex", "query": "readme" }] })).unwrap();
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
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
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
    let status = client.call_tool(call("status", json!({}))).await.expect("status");
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
