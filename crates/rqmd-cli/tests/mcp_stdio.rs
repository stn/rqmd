//! End-to-end stdio test: spawn the real `rqmd mcp` binary and drive it with an
//! rmcp client over its actual stdin/stdout.
//!
//! qmd's `test/mcp.test.ts` only runs the server over real HTTP (no stdio
//! subprocess), and the in-process `rqmd-mcp` tests wire the handler to a
//! `tokio::io::duplex` pipe — so nothing exercises the *default* transport MCP
//! clients actually use: the spawned `rqmd mcp` process speaking JSON-RPC over
//! stdio. This test seeds an index via the CLI, launches `rqmd mcp`, and replays
//! the same parity checks as the HTTP roundtrip (server name `rqmd`, tool list,
//! `status`, `get`, and the absolute-line `query`).
//!
//! Assertions are LLM-free: `status`/`get` are pure SQL and the `query` uses a
//! `lex` sub-query with `rerank:false`, which `structured_search` resolves
//! without loading a model.

use std::path::Path;
use std::process::Command;

use rmcp::serve_client;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde_json::{Value, json};

/// Pad line used to push the absolute-line marker past the first chunk boundary
/// (300 × 37 chars > CHUNK_SIZE_CHARS). Matches `rqmd-mcp`'s `seed_store`.
const PAD: &str = "Pad line for chunk boundary coverage\n";

/// Write the same documents as `rqmd-mcp`'s `seed_store` into `<dir>/docs`.
fn write_fixtures(dir: &Path) {
    let docs = dir.join("docs");
    std::fs::create_dir_all(docs.join("meetings")).unwrap();
    std::fs::write(
        docs.join("readme.md"),
        "# Project README\n\nThis is the main readme file for the project.\n\nIt contains important information about setup and usage.",
    )
    .unwrap();
    std::fs::write(
        docs.join("api.md"),
        "# API Documentation\n\nThis document describes the REST API endpoints.\n\n## Authentication\n\nUse Bearer tokens for auth.",
    )
    .unwrap();
    std::fs::write(
        docs.join("meetings/meeting-2024-01.md"),
        "# January Meeting Notes\n\nDiscussed Q1 goals and roadmap.",
    )
    .unwrap();
    std::fs::write(
        docs.join("meetings/meeting-2024-02.md"),
        "# February Meeting Notes\n\nFollowed up on Q1 progress.",
    )
    .unwrap();
    std::fs::write(
        docs.join("large-file.md"),
        format!("# Large Document\n\n{}", "Lorem ipsum ".repeat(2000)),
    )
    .unwrap();
    // Marker sits on absolute line 301.
    let abs = format!(
        "{}{}{}",
        PAD.repeat(300),
        "UNIQUE_KEYWORD_XYZ marker\n",
        PAD.repeat(20)
    );
    std::fs::write(docs.join("absolute-line-fixture.md"), abs).unwrap();
}

fn call(name: &str, args: Value) -> rmcp::model::CallToolRequestParams {
    serde_json::from_value(json!({ "name": name, "arguments": args })).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_stdio_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let docs = root.join("docs");
    let cfg = root.join("config");
    let db = root.join("index.sqlite");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::write(cfg.join("index.yml"), "collections: {}\n").unwrap();
    write_fixtures(root);

    // --- seed the index via the CLI (synchronous; `collection add` indexes) ---
    // `--name docs` pins the collection name so display paths are `docs/...`
    // regardless of the temp dir's basename.
    let out = Command::new(env!("CARGO_BIN_EXE_rqmd"))
        .args([
            "--index",
            "index",
            "collection",
            "add",
            docs.to_str().unwrap(),
            "--name",
            "docs",
        ])
        .env("RQMD_INDEX_PATH", &db)
        .env("RQMD_CONFIG_DIR", &cfg)
        .env("CI", "1")
        .env("NO_COLOR", "1")
        // Match the CLI harness's hermeticity (common/mod.rs): keep ambient
        // model/config caches from leaking into the test.
        .env_remove("RUST_LOG")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("RQMD_CACHE_DIR")
        .current_dir(root)
        .output()
        .expect("spawn `rqmd collection add`");
    assert!(
        out.status.success(),
        "collection add failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // --- spawn the real `rqmd mcp` over stdio and connect an rmcp client ---
    // stderr defaults to inherit (TokioChildProcess), so any logging stays off
    // the stdout JSON-RPC stream; `CI=1` + no `RUST_LOG` keeps it quiet anyway.
    let transport = TokioChildProcess::new(
        tokio::process::Command::new(env!("CARGO_BIN_EXE_rqmd")).configure(|cmd| {
            cmd.args(["--index", "index", "mcp"])
                .env("RQMD_INDEX_PATH", &db)
                .env("RQMD_CONFIG_DIR", &cfg)
                .env("CI", "1")
                .env("NO_COLOR", "1")
                .env_remove("RUST_LOG")
                .env_remove("XDG_CACHE_HOME")
                .env_remove("XDG_CONFIG_HOME")
                .env_remove("RQMD_CACHE_DIR")
                .current_dir(root);
        }),
    )
    .expect("spawn `rqmd mcp`");
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
}
