//! LLM-dependent MCP tests — qmd's `searchVec` + `hybridQuery` describe blocks
//! (`tobi/qmd/test/mcp.test.ts:326-435`), driven through the real MCP `query` tool.
//!
//! Unlike the LLM-free tests in `src/tests.rs`, these load real GGUF models: the
//! rqmd/qmd defaults `embeddinggemma-300M` (embedding) and `Qwen3-Reranker-0.6B`
//! (`rerank:true`). They **run by default**; set `RQMD_SKIP_LLM_TESTS=1` to skip
//! (CI does, mirroring qmd's `skipIf(CI)`). All sub-queries are *typed*, so the
//! 1.7B query-expansion model is never loaded.
//!
//! The four qmd scenarios are folded into one `#[test]` so each model loads once
//! and parallel test threads don't trigger concurrent model loads. First run
//! downloads the default models to the model cache.

use std::path::Path;

use rmcp::{serve_client, serve_server};
use serde_json::{Value, json};

use rqmd_core::{
    AddCollectionOptions, RqmdStore, RqmdStoreOptions, StoreOpsEmbedOptions, UpdateOptions,
};
use rqmd_mcp::QmdMcpServer;

/// Run by default; skip when `RQMD_SKIP_LLM_TESTS` is set.
fn skip() -> bool {
    std::env::var("RQMD_SKIP_LLM_TESTS").is_ok()
}

fn call(name: &str, args: Value) -> rmcp::model::CallToolRequestParams {
    serde_json::from_value(json!({ "name": name, "arguments": args })).unwrap()
}

fn structured_results(r: &rmcp::model::CallToolResult) -> Vec<Value> {
    r.structured_content.as_ref().expect("structuredContent")["results"]
        .as_array()
        .expect("results array")
        .clone()
}

/// Seed a small corpus and generate real embeddings with the default embed model.
async fn seed_embedded(dir: &Path) -> RqmdStore {
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("readme.md"),
        "# Project README\n\nThis is the main readme file for the project. It covers setup and usage.",
    )
    .unwrap();
    std::fs::write(
        docs.join("api.md"),
        "# API Documentation\n\nThis describes the REST API endpoints. Authentication uses Bearer tokens.",
    )
    .unwrap();
    std::fs::write(
        docs.join("guide.md"),
        "# User Guide\n\nStep by step instructions for getting started with the application.",
    )
    .unwrap();

    let mut store = RqmdStore::open(RqmdStoreOptions {
        db_path: dir.join("index.sqlite"),
        ..Default::default()
    })
    .unwrap();
    store
        .add_collection(
            "docs",
            AddCollectionOptions {
                path: docs.to_string_lossy().into_owned(),
                pattern: None,
                ignore: None,
            },
        )
        .unwrap();
    store.update(UpdateOptions::default()).await.unwrap();
    // Real embeddings (default embed model) so vec/hyde search has vectors. qmd
    // seeds random vectors directly; rqmd embeds for real via the public API.
    store
        .embed(StoreOpsEmbedOptions::default())
        .await
        .expect("embed with default model");
    store
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_llm_query_pipeline() {
    if skip() {
        eprintln!("RQMD_SKIP_LLM_TESTS set — skipping LLM MCP pipeline test");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let server = QmdMcpServer::new(seed_embedded(tmp.path()).await);
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let (srv, cli) = tokio::join!(serve_server(server, server_io), serve_client((), client_io));
    let _server = srv.expect("server initialized");
    let client = cli.expect("client initialized");

    // qmd searchVec "returns results for semantic query".
    let r = client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "vec", "query": "project documentation" }], "rerank": false }),
        ))
        .await
        .expect("vec query");
    assert!(
        !structured_results(&r).is_empty(),
        "vec query should return results"
    );

    // qmd searchVec "respects limit parameter".
    let r = client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "vec", "query": "documentation" }], "limit": 2, "rerank": false }),
        ))
        .await
        .expect("vec query with limit");
    assert!(structured_results(&r).len() <= 2);

    // qmd hybridQuery "reranks documents with LLM" (rerank:true loads the reranker).
    let r = client
        .call_tool(call(
            "query",
            json!({ "searches": [{ "type": "lex", "query": "API" }], "rerank": true }),
        ))
        .await
        .expect("rerank query");
    let res = structured_results(&r);
    assert!(!res.is_empty(), "rerank query should return results");
    assert!(
        res[0]["score"].as_f64().expect("score number") > 0.0,
        "top rerank score should be > 0"
    );

    // qmd hybridQuery "full hybrid search pipeline" (lex + vec + rerank).
    let r = client
        .call_tool(call(
            "query",
            json!({
                "searches": [
                    { "type": "lex", "query": "API" },
                    { "type": "vec", "query": "how do I authenticate requests" }
                ],
                "rerank": true
            }),
        ))
        .await
        .expect("hybrid query");
    assert!(
        !structured_results(&r).is_empty(),
        "hybrid pipeline should return results"
    );

    client.cancel().await.ok();
}
