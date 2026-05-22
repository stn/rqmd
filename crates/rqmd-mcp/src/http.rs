//! Axum HTTP layer: the MCP Streamable HTTP endpoint at `/mcp` plus the non-MCP
//! REST endpoints `/health`, `/query`, and `/search`.
//!
//! Port of the HTTP transport in `tobi/qmd/src/mcp/server.ts:579-852`.
//!
//! ## Streamable HTTP framing
//!
//! qmd sets `enableJsonResponse: true` (JSON, not SSE) while also keeping
//! sessions. rmcp couples its JSON-direct mode to *stateless* mode, so the two
//! can't be combined. We keep rmcp's default **stateful** mode (sessions + SSE
//! framing) for `/mcp` — that's the canonical Streamable HTTP behavior MCP
//! clients expect, and the SSE-vs-JSON framing is invisible to a compliant
//! client. Plain-JSON callers use the `/query` (`/search`) REST endpoint, which
//! returns the same `SearchResultItem[]` shape qmd's REST endpoint does.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};

use crate::server::QmdMcpServer;
use crate::tools::QueryArgs;
use crate::worker::StoreHandle;

#[derive(Clone)]
struct AppState {
    handle: StoreHandle,
    start: Instant,
}

/// Build the axum router. The MCP `/mcp` service shares `server`'s store via a
/// per-session clone (all sessions serialize through the one async mutex).
pub fn router(server: QmdMcpServer) -> Router {
    // Grab the shared store handle for the REST endpoints before `server` is
    // moved into the MCP service's per-session factory.
    let state = AppState {
        handle: server.handle(),
        start: Instant::now(),
    };

    let mcp_service = StreamableHttpService::new(
        // Per-session factory: every session gets a clone sharing the store.
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );

    Router::new()
        .route("/health", get(health))
        .route("/query", post(query_rest))
        .route("/search", post(query_rest))
        .nest_service("/mcp", mcp_service)
        .fallback(not_found)
        .with_state(state)
}

async fn health(State(st): State<AppState>) -> Json<Value> {
    Json(json!({ "status": "ok", "uptime": st.start.elapsed().as_secs() }))
}

/// `POST /query` (alias `/search`): structured search without the MCP protocol.
/// Returns `{ results: SearchResultItem[] }`. Mirrors `server.ts:673-725`.
async fn query_rest(State(st): State<AppState>, body: Bytes) -> Response {
    let params: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return error_json(StatusCode::BAD_REQUEST, "Invalid JSON body"),
    };
    // qmd validates `searches` is an array and 400s otherwise (server.ts:678-682).
    if !params.get("searches").map(Value::is_array).unwrap_or(false) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "Missing required field: searches (array)",
        );
    }
    let args: QueryArgs = match serde_json::from_value(params) {
        Ok(a) => a,
        Err(e) => return error_json(StatusCode::BAD_REQUEST, &format!("Invalid request: {e}")),
    };

    match st.handle.run_query(args).await {
        Ok(items) => Json(json!({ "results": items })).into_response(),
        Err(e) => error_json(StatusCode::INTERNAL_SERVER_ERROR, &e.message),
    }
}

async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, "Not Found").into_response()
}

fn error_json(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}
