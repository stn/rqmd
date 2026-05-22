//! `rqmd-mcp` — MCP server library for the `rqmd` workspace.
//!
//! Port of `tobi/qmd/src/mcp/server.ts`. Exposes rqmd's search and document
//! retrieval as MCP tools + a `qmd://` resource, over two transports:
//!
//! * [`serve_stdio`] — default; what MCP clients (Claude Code/Desktop, MCP
//!   Inspector) use.
//! * [`serve_http`] — Streamable HTTP at `/mcp` (sessions) plus the non-MCP REST
//!   endpoints `/health`, `/query`, `/search`.
//!
//! Built on the official [`rmcp`] SDK (the analog of qmd's
//! `@modelcontextprotocol/sdk`) with a manual [`QmdMcpServer`] handler so tool
//! schemas, descriptions, annotations, and response shapes match qmd. Launched
//! from `rqmd-cli` via the `rqmd mcp` subcommand.
//!
//! Runtime: requires a tokio runtime (the underlying `RqmdStore` search path is
//! async).

mod http;
mod server;
#[cfg(test)]
mod tests;
mod tools;
mod worker;

use anyhow::Context;
use rmcp::serve_server;

use rqmd_core::RqmdStore;

pub use server::QmdMcpServer;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Serve the MCP protocol over stdio (the default transport). Runs until the
/// client disconnects, then tears down LLM workers.
pub async fn serve_stdio(store: RqmdStore) -> anyhow::Result<()> {
    let server = QmdMcpServer::new(store);
    let running = serve_server(server.clone(), rmcp::transport::io::stdio())
        .await
        .context("failed to start MCP stdio server")?;
    // Returns when the client disconnects / the transport closes.
    let _ = running.waiting().await;
    server.shutdown().await;
    Ok(())
}

/// Serve over Streamable HTTP on `127.0.0.1:port` (port `0` picks an ephemeral
/// port). Mounts `/mcp` plus the REST endpoints; runs until Ctrl-C / SIGTERM.
pub async fn serve_http(port: u16, store: RqmdStore, quiet: bool) -> anyhow::Result<()> {
    let server = QmdMcpServer::new(store);
    let app = http::router(server.clone());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding MCP HTTP server to {addr}"))?;
    let local = listener.local_addr().context("reading local addr")?;
    if !quiet {
        eprintln!("rqmd MCP server listening on http://{local}/mcp");
    }

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("MCP HTTP server error");
    server.shutdown().await;
    result
}

/// Resolve when the process receives Ctrl-C (any platform) or SIGTERM (unix).
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
