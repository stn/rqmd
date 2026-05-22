//! `rqmd mcp` — start the Model Context Protocol server.
//!
//! Maps to qmd's `mcp` CLI handler, which calls `startMcpServer` /
//! `startMcpHttpServer` from `src/mcp/server.ts`. The protocol + transports live
//! in the `rqmd-mcp` crate; this handler just resolves the active index into an
//! [`RqmdStore`] and hands it off.
//!
//! - `rqmd mcp` — stdio transport (default; what MCP clients use).
//! - `rqmd mcp --http [--port N]` — Streamable HTTP at `/mcp` (default port 8181)
//!   plus the REST `/health`, `/query`, `/search` endpoints.

use anyhow::{Context, Result};

use rqmd_core::RqmdStore;

use crate::cli::McpArgs;
use crate::state::IndexState;

/// Default HTTP port (matches qmd's `mcp --http` default).
const DEFAULT_HTTP_PORT: u16 = 8181;

pub async fn run(args: McpArgs, state: &mut IndexState) -> Result<()> {
    let options = state.rqmd_store_options()?;
    let store = RqmdStore::open(options).context("opening index for the MCP server")?;

    if args.http {
        let port = args.port.unwrap_or(DEFAULT_HTTP_PORT);
        rqmd_mcp::serve_http(port, store, false).await?;
    } else {
        rqmd_mcp::serve_stdio(store).await?;
    }
    Ok(())
}
