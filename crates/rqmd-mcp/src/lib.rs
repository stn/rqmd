//! `rqmd-mcp` — MCP server library for the `rqmd` workspace.
//!
//! Maps to `src/mcp/server.ts` in the original `tobi/qmd` TypeScript
//! implementation. Exposed as a library and launched from `rqmd-cli` via
//! the `rqmd mcp` subcommand.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
