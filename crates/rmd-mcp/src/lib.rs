//! `rmd-mcp` — MCP server library for the `rmd` workspace.
//!
//! Maps to `src/mcp/server.ts` in the original `tobi/qmd` TypeScript
//! implementation. Exposed as a library and launched from `rmd-cli` via
//! the `rmd mcp` subcommand.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
