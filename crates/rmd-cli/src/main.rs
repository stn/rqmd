//! `rmd` — CLI entry point.
//!
//! Maps to `src/cli/qmd.ts` in the original `tobi/qmd` TypeScript
//! implementation. Subcommands (`search`, `get`, `collection`, `mcp`, ...)
//! will be added in subsequent phases.

fn main() {
    println!(
        "rmd v{} — not yet implemented (core v{}, llm v{}, mcp v{})",
        env!("CARGO_PKG_VERSION"),
        rmd_core::VERSION,
        rmd_llm::VERSION,
        rmd_mcp::VERSION,
    );
}
