//! Stubs for LLM-dependent commands (`search`, `vsearch`, `query`, `embed`,
//! `pull`, `mcp`). Their `clap` definitions exist so `--help` is complete,
//! but actually executing them requires the not-yet-ported `rmd-llm` /
//! `rmd-mcp` crates. Exit 2 distinguishes "not implemented" from real errors.

use anyhow::Result;

pub fn run(cmd: &str) -> Result<()> {
    eprintln!("error: '{cmd}' requires the LLM backend (rmd-llm), which is not yet implemented");
    std::process::exit(2);
}
