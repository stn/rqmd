//! Stub for the still-unimplemented `mcp` command. After PR2, only this
//! subcommand needs the placeholder — `search` / `vsearch` / `query` /
//! `embed` / `pull` are wired up. Exit 2 distinguishes "not implemented"
//! from real errors.

use anyhow::Result;

pub fn run(cmd: &str) -> Result<()> {
    eprintln!("error: '{cmd}' is not yet implemented (rmd-mcp wiring pending)");
    std::process::exit(2);
}
