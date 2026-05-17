//! `rmd` — command-line interface.
//!
//! Maps to qmd's `src/cli/qmd.ts` (3828 lines). This pass ports the non-LLM
//! commands; `search`, `vsearch`, `query`, `embed`, `pull`, and `mcp` are
//! parsed for `--help` completeness but their handlers exit with a clear
//! "requires rmd-llm" message.

use anyhow::Result;
use clap::Parser;

mod cli;
mod color;
mod commands;
mod format_helpers;
mod output;
mod state;

use cli::{Cli, Command};
use state::IndexState;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Cli::parse();

    // Flip to production mode so default_db_path() returns a real path
    // instead of the test-only DbPathNotSet error.
    rmd_core::store::path::enable_production_mode();

    if args.no_gpu {
        // SAFETY: we are still single-threaded at this point.
        unsafe { std::env::set_var("QMD_FORCE_CPU", "1") };
    }

    let palette = color::Palette::new(args.no_color);
    let mut state = IndexState::new(args.index.as_deref());

    match args.command {
        Command::Collection(sub) => commands::collection::run(sub, &mut state, &palette),
        Command::Context(sub) => commands::context::run(sub, &mut state, &palette),
        Command::Get(a) => commands::get::run(a, &mut state, &palette),
        Command::MultiGet(a) => commands::multi_get::run(a, &mut state),
        Command::Ls(a) => commands::ls::run(a, &mut state, &palette),
        Command::Status => commands::status::run(&mut state, &palette),
        Command::Update => commands::update::run(&mut state, &palette),
        Command::Cleanup => commands::cleanup::run(&mut state, &palette),

        Command::Search(_) => commands::llm_stub::run("search"),
        Command::Vsearch(_) => commands::llm_stub::run("vsearch"),
        Command::Query(_) => commands::llm_stub::run("query"),
        Command::Embed(_) => commands::llm_stub::run("embed"),
        Command::Pull(_) => commands::llm_stub::run("pull"),
        Command::Mcp(_) => commands::llm_stub::run("mcp"),
    }
}
