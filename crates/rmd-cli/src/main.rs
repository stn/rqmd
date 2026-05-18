//! `rmd` — command-line interface.
//!
//! Maps to qmd's `src/cli/qmd.ts` (3828 lines). PR1 wired `pull` + `embed`;
//! PR2 wires `search` + `vsearch` + `query`. Only `mcp` remains stubbed.

use anyhow::Result;
use clap::Parser;

mod cli;
mod color;
mod commands;
mod format_helpers;
mod output;
mod search_view;
mod state;

use cli::{Cli, Command};
use state::IndexState;

// `rmd-llm` exposes async fns; `#[tokio::main]` provides the runtime.
// Sync commands keep their original signature and are simply called without
// `.await` from inside this async dispatcher.
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Cli::parse();

    // Flip to production mode so default_db_path() returns a real path
    // instead of the test-only DbPathNotSet error.
    rmd_core::store::path::enable_production_mode();

    if args.no_gpu {
        // SAFETY: single-threaded at this point — tokio worker threads only
        // come into play once we hit an async LLM call below.
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

        Command::Pull(a) => commands::pull::run(a, &mut state, &palette).await,
        Command::Embed(a) => commands::embed::run(a, &mut state, &palette).await,
        Command::Search(a) => commands::search::run(a, &mut state, &palette),
        Command::Vsearch(a) => commands::vsearch::run(a, &mut state, &palette).await,
        Command::Query(a) => commands::query::run(a, &mut state, &palette).await,

        Command::Mcp(_) => commands::llm_stub::run("mcp"),
    }
}
