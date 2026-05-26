//! `rqmd` — command-line interface.
//!
//! Maps to qmd's `src/cli/qmd.ts` (3828 lines). PR1 wired `pull` + `embed`;
//! PR2 wires `search` + `vsearch` + `query`; `mcp` is wired to `rqmd-mcp`.

use anyhow::Result;
use clap::Parser;

mod cli;
mod collection_filter;
mod color;
mod commands;
mod format_helpers;
mod output;
mod search_view;
mod state;

use cli::{Cli, Command};
use state::IndexState;

// `rqmd_core::llm` / `rqmd_core::store_ops` expose async fns; `#[tokio::main]` provides the runtime.
// Sync commands keep their original signature and are simply called without
// `.await` from inside this async dispatcher.
#[tokio::main(flavor = "multi_thread")]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

/// Install a `tracing` subscriber gated on `RUST_LOG`. When the env var is
/// unset the subscriber drops all events (default off — keeps the CLI quiet
/// for end users). Setting `RUST_LOG=rqmd_core::llm=debug` (for example)
/// surfaces the `expand_query` prompt-debug logs and the invalid-env-var
/// warnings on stderr. Failures are non-fatal — a duplicate-init or missing
/// env merely means logging stays at the previous (or default) level.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("off"));
    let _ = fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();
}

async fn run() -> Result<()> {
    let args = Cli::parse();

    init_tracing();

    // Flip to production mode so default_db_path() returns a real path
    // instead of the test-only DbPathNotSet error.
    rqmd_core::store::path::enable_production_mode();

    if args.no_gpu {
        // SAFETY: single-threaded at this point — tokio worker threads only
        // come into play once we hit an async LLM call below.
        unsafe { std::env::set_var("QMD_FORCE_CPU", "1") };
    }

    let palette = color::Palette::new(args.no_color);

    // `--skill` alias: print the bundled skill and exit (no index/store needed).
    // Checked before the missing-subcommand handling below so `rqmd --skill`
    // (which intentionally takes no subcommand) works.
    if args.skill {
        commands::skill::show(&palette);
        return Ok(());
    }

    // `command` is `Option` so the `--skill` alias can run with no subcommand.
    // A bare `rqmd` (no subcommand, no `--skill`) reproduces clap's own
    // missing-subcommand usage error (stderr, exit 2) — matching the prior
    // behaviour when `command` was required.
    let Some(command) = args.command else {
        use clap::CommandFactory;
        Cli::command()
            .error(
                clap::error::ErrorKind::MissingSubcommand,
                "a subcommand is required",
            )
            .exit();
    };

    let mut state = IndexState::new(args.index.as_deref());

    // Bind the dispatch result so we can dispose the LLM workers *after* the
    // command completes (and the `&mut state` borrow inside each match arm
    // ends), regardless of Ok/Err. `state.close(self)` consumes `self`, so it
    // must come after the match. Panics still skip `close` — same as pre-PR
    // behaviour; the OS reclaims worker threads on process exit.
    let result: Result<()> = match command {
        Command::Init => commands::init::run(&palette),
        Command::Collection(sub) => commands::collection::run(sub, &mut state, &palette),
        Command::Context(sub) => commands::context::run(sub, &mut state, &palette),
        Command::Get(a) => commands::get::run(a, &mut state, &palette),
        Command::MultiGet(a) => commands::multi_get::run(a, &mut state),
        Command::Ls(a) => commands::ls::run(a, &mut state, &palette),
        Command::Status => commands::status::run(&mut state, &palette),
        Command::Doctor => commands::doctor::run(&mut state, &palette).await,
        Command::Update => commands::update::run(&mut state, &palette),
        Command::Cleanup => commands::cleanup::run(&mut state, &palette),

        Command::Pull(a) => commands::pull::run(a, &mut state, &palette).await,
        Command::Embed(a) => commands::embed::run(a, &mut state).await,
        Command::Search(a) => commands::search::run(a, &mut state, &palette),
        Command::Vsearch(a) => commands::vsearch::run(a, &mut state, &palette).await,
        Command::Query(a) => commands::query::run(a, &mut state, &palette).await,
        Command::Bench(a) => commands::bench::run(a, &mut state).await,

        Command::Mcp(a) => commands::mcp::run(a, &mut state).await,
        Command::Skill(sub) => commands::skill::run_skill(sub, &palette),
        Command::Skills(sub) => commands::skill::run_skills(sub, &palette),
    };
    state.close().await;
    result
}
