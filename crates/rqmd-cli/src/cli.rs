//! clap definitions for the `rqmd` binary.
//!
//! Maps to qmd's `parseArgs` block in `src/cli/qmd.ts` (lines 2632–2748).

use clap::{Args, Parser, Subcommand, ValueEnum};

use rqmd_core::store::chunking::ChunkStrategy;

#[derive(Debug, Parser)]
#[command(
    name = "rqmd",
    version,
    about = "On-device hybrid search for markdown (Rust port of tobi/qmd)"
)]
pub struct Cli {
    /// Use a named index (default: "index"). Selects `<name>.sqlite` under the
    /// cache directory and the matching YAML config.
    #[arg(long, global = true, value_name = "NAME")]
    pub index: Option<String>,

    /// Force CPU mode for llama.cpp operations (sets QMD_FORCE_CPU=1). Harmless
    /// for non-LLM subcommands but accepted for forward-compat with qmd.
    #[arg(long, global = true)]
    pub no_gpu: bool,

    /// Disable ANSI colour output (also respects the NO_COLOR env var).
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Print the bundled rqmd skill and exit (alias for `skill show`).
    #[arg(long)]
    pub skill: bool,

    /// `None` enables the `--skill` alias (which takes no subcommand); a missing
    /// subcommand without `--skill` is handled in `main` to reproduce clap's
    /// "missing subcommand" usage error (exit 2).
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage indexed collections (folders of markdown).
    #[command(subcommand)]
    Collection(CollectionCmd),

    /// Attach human-written context summaries to collections / paths.
    #[command(subcommand)]
    Context(ContextCmd),

    /// Show a single document, optionally a line slice.
    Get(GetArgs),

    /// Batch fetch documents by glob or comma-separated list.
    #[command(name = "multi-get")]
    MultiGet(MultiGetArgs),

    /// List collections, or files within a collection.
    Ls(LsArgs),

    /// Show index + collection health.
    Status,

    /// Re-index all collections (no shell-out; run any pre-update commands manually).
    Update,

    /// Clear caches, drop inactive docs, vacuum the database.
    Cleanup,

    /// Full-text BM25 search (no LLM required).
    Search(SearchArgs),

    /// Vector similarity search with automatic query expansion.
    Vsearch(VsearchArgs),

    /// Hybrid search: BM25 + vector + LLM expansion + reranking.
    Query(QueryArgs),

    /// Run a benchmark fixture across all four search backends.
    Bench(BenchArgs),

    /// Generate/refresh vector embeddings for indexed documents.
    Embed(EmbedArgs),

    /// Download the configured LLM models from HuggingFace.
    Pull(PullArgs),

    /// Start the MCP (Model Context Protocol) server (stdio, or `--http`).
    Mcp(McpArgs),

    /// Show or install the bundled rqmd skill (legacy `show`/`install`).
    #[command(subcommand)]
    Skill(SkillCmd),

    /// Inspect bundled runtime skills (`list`/`get`/`path`).
    #[command(subcommand)]
    Skills(SkillsCmd),
}

// ============================================================================
// skill / skills
// ============================================================================

#[derive(Debug, Subcommand)]
pub enum SkillCmd {
    /// Print the bundled rqmd skill.
    Show,
    /// Install the rqmd skill into a project (or `--global` into $HOME).
    Install(SkillInstallArgs),
}

#[derive(Debug, Args)]
pub struct SkillInstallArgs {
    /// Install into `$HOME/.agents/skills/rqmd` instead of `./.agents/skills/rqmd`.
    #[arg(long)]
    pub global: bool,
    /// Also create the `.claude/skills/rqmd` symlink without prompting.
    #[arg(long)]
    pub yes: bool,
    /// Replace an existing install / symlink.
    #[arg(short = 'f', long)]
    pub force: bool,
}

#[derive(Debug, Subcommand)]
pub enum SkillsCmd {
    /// List bundled runtime skills.
    List(SkillsListArgs),
    /// Print a bundled runtime skill.
    Get(SkillsGetArgs),
    /// Print the on-disk path of a bundled runtime skill (or the search dir).
    Path(SkillsPathArgs),
}

#[derive(Debug, Args)]
pub struct SkillsListArgs {
    /// Emit structured JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SkillsGetArgs {
    /// Skill name (rqmd ships a single skill: `rqmd`).
    pub name: Option<String>,
    /// Include references/templates/scripts.
    #[arg(long)]
    pub full: bool,
    /// Print all bundled skills.
    #[arg(long)]
    pub all: bool,
    /// Emit structured JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SkillsPathArgs {
    /// Skill name; omit to print the skills search directory.
    pub name: Option<String>,
}

// ============================================================================
// collection
// ============================================================================

#[derive(Debug, Subcommand)]
pub enum CollectionCmd {
    /// List all configured collections.
    List,
    /// Add a collection and index its files.
    Add(CollectionAddArgs),
    /// Remove a collection (deletes documents + content from the index).
    #[command(alias = "rm")]
    Remove(CollectionRemoveArgs),
    /// Rename a collection.
    #[command(alias = "mv")]
    Rename(CollectionRenameArgs),
    /// Show details for one collection.
    #[command(alias = "info")]
    Show(CollectionShowArgs),
    /// Set or clear the pre-update command (e.g., `git pull`). Stored only;
    /// rqmd does not currently execute it.
    #[command(name = "update-cmd", alias = "set-update")]
    UpdateCmd(CollectionUpdateCmdArgs),
    /// Include the collection in default queries (reset to default).
    Include(CollectionNameArg),
    /// Exclude the collection from default queries.
    Exclude(CollectionNameArg),
}

#[derive(Debug, Args)]
pub struct CollectionAddArgs {
    /// Filesystem path to index (defaults to the current directory).
    pub path: Option<String>,
    /// Collection name (defaults to the basename of `path`).
    #[arg(long)]
    pub name: Option<String>,
    /// Glob pattern (defaults to `**/*.md`).
    #[arg(long)]
    pub mask: Option<String>,
}

#[derive(Debug, Args)]
pub struct CollectionRemoveArgs {
    pub name: String,
}

#[derive(Debug, Args)]
pub struct CollectionRenameArgs {
    pub old: String,
    pub new: String,
}

#[derive(Debug, Args)]
pub struct CollectionShowArgs {
    pub name: String,
}

#[derive(Debug, Args)]
pub struct CollectionUpdateCmdArgs {
    pub name: String,
    /// Command to run before indexing. Omit to clear.
    pub command: Vec<String>,
}

#[derive(Debug, Args)]
pub struct CollectionNameArg {
    pub name: String,
}

// ============================================================================
// context
// ============================================================================

#[derive(Debug, Subcommand)]
pub enum ContextCmd {
    /// Add context to a path. Use `/` for global, `qmd://col/...` for a
    /// virtual path, or a filesystem path (defaults to `.`).
    Add(ContextAddArgs),
    /// List all configured contexts.
    List,
    /// Remove a context entry.
    #[command(alias = "remove")]
    Rm(ContextRmArgs),
}

#[derive(Debug, Args)]
pub struct ContextAddArgs {
    /// Either: a path then text (2+ args), or just text (1 arg, defaults to current directory).
    #[arg(num_args = 1.., required = true)]
    pub args: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ContextRmArgs {
    pub path: String,
}

// ============================================================================
// get / multi-get / ls
// ============================================================================

#[derive(Debug, Args)]
pub struct GetArgs {
    /// File path, virtual path (`qmd://...`), `collection/path`, or docid.
    /// A trailing `:N` is parsed as a line number.
    pub file: String,
    /// Start at this line (1-based).
    #[arg(long)]
    pub from: Option<usize>,
    /// Maximum lines to print.
    #[arg(short = 'l')]
    pub lines: Option<usize>,
    /// Prefix each line with its line number.
    #[arg(long = "line-numbers")]
    pub line_numbers: bool,
}

#[derive(Debug, Args)]
pub struct MultiGetArgs {
    /// Glob (e.g. `**/*.md`) or comma-separated list of paths.
    pub pattern: String,
    /// Maximum lines per file.
    #[arg(short = 'l')]
    pub lines: Option<usize>,
    /// Skip files larger than N bytes (default 10240).
    #[arg(long = "max-bytes")]
    pub max_bytes: Option<usize>,
    #[command(flatten)]
    pub format: FormatFlags,
}

#[derive(Debug, Args)]
pub struct LsArgs {
    /// `<collection[/path]>`. With no argument, lists all collections.
    pub path: Option<String>,
}

// ============================================================================
// search / vsearch / query
// ============================================================================

/// Flags common to `search`, `vsearch`, and `query`. `FormatFlags` is
/// flattened separately at the `*Args` level (not here) — see the comment
/// on [`FormatFlags`] for why.
#[derive(Debug, Args, Clone)]
pub struct SearchFlags {
    /// Restrict to one or more collections. Repeat `-c` for several; omit to
    /// search the default collections (those not excluded via `collection exclude`).
    #[arg(short = 'c', long)]
    pub collection: Vec<String>,
    /// Limit results.
    #[arg(short = 'n', long)]
    pub limit: Option<usize>,
    /// Effectively no limit (set to 500/100000 internally depending on subcommand).
    #[arg(long)]
    pub all: bool,
    /// Minimum normalized score (0.0-1.0). Defaults: search=0.0, vsearch=0.3, query=0.0.
    #[arg(long = "min-score")]
    pub min_score: Option<f64>,
    /// Show full body instead of a snippet.
    #[arg(long)]
    pub full: bool,
    /// Prefix snippet lines with 1-indexed line numbers.
    #[arg(long = "line-numbers")]
    pub line_numbers: bool,
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    #[command(flatten)]
    pub flags: SearchFlags,
    #[command(flatten)]
    pub format: FormatFlags,
    /// Query string (positional, joined by spaces). Flags may appear before or
    /// after the query words; all non-flag tokens are joined as the query
    /// (qmd parity, matching the documented SKILL.md usage). A query token
    /// starting with `-` must be escaped via `--`, e.g. `rqmd search -- -foo`.
    /// Value-taking options consume the next token (e.g. `-c concepts meeting`
    /// → collection=concepts, query=meeting), so place such flags accordingly.
    pub query: Vec<String>,
}

#[derive(Debug, Args)]
pub struct VsearchArgs {
    #[command(flatten)]
    pub flags: SearchFlags,
    #[command(flatten)]
    pub format: FormatFlags,
    /// Domain intent (steers the vector / hyde expansion).
    #[arg(long)]
    pub intent: Option<String>,
    /// Query string (positional, joined by spaces). Flags may appear before or
    /// after the query words; all non-flag tokens are joined as the query
    /// (qmd parity, matching the documented SKILL.md usage). A query token
    /// starting with `-` must be escaped via `--`, e.g. `rqmd search -- -foo`.
    /// Value-taking options consume the next token (e.g. `-c concepts meeting`
    /// → collection=concepts, query=meeting), so place such flags accordingly.
    pub query: Vec<String>,
}

#[derive(Debug, Args)]
pub struct QueryArgs {
    #[command(flatten)]
    pub flags: SearchFlags,
    #[command(flatten)]
    pub format: FormatFlags,
    /// Domain intent.
    #[arg(long)]
    pub intent: Option<String>,
    /// Number of candidates passed to the LLM reranker.
    #[arg(short = 'C', long = "candidate-limit")]
    pub candidate_limit: Option<usize>,
    /// Skip LLM reranking and return RRF-blended ranks.
    #[arg(long = "no-rerank")]
    pub no_rerank: bool,
    /// Include score trace in output (JSON only; CLI shows a brief summary).
    #[arg(long)]
    pub explain: bool,
    /// Chunking strategy override (reuses the `embed` enum).
    #[arg(long = "chunk-strategy", value_enum)]
    pub chunk_strategy: Option<ChunkStrategyArg>,
    /// Query string (positional, joined by spaces). Flags may appear before or
    /// after the query words; all non-flag tokens are joined as the query
    /// (qmd parity, matching the documented SKILL.md usage). A query token
    /// starting with `-` must be escaped via `--`, e.g. `rqmd search -- -foo`.
    /// Value-taking options consume the next token (e.g. `-c concepts meeting`
    /// → collection=concepts, query=meeting), so place such flags accordingly.
    pub query: Vec<String>,
}

// ============================================================================
// bench
// ============================================================================

#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Path to the benchmark fixture JSON. See
    /// `crates/rqmd-core/src/bench/fixtures/example.json` for the format.
    pub fixture: String,
    /// Emit the full result as JSON instead of the ASCII table + summary.
    #[arg(long)]
    pub json: bool,
    /// Restrict the benchmark to a single collection (overrides the fixture's
    /// `collection` field).
    #[arg(short = 'c', long)]
    pub collection: Option<String>,
}

// ============================================================================
// embed / pull / mcp
// ============================================================================

#[derive(Debug, Args)]
pub struct EmbedArgs {
    /// Drop existing embeddings and rebuild from scratch.
    #[arg(short = 'f', long)]
    pub force: bool,
    /// Documents per embed batch (default: crate default).
    #[arg(long = "max-docs-per-batch")]
    pub max_docs_per_batch: Option<usize>,
    /// Total payload cap per batch in megabytes (default: crate default).
    #[arg(long = "max-batch-mb")]
    pub max_batch_mb: Option<usize>,
    /// Chunking strategy.
    #[arg(long = "chunk-strategy", value_enum)]
    pub chunk_strategy: Option<ChunkStrategyArg>,
    /// Restrict to a single collection. Pass at most once.
    #[arg(short = 'c', long = "collection")]
    pub collection: Vec<String>,
}

#[derive(Debug, Args)]
pub struct PullArgs {
    /// Re-download even if a cached copy exists (ETag-checked).
    #[arg(long)]
    pub refresh: bool,
}

#[derive(Debug, Args)]
pub struct McpArgs {
    /// Lifecycle verb: `stop` to stop a running HTTP daemon. Omit to start.
    #[arg(value_enum)]
    pub action: Option<McpAction>,
    /// Use HTTP transport instead of stdio.
    #[arg(long)]
    pub http: bool,
    /// HTTP port (default 8181).
    #[arg(long)]
    pub port: Option<u16>,
    /// Run the HTTP server in the background (writes a PID file under the cache
    /// dir). Implies `--http`.
    #[arg(long)]
    pub daemon: bool,
}

/// Lifecycle verb for `rqmd mcp <action>`. A restricted enum (rather than a raw
/// string) so unknown verbs produce a clap "invalid value" error.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum McpAction {
    /// Stop a running MCP HTTP daemon (kills the PID file's process).
    Stop,
}

// ============================================================================
// shared
// ============================================================================

/// CLI-side mirror of [`ChunkStrategy`] so we can derive `ValueEnum` (the
/// upstream enum lives in `rqmd-core` which deliberately doesn't depend on
/// `clap`).
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ChunkStrategyArg {
    Auto,
    Regex,
}

impl From<ChunkStrategyArg> for ChunkStrategy {
    fn from(v: ChunkStrategyArg) -> Self {
        match v {
            ChunkStrategyArg::Auto => ChunkStrategy::Auto,
            ChunkStrategyArg::Regex => ChunkStrategy::Regex,
        }
    }
}

#[cfg(test)]
mod arg_order_tests {
    //! Argument-order parsing for `search`/`vsearch`/`query`. Guards the qmd
    //! parity fix: the variadic `query` positional carries no `trailing_var_arg`
    //! / `allow_hyphen_values`, so flags parse before *or after* the query words
    //! instead of being swallowed. `vsearch`/`query` have no e2e coverage (they
    //! need local models), so these unit tests are their only arg-parse guard.
    use super::*;

    /// Parse argv into a `Command`, surfacing the clap error in the panic.
    fn cmd(argv: &[&str]) -> Command {
        Cli::try_parse_from(argv)
            .unwrap_or_else(|e| panic!("parse failed for {argv:?}: {e}"))
            .command
            .expect("subcommand present")
    }

    #[test]
    fn search_flag_after_query_is_parsed_not_swallowed() {
        let Command::Search(a) = cmd(&["rqmd", "search", "foo", "bar", "-n", "5"]) else {
            panic!("expected search");
        };
        assert_eq!(a.query, ["foo", "bar"]);
        assert_eq!(a.flags.limit, Some(5));
    }

    #[test]
    fn search_flag_before_query_still_works() {
        let Command::Search(a) = cmd(&["rqmd", "search", "-n", "5", "foo", "bar"]) else {
            panic!("expected search");
        };
        assert_eq!(a.query, ["foo", "bar"]);
        assert_eq!(a.flags.limit, Some(5));
    }

    #[test]
    fn search_value_option_consumes_next_token_then_query() {
        // `-c concepts meeting` → collection=[concepts], query=[meeting].
        let Command::Search(a) = cmd(&["rqmd", "search", "-c", "concepts", "meeting"]) else {
            panic!("expected search");
        };
        assert_eq!(a.flags.collection, ["concepts"]);
        assert_eq!(a.query, ["meeting"]);
    }

    #[test]
    fn search_format_flag_after_query() {
        let Command::Search(a) = cmd(&["rqmd", "search", "foo", "--json"]) else {
            panic!("expected search");
        };
        assert_eq!(a.query, ["foo"]);
        assert!(a.format.json);
    }

    #[test]
    fn search_hyphen_leading_query_needs_double_dash_escape() {
        // Bare `-foo` is an unknown flag (exit 2); `-- -foo` is taken literally.
        assert!(Cli::try_parse_from(["rqmd", "search", "-foo"]).is_err());
        let Command::Search(a) = cmd(&["rqmd", "search", "--", "-foo"]) else {
            panic!("expected search");
        };
        assert_eq!(a.query, ["-foo"]);
    }

    #[test]
    fn vsearch_flag_after_query_is_parsed() {
        let Command::Vsearch(a) = cmd(&["rqmd", "vsearch", "foo", "bar", "-n", "3"]) else {
            panic!("expected vsearch");
        };
        assert_eq!(a.query, ["foo", "bar"]);
        assert_eq!(a.flags.limit, Some(3));
    }

    #[test]
    fn query_flag_after_query_is_parsed() {
        let Command::Query(a) = cmd(&["rqmd", "query", "foo", "bar", "-n", "7"]) else {
            panic!("expected query");
        };
        assert_eq!(a.query, ["foo", "bar"]);
        assert_eq!(a.flags.limit, Some(7));
    }

    #[test]
    fn query_intent_flag_after_query_is_parsed() {
        let Command::Query(a) = cmd(&["rqmd", "query", "decision", "quality", "--intent", "find"])
        else {
            panic!("expected query");
        };
        assert_eq!(a.query, ["decision", "quality"]);
        assert_eq!(a.intent.as_deref(), Some("find"));
    }
}

/// Group declared at struct level (not per-field) so the mutual-exclusion
/// survives being nested twice — `FormatFlags` is flattened into both
/// `MultiGetArgs` (one level) and `SearchFlags` → `SearchArgs` (two levels);
/// per-field `group = "format"` was silently dropped at the second level.
#[derive(Debug, Args, Clone, Copy, Default)]
#[group(id = "format", multiple = false)]
pub struct FormatFlags {
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub csv: bool,
    #[arg(long)]
    pub md: bool,
    #[arg(long)]
    pub xml: bool,
    #[arg(long)]
    pub files: bool,
}
