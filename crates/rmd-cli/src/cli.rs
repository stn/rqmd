//! clap definitions for the `rmd` binary.
//!
//! Maps to qmd's `parseArgs` block in `src/cli/qmd.ts` (lines 2632–2748).

use clap::{Args, Parser, Subcommand, ValueEnum};

use rmd_core::store::chunking::ChunkStrategy;

#[derive(Debug, Parser)]
#[command(
    name = "rmd",
    version,
    about = "On-device hybrid search for markdown (Rust port of tobi/qmd)",
    after_help = "Note: skill, skills, and bench commands from qmd are not yet implemented in rmd."
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

    #[command(subcommand)]
    pub command: Command,
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

    // ───────── LLM-dependent (stubbed) ─────────
    /// Full-text BM25 search. **Requires rmd-llm (not yet implemented).**
    Search(StubArgs),

    /// Vector similarity search. **Requires rmd-llm (not yet implemented).**
    Vsearch(StubArgs),

    /// Hybrid search with LLM expansion + reranking. **Requires rmd-llm (not yet implemented).**
    Query(StubArgs),

    /// Generate/refresh vector embeddings for indexed documents.
    Embed(EmbedArgs),

    /// Download the configured LLM models from HuggingFace.
    Pull(PullArgs),

    /// Start the MCP server. **Requires rmd-mcp + rmd-llm (not yet implemented).**
    Mcp(McpArgs),
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
    /// rmd does not currently execute it.
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
// LLM stubs (defined for --help completeness only)
// ============================================================================

#[derive(Debug, Args)]
pub struct StubArgs {
    /// Query string (positional, joined by spaces).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub query: Vec<String>,
}

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
    /// Use HTTP transport instead of stdio.
    #[arg(long)]
    pub http: bool,
    /// HTTP port (default 8181).
    #[arg(long)]
    pub port: Option<u16>,
}

// ============================================================================
// shared
// ============================================================================

/// CLI-side mirror of [`ChunkStrategy`] so we can derive `ValueEnum` (the
/// upstream enum lives in `rmd-core` which deliberately doesn't depend on
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

#[derive(Debug, Args, Clone, Copy, Default)]
pub struct FormatFlags {
    #[arg(long, group = "format")]
    pub json: bool,
    #[arg(long, group = "format")]
    pub csv: bool,
    #[arg(long, group = "format")]
    pub md: bool,
    #[arg(long, group = "format")]
    pub xml: bool,
    #[arg(long, group = "format")]
    pub files: bool,
}
