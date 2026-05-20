//! clap definitions for the `rqmd` binary.
//!
//! Maps to qmd's `parseArgs` block in `src/cli/qmd.ts` (lines 2632–2748).

use clap::{Args, Parser, Subcommand, ValueEnum};

use rqmd_core::store::chunking::ChunkStrategy;

#[derive(Debug, Parser)]
#[command(
    name = "rqmd",
    version,
    about = "On-device hybrid search for markdown (Rust port of tobi/qmd)",
    after_help = "Note: skill, skills, and bench commands from qmd are not yet implemented in rqmd."
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

    /// Full-text BM25 search (no LLM required).
    Search(SearchArgs),

    /// Vector similarity search with automatic query expansion.
    Vsearch(VsearchArgs),

    /// Hybrid search: BM25 + vector + LLM expansion + reranking.
    Query(QueryArgs),

    /// Generate/refresh vector embeddings for indexed documents.
    Embed(EmbedArgs),

    /// Download the configured LLM models from HuggingFace.
    Pull(PullArgs),

    /// Start the MCP server. **Requires rqmd-mcp (not yet implemented).**
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
    /// Restrict to a single collection. Pass at most once.
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
    /// Query string (positional, joined by spaces).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
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
    /// Query string (positional, joined by spaces).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
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
    /// Query string (positional, joined by spaces).
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub query: Vec<String>,
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
