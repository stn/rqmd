# RQMD

On-device hybrid search for your markdown — notes, meeting transcripts, docs,
knowledge bases, whatever you need to remember.

A Rust port of [tobi/qmd](https://github.com/tobi/qmd) by Tobi Lütke. It keeps
qmd's design and search pipeline; it just runs as a native binary.

rqmd combines BM25 full-text search (SQLite FTS5), vector semantic search, and
local LLM re-ranking — all on-device via
[llama.cpp](https://github.com/ggerganov/llama.cpp) with GGUF models. Nothing
leaves your machine.

## Features

- **Three search modes** — `search` (BM25 keyword), `vsearch` (vector + query
  expansion), and `query` (hybrid BM25 + vector + RRF fusion + LLM rerank).
- **Smart chunking** — semantic break detection (headings, code blocks, lists)
  with optional tree-sitter AST awareness for code (TS/JS/Python/Go/Rust).
- **Collections & context** — index folders of markdown, attach human-written
  context summaries to paths.
- **MCP server** — stdio for editors/agents, plus an HTTP daemon that keeps
  models warm in VRAM across requests.
- **Multiple output formats** — `--json`, `--csv`, `--md`, `--xml`, `--files`.
- **Bundled Claude skill** — `rqmd --skill` / `rqmd skill install`.

## Requirements

Building from source compiles llama.cpp and the tree-sitter grammars, so you
need a toolchain:

- A recent stable **Rust** (edition 2024).
- A **C/C++ compiler and CMake** (llama.cpp is built from source). On Windows
  this means the **MSVC Build Tools** with the "Desktop development with C++"
  workload.
- **~2 GB of free disk** for the GGUF models (downloaded on first LLM use), plus
  a few GB of RAM to load them. The first build is slow.

## Install

Install from crates.io — this builds from source and puts the `rqmd` binary in
`~/.cargo/bin`:

```sh
cargo install rqmd
```

Or from a local clone (for development):

```sh
cargo install --path crates/rqmd
```

GPU acceleration is opt-in; the default install is **CPU-only**. Add a backend
feature to either command:

```sh
cargo install rqmd --features metal    # macOS
cargo install rqmd --features cuda     # NVIDIA
cargo install rqmd --features vulkan   # cross-vendor
```

At runtime, `--no-gpu` forces CPU even on a GPU build. For development, run from
a clone with `cargo run -p rqmd -- <args>`.

## Quickstart

```sh
rqmd collection add ./notes --name notes   # index a folder of markdown
rqmd update                                 # (re)index all collections
rqmd pull                                   # download the models (optional; else lazy)
rqmd embed                                  # build vector embeddings
rqmd status                                 # check index health

rqmd search "kafka rebalance"               # fast BM25 keyword search (no models)
rqmd query "why did we pick postgres"       # best quality (hybrid + rerank)
rqmd get notes/decisions.md --from 40 -l 20 # fetch a document slice
```

`search` works immediately. `vsearch` and `query` need embeddings (`rqmd embed`)
and will download the models on first run.

This README covers the commands you'll reach for most. rqmd closely mirrors
qmd's CLI (with a few [documented differences](#differences-from-qmd)), so for
anything not covered here, see [qmd's README](https://github.com/tobi/qmd).

## Excluding files

By default rqmd indexes every file matching a collection's pattern (`**/*.md`).
Like qmd, rqmd reads **no** ignore files — `.gitignore` and `.ignore` are never
consulted, so your VCS and editor config can't change what gets searched.

To skip files or folders, add gitignore-syntax patterns to a collection's
`ignore` list in the index config:

```yaml
collections:
  notes:
    path: /home/me/notes
    pattern: "**/*.md"
    ignore:
      - "old/"             # skip a whole folder
      - "*.excalidraw.md"  # skip files by pattern
      - "drafts/secret.md" # skip one specific file
```

Built-in excludes (`node_modules`, `.git`, `.cache`, `vendor`, `dist`, `build`)
and any path component starting with `.` (e.g. `.obsidian`) are always skipped.
On Windows the hidden *attribute* alone does not exclude a file — only a leading
dot does.

## Search modes

| Command   | What it does                                              | LLM needed |
|-----------|----------------------------------------------------------|------------|
| `search`  | BM25 lexical search over FTS5.                            | no         |
| `vsearch` | Vector similarity search with automatic query expansion. | yes        |
| `query`   | Hybrid: BM25 + vector, fused via RRF, then LLM reranked.  | yes        |

Common flags: `-c <collection>` (repeatable), `-n <limit>`, `--all`,
`--min-score`, `--full`, `--explain` (scoring breakdown). Run
`rqmd <command> --help` for the full set.

## MCP server

```sh
rqmd mcp                          # stdio (Claude Code, Claude Desktop, Inspector)
rqmd mcp --http --port 8181       # foreground HTTP server (Streamable HTTP at /mcp)
rqmd mcp --daemon                 # background HTTP (writes a PID file); implies --http
rqmd mcp stop                     # stop the running daemon
```

Tools exposed: `query`, `get`, `multi_get`, `status`. Documents are served as
`qmd://…` resources. For agent setup, see `rqmd --skill` (the bundled skill
covers Claude Code / Desktop integration).

## Models

Defaults (overridable via `RQMD_EMBED_MODEL` / `RQMD_GENERATE_MODEL` /
`RQMD_RERANK_MODEL`):

| Role            | Model                                                  | Size    |
|-----------------|--------------------------------------------------------|---------|
| Embedding       | `embeddinggemma-300M` (Q8_0 GGUF)                      | ~300 MB |
| Query expansion | `qmd-query-expansion-1.7B` — Tobi's fine-tune (Q4_K_M) | ~1.1 GB |
| Reranking       | `Qwen3-Reranker-0.6B` (Q8_0 GGUF)                      | ~640 MB |

## Where data lives

- **Index** (SQLite): `~/.cache/rqmd/<index>.sqlite`. Override with
  `--index <name>`, `RQMD_INDEX_PATH`, or `RQMD_CACHE_DIR`.
- **Config** (YAML): `~/.config/rqmd/<index>.yml` (or `RQMD_CONFIG_DIR`).
- **Models** (GGUF): the platform cache dir under `qmd/models` —
  `~/Library/Caches/qmd/models` on macOS, `~/.cache/qmd/models` on Linux, or
  `$XDG_CACHE_HOME/qmd/models`. (Models keep qmd's `qmd` namespace for cache
  compatibility; rqmd's own data uses `rqmd`.)

All runtime env vars use the `RQMD_` prefix; run `rqmd doctor` to see the
full list.

## Workspace layout

| Crate       | Role                                                                 | Maps to (qmd)                                            |
|-------------|---------------------------------------------------------------------|---------------------------------------------------------|
| `rqmd-core` | Engine: store (FTS5/db/chunking/AST), store-ops (hybrid/rerank/expand/embed), llm (llama.cpp), bench | `src/store.ts`, `src/db.ts`, `src/ast.ts`, `src/llm.ts` |
| `rqmd-mcp`  | MCP server library (stdio + HTTP), launched via `rqmd mcp`          | `src/mcp/server.ts`                                      |
| `rqmd`      | The `rqmd` binary and all subcommands                               | `src/cli/qmd.ts`                                         |

## Differences from qmd

The port aims to be faithful, not byte-identical. Notable divergences:

- The MCP server identifies itself as `rqmd`, not `qmd`.
- Tree-sitter uses native Rust grammars rather than WASM.
- `rqmd mcp --daemon` always implies `--http`.
- `rqmd update` does not shell out; `collection update-cmd` is stored but not
  executed, so run any pre-update commands (e.g. `git pull`) yourself.

## Migrating from 0.1.x

`0.2.0` renames every runtime env var from `QMD_*` to `RQMD_*` so qmd and
rqmd can coexist on the same machine without configuration crosstalk. If
you have shell exports, systemd units, or wrapper scripts that set any of
`QMD_EMBED_MODEL`, `QMD_GENERATE_MODEL`, `QMD_RERANK_MODEL`,
`QMD_FORCE_CPU`, `QMD_LLAMA_GPU`, `QMD_DOCTOR_DEVICE_PROBE`,
`QMD_EMBED_PARALLELISM`, `QMD_RERANK_PARALLELISM`,
`QMD_EMBED_CONTEXT_SIZE`, `QMD_RERANK_CONTEXT_SIZE`,
`QMD_EXPAND_CONTEXT_SIZE`, `QMD_EXPAND_USER_MESSAGE_PREFIX`,
`QMD_EXPAND_SYSTEM_MESSAGE`, `QMD_EXPAND_FALLBACK_HYDE_TEMPLATE`,
`QMD_EXPAND_TEMP`, `QMD_EXPAND_TOP_K`, `QMD_EXPAND_TOP_P`, or
`QMD_EDITOR_URI`, replace the `QMD_` prefix with `RQMD_`. The previous
`QMD_*` names are no longer read. See `CHANGELOG.md` for the full list.

## Credits & License

This project stands on the shoulders of [qmd](https://github.com/tobi/qmd) — its
design, search pipeline, and the query-expansion model are Tobi Lütke's work.
Thank you.

Licensed under the [MIT License](LICENSE) © 2026 Akira Ishino. The original qmd
is also MIT-licensed © 2024–2026 Tobi Lütke (see [`NOTICE`](NOTICE));
this Rust port inherits and complies with that license.
