# rmd

Rust port of [tobi/qmd](https://github.com/tobi/qmd) — an on-device hybrid search
engine for markdown that combines BM25 (SQLite FTS5), vector semantic search, and
local LLM re-ranking.

> **Status:** Project skeleton only. Implementation is in progress and the CLI
> currently prints a placeholder message.

## Workspace layout

| Crate       | Role                                                       | Maps to (qmd)                              |
|-------------|------------------------------------------------------------|--------------------------------------------|
| `rmd-core`  | Search engine core — store, db, chunking, collections, AST | `src/store.ts`, `src/db.ts`, `src/ast.ts`  |
| `rmd-llm`   | GGUF model integration — embeddings, rerank, query expand  | `src/llm.ts`                               |
| `rmd-mcp`   | MCP server (library, launched via `rmd mcp`)               | `src/mcp/server.ts`                        |
| `rmd-cli`   | CLI binary (`rmd`) — subcommands `search`, `mcp`, etc.     | `src/cli/qmd.ts`                           |

## Build

```sh
cargo build --workspace
cargo run --bin rmd
```

## License

MIT. See [LICENSE](LICENSE). The original qmd is MIT-licensed by Tobi Lutke
(2024-2026); this Rust port inherits and complies with that license.
