# rqmd

On-device hybrid search for your markdown — the `rqmd` command-line tool.

A Rust port of [tobi/qmd](https://github.com/tobi/qmd) by Tobi Lütke. It combines
BM25 full-text search (SQLite FTS5), vector semantic search, and local LLM
re-ranking — all on-device via [llama.cpp](https://github.com/ggerganov/llama.cpp)
with GGUF models. Nothing leaves your machine.

This crate provides the `rqmd` binary and all subcommands (`search`, `vsearch`,
`query`, `collection`, `pull`, `embed`, `mcp`, …). The engine lives in
[`rqmd-core`](https://crates.io/crates/rqmd-core); the MCP server in
[`rqmd-mcp`](https://crates.io/crates/rqmd-mcp).

## Install

```sh
cargo install rqmd
```

Building compiles llama.cpp and the tree-sitter grammars, so you need a C/C++
compiler and CMake (on Windows, the MSVC "Desktop development with C++"
workload). GPU acceleration is opt-in via the `metal`, `cuda`, or `vulkan`
features; the default install is CPU-only.

## Quickstart

```sh
rqmd collection add ./notes --name notes   # index a folder of markdown
rqmd query "how do I rotate the signing key"
```

See the [project README](https://github.com/stn/rqmd) for the full guide.

## License

MIT. See [LICENSE](https://github.com/stn/rqmd/blob/main/LICENSE) and
[NOTICE](https://github.com/stn/rqmd/blob/main/NOTICE) (attribution to the
upstream qmd project).
