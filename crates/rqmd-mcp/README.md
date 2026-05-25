# rqmd-mcp

The Model Context Protocol (MCP) server library for
[rqmd](https://crates.io/crates/rqmd) — on-device hybrid search for markdown, a
Rust port of [tobi/qmd](https://github.com/tobi/qmd).

This crate exposes rqmd's search over MCP so editors and agents can query your
markdown index. It offers two transports:

- **stdio** — for editors/agents that spawn the server as a child process.
- **HTTP** — an [axum](https://crates.io/crates/axum)-hosted Streamable HTTP
  endpoint at `/mcp`, plus REST `/health`, `/query`, and `/search` routes. The
  HTTP daemon keeps models warm in VRAM across requests.

It is normally launched via the `rqmd mcp` subcommand of the
[`rqmd`](https://crates.io/crates/rqmd) CLI; the search engine itself lives in
[`rqmd-core`](https://crates.io/crates/rqmd-core).

## License

MIT. See [LICENSE](https://github.com/stn/rqmd/blob/main/LICENSE) and
[NOTICE](https://github.com/stn/rqmd/blob/main/NOTICE) (attribution to the
upstream qmd project).
