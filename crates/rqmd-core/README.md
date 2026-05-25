# rqmd-core

The search-engine core behind [rqmd](https://crates.io/crates/rqmd) — on-device
hybrid search for markdown, a Rust port of
[tobi/qmd](https://github.com/tobi/qmd).

This library crate holds the engine that the `rqmd` CLI and the
[`rqmd-mcp`](https://crates.io/crates/rqmd-mcp) server are built on:

- **store** — SQLite FTS5 + `sqlite-vec` vector index, semantic chunking, and
  optional tree-sitter AST-aware chunking for code (TS/JS/Python/Go/Rust).
- **store-ops** — hybrid query (BM25 + vector + RRF fusion), query expansion,
  re-ranking, and embedding.
- **llm** — local inference via [llama.cpp](https://github.com/ggerganov/llama.cpp)
  (`llama-cpp-2`) with GGUF models pulled from Hugging Face.
- **bench** — scoring/result types for the four search backends.

GPU acceleration is opt-in via the `metal`, `cuda`, and `vulkan` features (each
forwards to `llama-cpp-2`); the default build is CPU-only. Building compiles
llama.cpp from source, so a C/C++ compiler and CMake are required.

## License

MIT. See [LICENSE](https://github.com/stn/rqmd/blob/main/LICENSE) and
[NOTICE](https://github.com/stn/rqmd/blob/main/NOTICE) (attribution to the
upstream qmd project).
