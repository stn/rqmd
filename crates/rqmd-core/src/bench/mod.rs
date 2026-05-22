//! Benchmark harness: scoring + fixture/result types.
//!
//! Port of `tobi/qmd/src/bench/` (`score.ts` + `types.ts`). The CLI-only
//! runner that drives the four search backends lives in `rqmd-cli`
//! (`commands/bench.rs`); only the pure scoring functions and serde data
//! types live here so the parity test (`tests/bench_score.rs`) can exercise
//! the public API directly.

pub mod score;
pub mod types;

pub use score::{ScoreMetrics, normalize_path, paths_match, score_results};
pub use types::{
    BackendResult, BenchmarkFixture, BenchmarkQuery, BenchmarkResult, QueryResult, SummaryStats,
};
