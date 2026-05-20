//! Fixture + result types for the bench harness.
//!
//! Port of `tobi/qmd/src/bench/types.ts`. A fixture defines labelled queries
//! with expected results; the harness runs each query through multiple search
//! backends and records precision/recall/MRR/F1 plus latency.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::score::ScoreMetrics;

/// One test query in a fixture.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkQuery {
    /// Unique identifier for the query.
    pub id: String,
    /// The search query text (may be a multi-line structured query).
    pub query: String,
    /// Query difficulty/type for grouping results (free-form; not validated,
    /// matching the TS union which `JSON.parse` does not enforce).
    #[serde(rename = "type")]
    pub r#type: String,
    /// Human-readable description of what this tests.
    pub description: String,
    /// File paths (relative to collection) expected in results.
    pub expected_files: Vec<String>,
    /// How many of `expected_files` should appear in top-k results.
    pub expected_in_top_k: usize,
}

/// A loaded benchmark fixture file.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkFixture {
    /// Description of the benchmark.
    pub description: String,
    /// Fixture format version.
    pub version: u32,
    /// Optional collection to search within.
    #[serde(default)]
    pub collection: Option<String>,
    /// The test queries.
    pub queries: Vec<BenchmarkQuery>,
}

/// Per-backend result for a single query: the [`ScoreMetrics`] fields plus
/// the bookkeeping the harness layers on top.
#[derive(Debug, Clone, Serialize)]
pub struct BackendResult {
    pub precision_at_k: f64,
    pub recall: f64,
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub f1: f64,
    pub hits_at_k: usize,
    /// Total expected files.
    pub total_expected: usize,
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u128,
    /// Top result file paths (capped at 10, for inspection).
    pub top_files: Vec<String>,
    /// Expected files found anywhere in the result set.
    pub matched_files: Vec<String>,
    /// Expected files missing from the result set.
    pub unmatched_expected_files: Vec<String>,
}

impl BackendResult {
    /// Combine computed [`ScoreMetrics`] with the harness bookkeeping.
    pub fn from_scores(
        scores: ScoreMetrics,
        total_expected: usize,
        latency_ms: u128,
        top_files: Vec<String>,
    ) -> Self {
        Self {
            precision_at_k: scores.precision_at_k,
            recall: scores.recall,
            recall_at_1: scores.recall_at_1,
            recall_at_3: scores.recall_at_3,
            recall_at_5: scores.recall_at_5,
            mrr: scores.mrr,
            f1: scores.f1,
            hits_at_k: scores.hits_at_k,
            total_expected,
            latency_ms,
            top_files,
            matched_files: scores.matched_files,
            unmatched_expected_files: scores.unmatched_expected_files,
        }
    }

    /// All-zero result for a backend that errored (e.g. vector search with no
    /// embeddings). Mirrors the TS `runQuery` catch path (`bench.ts:179-196`).
    pub fn zeroed(
        total_expected: usize,
        latency_ms: u128,
        unmatched_expected_files: Vec<String>,
    ) -> Self {
        Self {
            precision_at_k: 0.0,
            recall: 0.0,
            recall_at_1: 0.0,
            recall_at_3: 0.0,
            recall_at_5: 0.0,
            mrr: 0.0,
            f1: 0.0,
            hits_at_k: 0,
            total_expected,
            latency_ms,
            top_files: Vec::new(),
            matched_files: Vec::new(),
            unmatched_expected_files,
        }
    }
}

/// Results for one query across all backends. `backends` keeps insertion
/// order (bm25, vector, hybrid, full) via [`IndexMap`] + serde_json's
/// `preserve_order`.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub id: String,
    pub query: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub backends: IndexMap<String, BackendResult>,
}

/// Per-backend averaged metrics across all queries.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryStats {
    pub avg_precision: f64,
    pub avg_recall: f64,
    pub avg_recall_at_1: f64,
    pub avg_recall_at_3: f64,
    pub avg_recall_at_5: f64,
    pub avg_mrr: f64,
    pub avg_f1: f64,
    pub avg_latency_ms: f64,
}

/// The full benchmark run output.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub timestamp: String,
    pub fixture: String,
    pub results: Vec<QueryResult>,
    pub summary: IndexMap<String, SummaryStats>,
}
