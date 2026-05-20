//! Scoring functions for the bench harness.
//!
//! Port of `tobi/qmd/src/bench/score.ts`. Computes precision@k, recall, MRR,
//! and F1 for search results against ground-truth expected files.

use serde::Serialize;

/// Normalize a file path for comparison: strip the `qmd://<collection>/`
/// scheme, lowercase, and trim leading/trailing slashes.
///
/// `qmd://collection/docs/readme.md` → `docs/readme.md`.
pub fn normalize_path(p: &str) -> String {
    let stripped = if let Some(rest) = p.strip_prefix("qmd://") {
        // Drop the collection segment after the scheme.
        match rest.find('/') {
            Some(idx) => &rest[idx + 1..],
            None => rest,
        }
    } else {
        p
    };
    stripped.to_lowercase().trim_matches('/').to_string()
}

/// Whether two paths refer to the same file. Compares normalized forms by
/// equality or either-direction suffix match (handles relative vs absolute
/// and differing path formats).
pub fn paths_match(result: &str, expected: &str) -> bool {
    let nr = normalize_path(result);
    let ne = normalize_path(expected);
    nr == ne || nr.ends_with(&ne) || ne.ends_with(&nr)
}

/// Metrics for one query's results against its expected files. Mirrors the
/// TS `ScoreMetrics` shape.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreMetrics {
    pub precision_at_k: f64,
    pub recall: f64,
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub f1: f64,
    pub hits_at_k: usize,
    pub matched_files: Vec<String>,
    pub unmatched_expected_files: Vec<String>,
}

/// Count how many `expected_files` appear within the top-`k` of `result_files`.
fn hits_within(result_files: &[String], expected_files: &[String], k: usize) -> usize {
    let top_k = &result_files[..result_files.len().min(k)];
    expected_files
        .iter()
        .filter(|expected| top_k.iter().any(|r| paths_match(r, expected)))
        .count()
}

/// Score a set of search results against expected files. Port of
/// `scoreResults` (`score.ts:61-111`).
pub fn score_results(
    result_files: &[String],
    expected_files: &[String],
    top_k: usize,
) -> ScoreMetrics {
    let hits_at_k = hits_within(result_files, expected_files, top_k);

    let mut matched_files = Vec::new();
    let mut unmatched_expected_files = Vec::new();
    for expected in expected_files {
        if result_files.iter().any(|r| paths_match(r, expected)) {
            matched_files.push(expected.clone());
        } else {
            unmatched_expected_files.push(expected.clone());
        }
    }

    // MRR: reciprocal rank of the first relevant result.
    let mut mrr = 0.0;
    for (i, r) in result_files.iter().enumerate() {
        if expected_files.iter().any(|e| paths_match(r, e)) {
            mrr = 1.0 / (i as f64 + 1.0);
            break;
        }
    }

    let expected_len = expected_files.len();
    let denominator = top_k.min(expected_len);
    let precision_at_k = if denominator > 0 {
        hits_at_k as f64 / denominator as f64
    } else {
        0.0
    };
    let recall = if expected_len > 0 {
        matched_files.len() as f64 / expected_len as f64
    } else {
        0.0
    };
    let recall_at = |k: usize| {
        if expected_len > 0 {
            hits_within(result_files, expected_files, k) as f64 / expected_len as f64
        } else {
            0.0
        }
    };
    let recall_at_1 = recall_at(1);
    let recall_at_3 = recall_at(3);
    let recall_at_5 = recall_at(5);
    let f1 = if precision_at_k + recall > 0.0 {
        2.0 * (precision_at_k * recall) / (precision_at_k + recall)
    } else {
        0.0
    };

    ScoreMetrics {
        precision_at_k,
        recall,
        recall_at_1,
        recall_at_3,
        recall_at_5,
        mrr,
        f1,
        hits_at_k,
        matched_files,
        unmatched_expected_files,
    }
}
