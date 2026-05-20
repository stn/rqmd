//! Parity port of `tobi/qmd/test/bench-score.test.ts`.
//!
//! Covers `normalize_path` (4), `paths_match` (6), and `score_results` (8).

use rqmd_core::bench::score::{normalize_path, paths_match, score_results};

/// `&["a", "b"]` → `Vec<String>`, to match the `&[String]` signatures.
fn v(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// `toBeCloseTo` equivalent (vitest defaults to ~2 digits, but the TS values
/// are exact rationals like 1/3; 1e-9 is comfortably tighter).
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {a} ≈ {b}");
}

// ---------------------------------------------------------------------------
// normalize_path
// ---------------------------------------------------------------------------

#[test]
fn normalize_path_lowercases() {
    assert_eq!(
        normalize_path("Resources/Concepts/Context Engineering.md"),
        "resources/concepts/context engineering.md"
    );
}

#[test]
fn normalize_path_strips_qmd_prefix() {
    assert_eq!(
        normalize_path("qmd://collection/docs/readme.md"),
        "docs/readme.md"
    );
}

#[test]
fn normalize_path_strips_leading_trailing_slashes() {
    assert_eq!(normalize_path("/docs/readme.md/"), "docs/readme.md");
}

#[test]
fn normalize_path_handles_plain_filename() {
    assert_eq!(normalize_path("readme.md"), "readme.md");
}

// ---------------------------------------------------------------------------
// paths_match
// ---------------------------------------------------------------------------

#[test]
fn paths_match_exact() {
    assert!(paths_match("docs/readme.md", "docs/readme.md"));
}

#[test]
fn paths_match_case_insensitive() {
    assert!(paths_match("Docs/README.md", "docs/readme.md"));
}

#[test]
fn paths_match_suffix_result_longer() {
    assert!(paths_match("/full/path/docs/readme.md", "docs/readme.md"));
}

#[test]
fn paths_match_suffix_expected_longer() {
    assert!(paths_match("readme.md", "docs/readme.md"));
}

#[test]
fn paths_match_qmd_prefix() {
    assert!(paths_match("qmd://col/docs/readme.md", "docs/readme.md"));
}

#[test]
fn paths_match_different_files_dont_match() {
    assert!(!paths_match("docs/readme.md", "docs/other.md"));
}

// ---------------------------------------------------------------------------
// score_results
// ---------------------------------------------------------------------------

#[test]
fn score_perfect_all_expected_in_top_k() {
    let r = score_results(&v(&["a.md", "b.md", "c.md"]), &v(&["a.md", "b.md"]), 2);
    assert_eq!(r.precision_at_k, 1.0);
    assert_eq!(r.recall, 1.0);
    assert_eq!(r.mrr, 1.0);
    assert_eq!(r.f1, 1.0);
    assert_eq!(r.hits_at_k, 2);
}

#[test]
fn score_zero_none_found() {
    let r = score_results(&v(&["x.md", "y.md", "z.md"]), &v(&["a.md", "b.md"]), 2);
    assert_eq!(r.precision_at_k, 0.0);
    assert_eq!(r.recall, 0.0);
    assert_eq!(r.mrr, 0.0);
    assert_eq!(r.f1, 0.0);
    assert_eq!(r.hits_at_k, 0);
}

#[test]
fn score_partial_found_outside_top_k() {
    let r = score_results(&v(&["x.md", "y.md", "a.md"]), &v(&["a.md"]), 1);
    assert_eq!(r.precision_at_k, 0.0); // not in top-1
    assert_eq!(r.recall, 1.0); // found somewhere
    close(r.mrr, 1.0 / 3.0); // rank 3
    assert_eq!(r.hits_at_k, 0);
}

#[test]
fn score_mrr_first_relevant_at_rank_2() {
    let r = score_results(&v(&["x.md", "a.md", "b.md"]), &v(&["a.md", "b.md"]), 3);
    close(r.mrr, 0.5); // 1/2
}

#[test]
fn score_reports_recall_at_k_and_matched_documents() {
    let r = score_results(
        &v(&[
            "x.md",
            "qmd://concepts/a.md",
            "docs/b.md",
            "docs/c.md",
            "docs/d.md",
        ]),
        &v(&["concepts/a.md", "b.md", "missing.md"]),
        3,
    );
    assert_eq!(r.recall_at_1, 0.0);
    close(r.recall_at_3, 2.0 / 3.0);
    close(r.recall_at_5, 2.0 / 3.0);
    assert_eq!(r.matched_files, v(&["concepts/a.md", "b.md"]));
    assert_eq!(r.unmatched_expected_files, v(&["missing.md"]));
}

#[test]
fn score_empty_results() {
    let r = score_results(&v(&[]), &v(&["a.md"]), 1);
    assert_eq!(r.precision_at_k, 0.0);
    assert_eq!(r.recall, 0.0);
    assert_eq!(r.mrr, 0.0);
}

#[test]
fn score_empty_expected() {
    let r = score_results(&v(&["a.md"]), &v(&[]), 1);
    assert_eq!(r.precision_at_k, 0.0);
    assert_eq!(r.recall, 0.0);
}
