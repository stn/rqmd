//! Reciprocal Rank Fusion (RRF) for blending lexical and semantic rankings.
//!
//! Port of `tobi/qmd`'s `src/store.ts` lines 1905–1942 (types) and
//! 3583–3692 (`reciprocalRankFusion`, `buildRrfTrace`). Pure logic — no
//! database, no LLM.
//!
//! The fusion formula is `score = Σ weight / (k + rank)` (rank 1-indexed),
//! with a small top-rank bonus: +0.05 if the document appears at rank 1 in
//! any list, +0.02 if at rank 2 or 3.

use std::collections::HashMap;

use super::search::{RankedResult, SearchSource};

/// Metadata about a ranked list — used by [`build_rrf_trace`].
#[derive(Debug, Clone, PartialEq)]
pub struct RankedListMeta {
    pub source: SearchSource,
    pub query_type: QueryType,
    pub query: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Original,
    Lex,
    Vec,
    Hyde,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RRFContributionTrace {
    pub list_index: usize,
    pub source: SearchSource,
    pub query_type: QueryType,
    pub query: String,
    pub rank: usize,
    pub weight: f64,
    pub backend_score: f64,
    pub rrf_contribution: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RRFScoreTrace {
    pub contributions: Vec<RRFContributionTrace>,
    pub base_score: f64,
    pub top_rank: usize,
    pub top_rank_bonus: f64,
    pub total_score: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct HybridQueryExplain {
    pub fts_scores: Vec<f64>,
    pub vector_scores: Vec<f64>,
    pub rrf: RRFExplain,
    pub rerank_score: f64,
    pub blended_score: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RRFExplain {
    pub rank: usize,
    pub position_score: f64,
    pub weight: f64,
    pub base_score: f64,
    pub top_rank_bonus: f64,
    pub total_score: f64,
    pub contributions: Vec<RRFContributionTrace>,
}

const DEFAULT_K: usize = 60;

/// Reciprocal Rank Fusion. Returns the fused list sorted descending by
/// score. Mirrors `reciprocalRankFusion` (`store.ts:3583–3626`).
pub fn reciprocal_rank_fusion(
    result_lists: &[Vec<RankedResult>],
    weights: &[f64],
    k: Option<usize>,
) -> Vec<RankedResult> {
    let k = k.unwrap_or(DEFAULT_K) as f64;

    struct Entry {
        result: RankedResult,
        rrf: f64,
        top_rank0: usize,
    }
    let mut scores: HashMap<String, Entry> = HashMap::new();

    for (list_idx, list) in result_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank0, result) in list.iter().enumerate() {
            let contribution = weight / (k + (rank0 + 1) as f64);
            scores
                .entry(result.file.clone())
                .and_modify(|e| {
                    e.rrf += contribution;
                    if rank0 < e.top_rank0 {
                        e.top_rank0 = rank0;
                    }
                })
                .or_insert_with(|| Entry {
                    result: result.clone(),
                    rrf: contribution,
                    top_rank0: rank0,
                });
        }
    }

    for entry in scores.values_mut() {
        if entry.top_rank0 == 0 {
            entry.rrf += 0.05;
        } else if entry.top_rank0 <= 2 {
            entry.rrf += 0.02;
        }
    }

    let mut out: Vec<_> = scores.into_values().collect();
    out.sort_by(|a, b| {
        b.rrf
            .partial_cmp(&a.rrf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.into_iter()
        .map(|e| RankedResult {
            score: e.rrf,
            ..e.result
        })
        .collect()
}

/// Per-document RRF contribution trace. Mirrors `buildRrfTrace`
/// (`store.ts:3631–3692`).
pub fn build_rrf_trace(
    result_lists: &[Vec<RankedResult>],
    weights: &[f64],
    list_meta: &[RankedListMeta],
    k: Option<usize>,
) -> HashMap<String, RRFScoreTrace> {
    let k = k.unwrap_or(DEFAULT_K) as f64;
    let mut traces: HashMap<String, RRFScoreTrace> = HashMap::new();

    for (list_idx, list) in result_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        let meta = list_meta.get(list_idx).cloned().unwrap_or(RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryType::Original,
            query: String::new(),
        });
        for (rank0, result) in list.iter().enumerate() {
            let rank = rank0 + 1;
            let contribution = weight / (k + rank as f64);
            let detail = RRFContributionTrace {
                list_index: list_idx,
                source: meta.source,
                query_type: meta.query_type,
                query: meta.query.clone(),
                rank,
                weight,
                backend_score: result.score,
                rrf_contribution: contribution,
            };
            traces
                .entry(result.file.clone())
                .and_modify(|t| {
                    t.base_score += contribution;
                    if rank < t.top_rank {
                        t.top_rank = rank;
                    }
                    t.contributions.push(detail.clone());
                })
                .or_insert_with(|| RRFScoreTrace {
                    contributions: vec![detail],
                    base_score: contribution,
                    top_rank: rank,
                    top_rank_bonus: 0.0,
                    total_score: 0.0,
                });
        }
    }

    for trace in traces.values_mut() {
        let bonus = if trace.top_rank == 1 {
            0.05
        } else if trace.top_rank <= 3 {
            0.02
        } else {
            0.0
        };
        trace.top_rank_bonus = bonus;
        trace.total_score = trace.base_score + bonus;
    }

    traces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(file: &str, score: f64) -> RankedResult {
        RankedResult {
            file: file.into(),
            display_path: file.into(),
            title: file.into(),
            body: String::new(),
            score,
        }
    }

    #[test]
    fn rrf_blends_two_lists() {
        // Two lists of 3 docs each:
        //   list 1: [a, b, c]
        //   list 2: [b, c, a]
        // weights [1, 1], k=60.
        //
        // a: 1/61 (rank 1, list1) + 1/63 (rank 3, list2) + 0.05 bonus
        // b: 1/62 (rank 2, list1) + 1/61 (rank 1, list2) + 0.05 bonus
        // c: 1/63 (rank 3, list1) + 1/62 (rank 2, list2) + 0.02 bonus
        let l1 = vec![ranked("a", 0.0), ranked("b", 0.0), ranked("c", 0.0)];
        let l2 = vec![ranked("b", 0.0), ranked("c", 0.0), ranked("a", 0.0)];
        let fused = reciprocal_rank_fusion(&[l1, l2], &[1.0, 1.0], Some(60));

        // b should top because it has rank-1 in list 2 AND rank-2 in list 1.
        assert_eq!(fused[0].file, "b");

        let by_file: HashMap<_, _> = fused.iter().map(|r| (r.file.clone(), r.score)).collect();
        let a = by_file["a"];
        let b = by_file["b"];
        let c = by_file["c"];

        let exp_a = 1.0 / 61.0 + 1.0 / 63.0 + 0.05;
        let exp_b = 1.0 / 62.0 + 1.0 / 61.0 + 0.05;
        let exp_c = 1.0 / 63.0 + 1.0 / 62.0 + 0.02;

        assert!((a - exp_a).abs() < 1e-9, "a = {} vs {}", a, exp_a);
        assert!((b - exp_b).abs() < 1e-9, "b = {} vs {}", b, exp_b);
        assert!((c - exp_c).abs() < 1e-9, "c = {} vs {}", c, exp_c);
    }

    #[test]
    fn rrf_trace_captures_contributions() {
        let l1 = vec![ranked("a", 0.5)];
        let l2 = vec![ranked("a", 0.7)];
        let meta = vec![
            RankedListMeta {
                source: SearchSource::Fts,
                query_type: QueryType::Original,
                query: "q1".into(),
            },
            RankedListMeta {
                source: SearchSource::Vec,
                query_type: QueryType::Vec,
                query: "q2".into(),
            },
        ];
        let traces = build_rrf_trace(&[l1, l2], &[1.0, 1.0], &meta, Some(60));
        let t = &traces["a"];
        assert_eq!(t.contributions.len(), 2);
        assert_eq!(t.top_rank, 1);
        assert!((t.top_rank_bonus - 0.05).abs() < 1e-9);
        let expected = 1.0 / 61.0 + 1.0 / 61.0 + 0.05;
        assert!((t.total_score - expected).abs() < 1e-9);
    }

    // --- ported from rrf-trace.test.ts `describe("buildRrfTrace")` ---

    #[test]
    fn rrf_trace_matches_fusion_totals_and_records_contributions() {
        let list1 = vec![
            ranked("qmd://docs/a.md", 0.92),
            ranked("qmd://docs/b.md", 0.81),
        ];
        let list2 = vec![
            ranked("qmd://docs/b.md", 0.77),
            ranked("qmd://docs/a.md", 0.65),
        ];
        let weights = [2.0, 1.0];
        let meta = vec![
            RankedListMeta {
                source: SearchSource::Fts,
                query_type: QueryType::Lex,
                query: "lex query".into(),
            },
            RankedListMeta {
                source: SearchSource::Vec,
                query_type: QueryType::Vec,
                query: "vec query".into(),
            },
        ];

        // Bind once and share `&lists` across both APIs (no clone needed).
        let lists = vec![list1, list2];
        let traces = build_rrf_trace(&lists, &weights, &meta, None);
        let fused = reciprocal_rank_fusion(&lists, &weights, None);

        // build_rrf_trace totals must equal reciprocal_rank_fusion scores.
        // TS `toBeCloseTo(_, 10)` ⇒ |diff| < 0.5e-10.
        for result in &fused {
            let trace = traces
                .get(&result.file)
                .expect("trace defined for fused result");
            assert!(
                (trace.total_score - result.score).abs() < 5e-11,
                "total_score {} vs fused score {} for {}",
                trace.total_score,
                result.score,
                result.file
            );
        }

        let a_trace = &traces["qmd://docs/a.md"];
        assert_eq!(a_trace.contributions.len(), 2);
        assert_eq!(a_trace.contributions[0].source, SearchSource::Fts);
        assert_eq!(a_trace.contributions[1].source, SearchSource::Vec);
        assert_eq!(a_trace.top_rank, 1);
        assert!((a_trace.top_rank_bonus - 0.05).abs() < 5e-11);
    }

    #[test]
    fn rrf_trace_applies_top_rank_bonus_thresholds() {
        let list = vec![
            ranked("qmd://docs/r1.md", 0.9),
            ranked("qmd://docs/r2.md", 0.8),
            ranked("qmd://docs/r3.md", 0.7),
            ranked("qmd://docs/r4.md", 0.6),
        ];
        let meta = vec![RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryType::Lex,
            query: "rank".into(),
        }];
        let traces = build_rrf_trace(&[list], &[1.0], &meta, None);

        assert!((traces["qmd://docs/r1.md"].top_rank_bonus - 0.05).abs() < 5e-11);
        assert!((traces["qmd://docs/r2.md"].top_rank_bonus - 0.02).abs() < 5e-11);
        assert!((traces["qmd://docs/r3.md"].top_rank_bonus - 0.02).abs() < 5e-11);
        assert!((traces["qmd://docs/r4.md"].top_rank_bonus - 0.0).abs() < 5e-11);
    }

    // --- ported from store.test.ts `describe("Reciprocal Rank Fusion")` ---

    #[test]
    fn rrf_combines_single_list_in_order() {
        let l1 = vec![
            ranked("doc1", 0.9),
            ranked("doc2", 0.8),
            ranked("doc3", 0.7),
        ];
        let fused = reciprocal_rank_fusion(&[l1], &[], None);
        assert_eq!(fused[0].file, "doc1");
        assert_eq!(fused[1].file, "doc2");
        assert_eq!(fused[2].file, "doc3");
    }

    #[test]
    fn rrf_merges_documents_from_multiple_lists() {
        let l1 = vec![ranked("doc1", 0.9), ranked("doc2", 0.8)];
        let l2 = vec![ranked("doc2", 0.95), ranked("doc3", 0.85)];
        let fused = reciprocal_rank_fusion(&[l1, l2], &[], None);
        let files: Vec<_> = fused.iter().map(|r| r.file.as_str()).collect();
        assert!(files.contains(&"doc1"));
        assert!(files.contains(&"doc2"));
        assert!(files.contains(&"doc3"));
    }

    #[test]
    fn rrf_respects_weights() {
        let l1 = vec![ranked("doc1", 0.9)];
        let l2 = vec![ranked("doc2", 0.9)];
        // Double weight on list 1 → doc1 ranks first.
        let fused = reciprocal_rank_fusion(&[l1, l2], &[2.0, 1.0], None);
        assert_eq!(fused[0].file, "doc1");
    }

    #[test]
    fn rrf_adds_top_rank_bonus() {
        let l1 = vec![ranked("doc1", 0.9), ranked("doc2", 0.8)];
        let l2 = vec![ranked("doc3", 0.85)];
        let fused = reciprocal_rank_fusion(&[l1, l2], &[], None);
        let by_file: HashMap<_, _> = fused.iter().map(|r| (r.file.clone(), r.score)).collect();
        // doc1 is #1 (+0.05), doc2 is #2 (+0.02) → doc1 scores higher.
        assert!(by_file["doc1"] > by_file["doc2"]);
    }

    #[test]
    fn rrf_handles_empty_lists() {
        let fused = reciprocal_rank_fusion(&[vec![], vec![]], &[], None);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_uses_k_parameter() {
        let list = vec![ranked("doc1", 0.9)];
        let fused60 = reciprocal_rank_fusion(std::slice::from_ref(&list), &[], Some(60));
        let fused30 = reciprocal_rank_fusion(&[list], &[], Some(30));
        // Lower k → higher score for top ranks.
        assert!(fused30[0].score > fused60[0].score);
    }
}
