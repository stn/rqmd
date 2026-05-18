//! `hybrid_query`: BM25 + vector + expansion + RRF + chunk-level rerank.
//!
//! Port of `tobi/qmd/src/store.ts` lines 4196–4553. The 7-step pipeline is
//! described in the plan file; this module owns step 4 onwards plus the
//! orchestration glue. Step 4–8 helpers are reused by
//! [`super::structured`] via `pub(super)`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use rmd_core::store::chunking::{chunk_document, Chunk, ChunkStrategy};
use rmd_core::store::embeddings::search_vec_with_embedding;
use rmd_core::store::rrf::{
    build_rrf_trace, reciprocal_rank_fusion, HybridQueryExplain, QueryType, RankedListMeta,
    RRFExplain, RRFScoreTrace,
};
use rmd_core::store::search::{search_fts, RankedResult, SearchResult, SearchSource};
use rmd_core::store::snippet::extract_intent_terms;
use rmd_core::store::Store;
use rmd_core::store::{
    INTENT_WEIGHT_CHUNK, RERANK_CANDIDATE_LIMIT, STRONG_SIGNAL_MIN_GAP, STRONG_SIGNAL_MIN_SCORE,
};

use crate::config::{resolve_embed_model, resolve_rerank_model};
use crate::format::format_query_for_embedding;
use crate::traits::Llm;
use crate::types::EmbedOptions;

use super::expand::{expand_query, ExpandedQuery, ExpandedQueryType};
use super::rerank::{rerank, RerankCandidate};
use super::Result;

/// Callback hooks for hybrid / vector / structured pipelines. Each is
/// optional; `None` is a no-op.
pub type StrongSignalHook = Arc<dyn Fn(f64) + Send + Sync>;
pub type StartHook = Arc<dyn Fn() + Send + Sync>;
pub type ExpandHook = Arc<dyn Fn(&str, &[ExpandedQuery], u128) + Send + Sync>;
pub type CountHook = Arc<dyn Fn(usize) + Send + Sync>;
pub type ElapsedHook = Arc<dyn Fn(u128) + Send + Sync>;

#[derive(Default, Clone)]
pub struct SearchHooks {
    pub on_strong_signal: Option<StrongSignalHook>,
    pub on_expand_start: Option<StartHook>,
    pub on_expand: Option<ExpandHook>,
    pub on_embed_start: Option<CountHook>,
    pub on_embed_done: Option<ElapsedHook>,
    pub on_rerank_start: Option<CountHook>,
    pub on_rerank_done: Option<ElapsedHook>,
}

impl std::fmt::Debug for SearchHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchHooks")
            .field("on_strong_signal", &self.on_strong_signal.is_some())
            .field("on_expand_start", &self.on_expand_start.is_some())
            .field("on_expand", &self.on_expand.is_some())
            .field("on_embed_start", &self.on_embed_start.is_some())
            .field("on_embed_done", &self.on_embed_done.is_some())
            .field("on_rerank_start", &self.on_rerank_start.is_some())
            .field("on_rerank_done", &self.on_rerank_done.is_some())
            .finish()
    }
}

#[derive(Debug, Default, Clone)]
pub struct HybridQueryOptions {
    pub collection: Option<String>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
    pub candidate_limit: Option<usize>,
    pub explain: bool,
    pub intent: Option<String>,
    pub skip_rerank: bool,
    pub chunk_strategy: Option<ChunkStrategy>,
    pub hooks: SearchHooks,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HybridQueryResult {
    pub file: String,
    pub display_path: String,
    pub title: String,
    pub body: String,
    pub best_chunk: String,
    pub best_chunk_pos: usize,
    pub score: f64,
    pub context: Option<String>,
    pub docid: String,
    pub explain: Option<HybridQueryExplain>,
}

/// Hybrid search: BM25 + vector + LLM expansion + RRF + chunked rerank.
/// See module docs for the pipeline and plan file for edge cases.
pub async fn hybrid_query(
    store: &Store,
    llm: Arc<dyn Llm>,
    query: &str,
    options: HybridQueryOptions,
) -> Result<Vec<HybridQueryResult>> {
    let limit = options.limit.unwrap_or(10);
    let min_score = options.min_score.unwrap_or(0.0);
    let candidate_limit = options.candidate_limit.unwrap_or(RERANK_CANDIDATE_LIMIT);
    let collection = options.collection.as_deref();
    let explain = options.explain;
    let intent = options.intent.as_deref();
    let skip_rerank = options.skip_rerank;
    let hooks = &options.hooks;
    let chunk_strategy = options.chunk_strategy.unwrap_or(ChunkStrategy::Auto);

    let embed_model = resolve_embed_model(None);
    let rerank_model = resolve_rerank_model(None);

    let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();
    let mut ranked_meta: Vec<RankedListMeta> = Vec::new();
    let mut docid_map: HashMap<String, String> = HashMap::new();

    let has_vectors = has_vector_index(store)?;

    // Step 1: BM25 probe.
    let initial_fts = store.with_connection(|c| search_fts(c, query, Some(20), collection))?;
    let top_score = initial_fts.first().map(|r| r.score).unwrap_or(0.0);
    let second_score = initial_fts.get(1).map(|r| r.score).unwrap_or(0.0);
    let has_strong_signal = intent.is_none()
        && !initial_fts.is_empty()
        && top_score >= STRONG_SIGNAL_MIN_SCORE
        && (top_score - second_score) >= STRONG_SIGNAL_MIN_GAP;
    if has_strong_signal && let Some(h) = &hooks.on_strong_signal {
        h(top_score);
    }

    // Step 2: expand (or skip).
    if let Some(h) = &hooks.on_expand_start {
        h();
    }
    let expand_start = Instant::now();
    let expanded: Vec<ExpandedQuery> = if has_strong_signal {
        Vec::new()
    } else {
        expand_query(store, llm.clone(), query, &embed_model, intent).await?
    };
    let expand_elapsed = expand_start.elapsed().as_millis();
    if let Some(h) = &hooks.on_expand {
        h(query, &expanded, expand_elapsed);
    }

    // Seed with initial FTS.
    if !initial_fts.is_empty() {
        for r in &initial_fts {
            docid_map.insert(r.doc.filepath.clone(), r.doc.docid.clone());
        }
        ranked_lists.push(initial_fts.iter().map(to_ranked).collect());
        ranked_meta.push(RankedListMeta {
            source: SearchSource::Fts,
            query_type: QueryType::Original,
            query: query.to_string(),
        });
    }

    // Step 3a: FTS for lex expansions.
    for q in &expanded {
        if matches!(q.type_, ExpandedQueryType::Lex) {
            let results =
                store.with_connection(|c| search_fts(c, &q.query, Some(20), collection))?;
            if !results.is_empty() {
                for r in &results {
                    docid_map.insert(r.doc.filepath.clone(), r.doc.docid.clone());
                }
                ranked_lists.push(results.iter().map(to_ranked).collect());
                ranked_meta.push(RankedListMeta {
                    source: SearchSource::Fts,
                    query_type: QueryType::Lex,
                    query: q.query.clone(),
                });
            }
        }
    }

    // Step 3b: batch-embed original + vec/hyde, then sequential vec lookup.
    if has_vectors {
        let mut vec_queries: Vec<(String, QueryType)> =
            vec![(query.to_string(), QueryType::Original)];
        for q in &expanded {
            match q.type_ {
                ExpandedQueryType::Vec => {
                    vec_queries.push((q.query.clone(), QueryType::Vec));
                }
                ExpandedQueryType::Hyde => {
                    vec_queries.push((q.query.clone(), QueryType::Hyde));
                }
                ExpandedQueryType::Lex => {}
            }
        }

        let texts: Vec<String> = vec_queries
            .iter()
            .map(|(t, _)| format_query_for_embedding(t, &embed_model))
            .collect();
        if let Some(h) = &hooks.on_embed_start {
            h(texts.len());
        }
        let embed_start = Instant::now();
        let embeddings = llm
            .embed_batch(
                &texts,
                EmbedOptions {
                    model: Some(embed_model.clone()),
                    is_query: true,
                    title: None,
                },
            )
            .await?;
        if let Some(h) = &hooks.on_embed_done {
            h(embed_start.elapsed().as_millis());
        }

        for (i, (text, qtype)) in vec_queries.into_iter().enumerate() {
            let Some(Some(emb)) = embeddings.get(i) else {
                continue;
            };
            let vec_results = store.with_connection(|c| {
                search_vec_with_embedding(c, &emb.embedding, 20, collection)
            })?;
            if !vec_results.is_empty() {
                for r in &vec_results {
                    docid_map.insert(r.doc.filepath.clone(), r.doc.docid.clone());
                }
                ranked_lists.push(vec_results.iter().map(to_ranked).collect());
                ranked_meta.push(RankedListMeta {
                    source: SearchSource::Vec,
                    query_type: qtype,
                    query: text,
                });
            }
        }
    }

    // Step 4: RRF fusion.
    let weights = get_hybrid_rrf_weights(&ranked_meta);
    let fused = reciprocal_rank_fusion(&ranked_lists, &weights, None);
    let rrf_trace = if explain {
        Some(build_rrf_trace(&ranked_lists, &weights, &ranked_meta, None))
    } else {
        None
    };
    let candidates: Vec<RankedResult> = fused.into_iter().take(candidate_limit).collect();
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Step 5: chunk top candidates, pick best chunk by keyword overlap.
    let doc_chunk_map = build_doc_chunk_map(&candidates, query, intent, chunk_strategy);

    // Steps 6 + 7: skip-rerank or full rerank then blend.
    let candidate_meta: HashMap<&str, &RankedResult> =
        candidates.iter().map(|c| (c.file.as_str(), c)).collect();
    let rrf_rank_map: HashMap<&str, usize> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.file.as_str(), i + 1))
        .collect();

    let blended = if skip_rerank {
        build_skip_rerank_output(
            &candidates,
            &doc_chunk_map,
            &docid_map,
            rrf_trace.as_ref(),
            store,
            explain,
        )
    } else {
        let chunks_to_rerank = collect_rerank_candidates(&candidates, &doc_chunk_map);
        if let Some(h) = &hooks.on_rerank_start {
            h(chunks_to_rerank.len());
        }
        let r_start = Instant::now();
        let reranked = rerank(store, llm, query, &chunks_to_rerank, &rerank_model, intent).await?;
        if let Some(h) = &hooks.on_rerank_done {
            h(r_start.elapsed().as_millis());
        }
        build_blended_output(
            &reranked,
            &candidate_meta,
            &rrf_rank_map,
            &doc_chunk_map,
            &docid_map,
            rrf_trace.as_ref(),
            store,
            candidate_limit,
            explain,
        )
    };

    let mut sorted = blended;
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    let mut seen: HashSet<String> = HashSet::new();
    Ok(sorted
        .into_iter()
        .filter(|r| seen.insert(r.file.clone()))
        .filter(|r| r.score >= min_score)
        .take(limit)
        .collect())
}

// ---------------------------------------------------------------------------
// Helpers (also reused by `structured_search`).
// ---------------------------------------------------------------------------

pub(super) fn has_vector_index(store: &Store) -> Result<bool> {
    use rmd_core::db::rusqlite::OptionalExtension;
    Ok(store.with_connection(|c| {
        let row: Option<i64> = c
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
                [],
                |r| r.get(0),
            )
            .optional()
            .unwrap_or(None);
        row.is_some()
    }))
}

pub(super) fn to_ranked(r: &SearchResult) -> RankedResult {
    RankedResult {
        file: r.doc.filepath.clone(),
        display_path: r.doc.display_path.clone(),
        title: r.doc.title.clone(),
        body: r.doc.body.clone().unwrap_or_default(),
        score: r.score,
    }
}

pub fn get_hybrid_rrf_weights(ranked_list_meta: &[RankedListMeta]) -> Vec<f64> {
    ranked_list_meta
        .iter()
        .map(|m| {
            if matches!(m.query_type, QueryType::Original) {
                2.0
            } else {
                1.0
            }
        })
        .collect()
}

pub(super) struct ChunkInfo {
    pub chunks: Vec<Chunk>,
    pub best_idx: usize,
}

pub(super) fn build_doc_chunk_map(
    candidates: &[RankedResult],
    primary_query: &str,
    intent: Option<&str>,
    chunk_strategy: ChunkStrategy,
) -> HashMap<String, ChunkInfo> {
    let query_terms: Vec<String> = primary_query
        .to_lowercase()
        .split_whitespace()
        .filter(|t| t.len() > 2)
        .map(String::from)
        .collect();
    let intent_terms = intent.map(extract_intent_terms).unwrap_or_default();

    let mut map: HashMap<String, ChunkInfo> = HashMap::new();
    for cand in candidates {
        let chunks = chunk_document(&cand.body, chunk_strategy, None, None, None);
        if chunks.is_empty() {
            continue;
        }
        let mut best_idx = 0usize;
        let mut best_score = -1.0_f64;
        for (i, c) in chunks.iter().enumerate() {
            let lower = c.text.to_lowercase();
            let mut score = query_terms.iter().filter(|t| lower.contains(t.as_str())).count() as f64;
            for t in &intent_terms {
                if lower.contains(t) {
                    score += INTENT_WEIGHT_CHUNK;
                }
            }
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        map.insert(cand.file.clone(), ChunkInfo { chunks, best_idx });
    }
    map
}

pub(super) fn collect_rerank_candidates(
    candidates: &[RankedResult],
    doc_chunk_map: &HashMap<String, ChunkInfo>,
) -> Vec<RerankCandidate> {
    candidates
        .iter()
        .filter_map(|c| {
            doc_chunk_map.get(&c.file).map(|info| RerankCandidate {
                file: c.file.clone(),
                text: info.chunks[info.best_idx].text.clone(),
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_blended_output(
    reranked: &[super::rerank::RerankScore],
    candidate_meta: &HashMap<&str, &RankedResult>,
    rrf_rank_map: &HashMap<&str, usize>,
    doc_chunk_map: &HashMap<String, ChunkInfo>,
    docid_map: &HashMap<String, String>,
    rrf_trace: Option<&HashMap<String, RRFScoreTrace>>,
    store: &Store,
    candidate_limit: usize,
    explain: bool,
) -> Vec<HybridQueryResult> {
    reranked
        .iter()
        .map(|r| {
            let rank = rrf_rank_map
                .get(r.file.as_str())
                .copied()
                .unwrap_or(candidate_limit);
            let rrf_weight = position_weight(rank);
            let rrf_score = 1.0 / rank as f64;
            let blended_score = rrf_weight * rrf_score + (1.0 - rrf_weight) * r.score as f64;
            let candidate = candidate_meta.get(r.file.as_str());
            let (best_chunk, best_pos) =
                best_chunk_for(r.file.as_str(), doc_chunk_map, candidate.map(|c| c.body.as_str()));
            HybridQueryResult {
                file: r.file.clone(),
                display_path: candidate
                    .map(|c| c.display_path.clone())
                    .unwrap_or_default(),
                title: candidate.map(|c| c.title.clone()).unwrap_or_default(),
                body: candidate.map(|c| c.body.clone()).unwrap_or_default(),
                best_chunk,
                best_chunk_pos: best_pos,
                score: blended_score,
                context: get_context(store, &r.file),
                docid: docid_map.get(&r.file).cloned().unwrap_or_default(),
                explain: explain_block(
                    explain,
                    rrf_trace,
                    &r.file,
                    rank,
                    rrf_weight,
                    rrf_score,
                    r.score,
                    blended_score,
                ),
            }
        })
        .collect()
}

pub(super) fn build_skip_rerank_output(
    candidates: &[RankedResult],
    doc_chunk_map: &HashMap<String, ChunkInfo>,
    docid_map: &HashMap<String, String>,
    rrf_trace: Option<&HashMap<String, RRFScoreTrace>>,
    store: &Store,
    explain: bool,
) -> Vec<HybridQueryResult> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, cand)| {
            let (best_chunk, best_pos) =
                best_chunk_for(cand.file.as_str(), doc_chunk_map, Some(cand.body.as_str()));
            let rank = i + 1;
            let rrf_score = 1.0 / rank as f64;
            HybridQueryResult {
                file: cand.file.clone(),
                display_path: cand.display_path.clone(),
                title: cand.title.clone(),
                body: cand.body.clone(),
                best_chunk,
                best_chunk_pos: best_pos,
                score: rrf_score,
                context: get_context(store, &cand.file),
                docid: docid_map.get(&cand.file).cloned().unwrap_or_default(),
                explain: explain_block(
                    explain, rrf_trace, &cand.file, rank, 1.0, rrf_score, 0.0, rrf_score,
                ),
            }
        })
        .collect()
}

pub(super) fn position_weight(rank: usize) -> f64 {
    if rank <= 3 {
        0.75
    } else if rank <= 10 {
        0.60
    } else {
        0.40
    }
}

fn best_chunk_for(
    file: &str,
    doc_chunk_map: &HashMap<String, ChunkInfo>,
    body_fallback: Option<&str>,
) -> (String, usize) {
    if let Some(info) = doc_chunk_map.get(file) {
        let chunk = &info.chunks[info.best_idx];
        return (chunk.text.clone(), chunk.pos);
    }
    (body_fallback.unwrap_or("").to_string(), 0)
}

fn get_context(store: &Store, filepath: &str) -> Option<String> {
    store
        .with_connection(|c| rmd_core::store::context::get_context_for_file(c, filepath))
        .ok()
        .flatten()
}

#[allow(clippy::too_many_arguments)]
fn explain_block(
    explain: bool,
    rrf_trace: Option<&HashMap<String, RRFScoreTrace>>,
    file: &str,
    rank: usize,
    rrf_weight: f64,
    rrf_score: f64,
    rerank_score: f32,
    blended_score: f64,
) -> Option<HybridQueryExplain> {
    if !explain {
        return None;
    }
    let trace = rrf_trace.and_then(|t| t.get(file));
    let fts_scores = trace
        .map(|t| {
            t.contributions
                .iter()
                .filter(|c| matches!(c.source, SearchSource::Fts))
                .map(|c| c.backend_score)
                .collect()
        })
        .unwrap_or_default();
    let vector_scores = trace
        .map(|t| {
            t.contributions
                .iter()
                .filter(|c| matches!(c.source, SearchSource::Vec))
                .map(|c| c.backend_score)
                .collect()
        })
        .unwrap_or_default();
    Some(HybridQueryExplain {
        fts_scores,
        vector_scores,
        rrf: RRFExplain {
            rank,
            position_score: rrf_score,
            weight: rrf_weight,
            base_score: trace.map(|t| t.base_score).unwrap_or(0.0),
            top_rank_bonus: trace.map(|t| t.top_rank_bonus).unwrap_or(0.0),
            total_score: trace.map(|t| t.total_score).unwrap_or(0.0),
            contributions: trace.map(|t| t.contributions.clone()).unwrap_or_default(),
        },
        rerank_score: rerank_score as f64,
        blended_score,
    })
}
