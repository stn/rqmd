//! `structured_search`: pre-expanded query path with no LLM expansion.
//!
//! Port of `tobi/qmd/src/store.ts` lines 4671–4820. Same pipeline as
//! `hybrid_query` minus the expansion step; caller is expected to have
//! produced the `ExpandedQuery[]` (e.g. a larger LLM generated them).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use crate::store::chunking::ChunkStrategy;
use crate::store::embeddings::search_vec_with_embedding;
use crate::store::rrf::{
    build_rrf_trace, reciprocal_rank_fusion, QueryType, RankedListMeta,
};
use crate::store::search::{
    search_fts, validate_lex_query, validate_semantic_query, RankedResult, SearchSource,
};
use crate::store::Store;
use crate::store::RERANK_CANDIDATE_LIMIT;

use crate::llm::config::{resolve_embed_model, resolve_rerank_model};
use crate::llm::format::format_query_for_embedding;
use crate::llm::traits::Llm;
use crate::llm::types::EmbedOptions;

use super::expand::{ExpandedQuery, ExpandedQueryType};
use super::hybrid::{
    build_blended_output, build_doc_chunk_map, build_skip_rerank_output, collect_rerank_candidates,
    has_vector_index, to_ranked, HybridQueryResult, SearchHooks,
};
use super::rerank::rerank;
use super::{Error, Result};

#[derive(Debug, Default, Clone)]
pub struct StructuredSearchOptions {
    pub collections: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
    pub candidate_limit: Option<usize>,
    pub explain: bool,
    pub intent: Option<String>,
    pub skip_rerank: bool,
    pub chunk_strategy: Option<ChunkStrategy>,
    pub hooks: SearchHooks,
}

pub async fn structured_search(
    store: &Store,
    llm: Arc<dyn Llm>,
    searches: &[ExpandedQuery],
    options: StructuredSearchOptions,
) -> Result<Vec<HybridQueryResult>> {
    let limit = options.limit.unwrap_or(10);
    let min_score = options.min_score.unwrap_or(0.0);
    let candidate_limit = options.candidate_limit.unwrap_or(RERANK_CANDIDATE_LIMIT);
    let explain = options.explain;
    let intent = options.intent.as_deref();
    let skip_rerank = options.skip_rerank;
    let hooks = &options.hooks;
    let chunk_strategy = options.chunk_strategy.unwrap_or(ChunkStrategy::Auto);
    let collections = options.collections.clone();

    if searches.is_empty() {
        return Ok(Vec::new());
    }

    // Validate searches.
    for s in searches {
        let location = s
            .line
            .map(|l| format!("Line {l}"))
            .unwrap_or_else(|| "Structured search".to_string());
        if s.query.contains(['\r', '\n']) {
            return Err(Error::InvalidSearch(format!(
                "{location} ({}): queries must be single-line. Remove newline characters.",
                type_name(s.type_)
            )));
        }
        match s.type_ {
            ExpandedQueryType::Lex => {
                validate_lex_query(&s.query).map_err(|e| {
                    Error::InvalidSearch(format!("{location} (lex): {e}"))
                })?;
            }
            ExpandedQueryType::Vec | ExpandedQueryType::Hyde => {
                validate_semantic_query(&s.query).map_err(|e| {
                    Error::InvalidSearch(format!(
                        "{location} ({}): {e}",
                        type_name(s.type_)
                    ))
                })?;
            }
        }
    }

    let embed_model = resolve_embed_model(None);
    let rerank_model = resolve_rerank_model(None);

    let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();
    let mut ranked_meta: Vec<RankedListMeta> = Vec::new();
    let mut docid_map: HashMap<String, String> = HashMap::new();
    let has_vectors = has_vector_index(store)?;

    // `None` collection → search across all collections (single iter with None);
    // `Some([c1, c2])` → fan out per collection.
    let collection_iter: Vec<Option<String>> = match collections {
        Some(list) if !list.is_empty() => list.into_iter().map(Some).collect(),
        _ => vec![None],
    };

    // Step 1: FTS for lex searches.
    for s in searches {
        if !matches!(s.type_, ExpandedQueryType::Lex) {
            continue;
        }
        for coll in &collection_iter {
            let results = store.with_connection(|c| {
                search_fts(c, &s.query, Some(20), coll.as_deref())
            })?;
            if !results.is_empty() {
                for r in &results {
                    docid_map.insert(r.doc.filepath.clone(), r.doc.docid.clone());
                }
                ranked_lists.push(results.iter().map(to_ranked).collect());
                ranked_meta.push(RankedListMeta {
                    source: SearchSource::Fts,
                    query_type: QueryType::Lex,
                    query: s.query.clone(),
                });
            }
        }
    }

    // Step 2: batch embed + sequential vec lookup for vec/hyde searches.
    if has_vectors {
        let vec_searches: Vec<&ExpandedQuery> = searches
            .iter()
            .filter(|s| matches!(s.type_, ExpandedQueryType::Vec | ExpandedQueryType::Hyde))
            .collect();
        if !vec_searches.is_empty() {
            let texts: Vec<String> = vec_searches
                .iter()
                .map(|s| format_query_for_embedding(&s.query, &embed_model))
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

            for (i, s) in vec_searches.iter().enumerate() {
                let Some(Some(emb)) = embeddings.get(i) else {
                    continue;
                };
                for coll in &collection_iter {
                    let results = store.with_connection(|c| {
                        search_vec_with_embedding(c, &emb.embedding, 20, coll.as_deref())
                    })?;
                    if !results.is_empty() {
                        for r in &results {
                            docid_map.insert(r.doc.filepath.clone(), r.doc.docid.clone());
                        }
                        ranked_lists.push(results.iter().map(to_ranked).collect());
                        ranked_meta.push(RankedListMeta {
                            source: SearchSource::Vec,
                            query_type: match s.type_ {
                                ExpandedQueryType::Vec => QueryType::Vec,
                                _ => QueryType::Hyde,
                            },
                            query: s.query.clone(),
                        });
                    }
                }
            }
        }
    }

    if ranked_lists.is_empty() {
        return Ok(Vec::new());
    }

    // Step 3: RRF — first list gets 2x weight regardless of source.
    let weights: Vec<f64> = (0..ranked_lists.len())
        .map(|i| if i == 0 { 2.0 } else { 1.0 })
        .collect();
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

    if let Some(h) = &hooks.on_expand {
        // Signal "no expansion" for parity with TS (`hooks.onExpand("", [], 0)`).
        h("", &[], 0);
    }

    // Step 4: chunk + best-chunk selection. Use first lex (or first vec, or
    // first overall) as the "primary query" for keyword scoring.
    let primary_query = searches
        .iter()
        .find(|s| matches!(s.type_, ExpandedQueryType::Lex))
        .or_else(|| {
            searches
                .iter()
                .find(|s| matches!(s.type_, ExpandedQueryType::Vec))
        })
        .or_else(|| searches.first())
        .map(|s| s.query.as_str())
        .unwrap_or("");
    let doc_chunk_map = build_doc_chunk_map(&candidates, primary_query, intent, chunk_strategy);

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
        let reranked = rerank(
            store,
            llm,
            primary_query,
            &chunks_to_rerank,
            &rerank_model,
            intent,
        )
        .await?;
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

fn type_name(t: ExpandedQueryType) -> &'static str {
    match t {
        ExpandedQueryType::Lex => "lex",
        ExpandedQueryType::Vec => "vec",
        ExpandedQueryType::Hyde => "hyde",
    }
}
