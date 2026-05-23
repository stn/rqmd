//! Search-quality regression suite — faithful port of
//! `tobi/qmd`'s `test/eval.test.ts`.
//!
//! Indexes 6 synthetic documents (`tests/eval-docs/*.md`) and runs 24
//! known-answer queries (easy / medium / hard / fusion) through three search
//! backends, asserting minimum Hit@k rates so search changes can't silently
//! regress quality:
//!
//! 1. **BM25 (FTS)** — lexical baseline. No model: always runs.
//! 2. **Vector** — semantic search via the default embed model.
//! 3. **Hybrid (RRF)** — lexical + vector fused with Reciprocal Rank Fusion.
//!
//! The Vector + Hybrid suites need the ~300 MB embeddinggemma model, so they are
//! folded into ONE `#[tokio::test]` (load & embed once; avoid parallel
//! model-load races, mirroring `rqmd-mcp`'s `mcp_llm.rs`). They run by default
//! and skip when `RQMD_SKIP_LLM_TESTS` is set — the rqmd analogue of qmd's
//! `describe.skipIf(!!process.env.CI)`. CI must export `RQMD_SKIP_LLM_TESTS=1`.

use std::sync::Arc;

use rqmd_core::llm::llama_cpp::{LlamaCpp, LlamaCppConfig};
use rqmd_core::llm::traits::Llm;
use rqmd_core::llm::types::EmbedOptions;
use rqmd_core::store::Store;
use rqmd_core::store::chunking::ChunkStrategy;
use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::embeddings::{ensure_vec_table, insert_embedding};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::rrf::reciprocal_rank_fusion;
use rqmd_core::store::search::{RankedResult, SearchResult, search_fts};
use rqmd_core::store_ops::{chunk_document_by_tokens, search_vec};
use tempfile::NamedTempFile;

// =============================================================================
// Eval queries with expected documents (eval.test.ts:42-79, verbatim)
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
    Fusion,
}

struct EvalQuery {
    query: &'static str,
    /// Substring expected to appear in the matching document's path.
    expected: &'static str,
    difficulty: Difficulty,
}

use Difficulty::{Easy, Fusion, Hard, Medium};

static EVAL_QUERIES: &[EvalQuery] = &[
    // EASY: Exact keyword matches
    EvalQuery {
        query: "API versioning",
        expected: "api-design",
        difficulty: Easy,
    },
    EvalQuery {
        query: "Series A fundraising",
        expected: "fundraising",
        difficulty: Easy,
    },
    EvalQuery {
        query: "CAP theorem",
        expected: "distributed-systems",
        difficulty: Easy,
    },
    EvalQuery {
        query: "overfitting machine learning",
        expected: "machine-learning",
        difficulty: Easy,
    },
    EvalQuery {
        query: "remote work VPN",
        expected: "remote-work",
        difficulty: Easy,
    },
    EvalQuery {
        query: "Project Phoenix retrospective",
        expected: "product-launch",
        difficulty: Easy,
    },
    // MEDIUM: Semantic/conceptual queries
    EvalQuery {
        query: "how to structure REST endpoints",
        expected: "api-design",
        difficulty: Medium,
    },
    EvalQuery {
        query: "raising money for startup",
        expected: "fundraising",
        difficulty: Medium,
    },
    EvalQuery {
        query: "consistency vs availability tradeoffs",
        expected: "distributed-systems",
        difficulty: Medium,
    },
    EvalQuery {
        query: "how to prevent models from memorizing data",
        expected: "machine-learning",
        difficulty: Medium,
    },
    EvalQuery {
        query: "working from home guidelines",
        expected: "remote-work",
        difficulty: Medium,
    },
    EvalQuery {
        query: "what went wrong with the launch",
        expected: "product-launch",
        difficulty: Medium,
    },
    // HARD: Vague, partial memory, indirect
    EvalQuery {
        query: "nouns not verbs",
        expected: "api-design",
        difficulty: Hard,
    },
    EvalQuery {
        query: "Sequoia investor pitch",
        expected: "fundraising",
        difficulty: Hard,
    },
    EvalQuery {
        query: "Raft algorithm leader election",
        expected: "distributed-systems",
        difficulty: Hard,
    },
    EvalQuery {
        query: "F1 score precision recall",
        expected: "machine-learning",
        difficulty: Hard,
    },
    EvalQuery {
        query: "quarterly team gathering travel",
        expected: "remote-work",
        difficulty: Hard,
    },
    EvalQuery {
        query: "beta program 47 bugs",
        expected: "product-launch",
        difficulty: Hard,
    },
    // FUSION: Multi-signal queries that need both lexical AND semantic matching.
    // These should have weak individual scores but strong combined RRF scores.
    EvalQuery {
        query: "how much runway before running out of money",
        expected: "fundraising",
        difficulty: Fusion,
    },
    EvalQuery {
        query: "datacenter replication sync strategy",
        expected: "distributed-systems",
        difficulty: Fusion,
    },
    EvalQuery {
        query: "splitting data for training and testing",
        expected: "machine-learning",
        difficulty: Fusion,
    },
    EvalQuery {
        query: "JSON response codes error messages",
        expected: "api-design",
        difficulty: Fusion,
    },
    EvalQuery {
        query: "video calls camera async messaging",
        expected: "remote-work",
        difficulty: Fusion,
    },
    EvalQuery {
        query: "CI/CD pipeline testing coverage",
        expected: "product-launch",
        difficulty: Fusion,
    },
];

fn by_difficulty(d: Difficulty) -> Vec<&'static EvalQuery> {
    EVAL_QUERIES.iter().filter(|q| q.difficulty == d).collect()
}

/// Mirrors qmd's `matchesExpected`: case-insensitive substring on the path.
fn matches_expected(path: &str, expected: &str) -> bool {
    path.to_lowercase().contains(expected)
}

// =============================================================================
// Fixture loading & indexing
// =============================================================================

fn read_eval_docs() -> Vec<(String, String)> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/eval-docs");
    let mut docs: Vec<(String, String)> = std::fs::read_dir(dir)
        .expect("read eval-docs dir")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                return None;
            }
            let file = path.file_name()?.to_str()?.to_string();
            let content = std::fs::read_to_string(&path).ok()?;
            Some((file, content))
        })
        .collect();
    // read_dir order is unspecified — sort for deterministic indexing.
    docs.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(docs.len(), 6, "expected 6 eval-docs, got {}", docs.len());
    docs
}

/// Title = first line minus the leading `# ` marker, falling back to the
/// filename (mirrors qmd's `content.split("\n")[0].replace(/^#\s*/, "") || file`).
fn title_of(content: &str, file: &str) -> String {
    content
        .lines()
        .next()
        .map(|l| l.trim_start_matches('#').trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| file.to_string())
}

/// Index the eval-docs into the FTS side of `store` (content + documents).
fn index_docs(store: &Store) {
    let now = now_rfc3339();
    for (file, content) in read_eval_docs() {
        let title = title_of(&content, &file);
        let hash = hash_content(&content);
        store
            .with_connection(|c| insert_content(c, &hash, &content, &now))
            .unwrap();
        store
            .with_connection(|c| insert_document(c, "eval-docs", &file, &title, &hash, &now, &now))
            .unwrap();
    }
}

// =============================================================================
// BM25 (Lexical) Tests — fast, no model loading needed
// =============================================================================

/// Hit-rate over `search_fts` (limit 5), counting a query as a hit when any of
/// the top-`top_k` results matches the expected doc. Mirrors qmd's `calcHitRate`
/// with `searchFn = q => searchFTS(db, q, 5)`.
fn fts_hit_rate(store: &Store, queries: &[&EvalQuery], top_k: usize) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let results = store
            .with_connection(|c| search_fts(c, q.query, Some(5), None))
            .unwrap();
        if results
            .iter()
            .take(top_k)
            .any(|r| matches_expected(&r.doc.filepath, q.expected))
        {
            hits += 1;
        }
    }
    hits as f64 / queries.len() as f64
}

fn bm25_store() -> (NamedTempFile, Store) {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    index_docs(&store);
    (tmp, store)
}

#[test]
fn bm25_easy_queries_hit_at_3() {
    let (_t, store) = bm25_store();
    let rate = fts_hit_rate(&store, &by_difficulty(Easy), 3);
    assert!(rate >= 0.8, "easy BM25 Hit@3 = {rate} (want >= 0.8)");
}

#[test]
fn bm25_medium_queries_hit_at_3() {
    // BM25 struggles with semantic queries.
    let (_t, store) = bm25_store();
    let rate = fts_hit_rate(&store, &by_difficulty(Medium), 3);
    assert!(rate >= 0.15, "medium BM25 Hit@3 = {rate} (want >= 0.15)");
}

#[test]
fn bm25_hard_queries_hit_at_5() {
    let (_t, store) = bm25_store();
    let rate = fts_hit_rate(&store, &by_difficulty(Hard), 5);
    assert!(rate >= 0.15, "hard BM25 Hit@5 = {rate} (want >= 0.15)");
}

#[test]
fn bm25_overall_hit_at_3() {
    let (_t, store) = bm25_store();
    let all: Vec<&EvalQuery> = EVAL_QUERIES.iter().collect();
    let rate = fts_hit_rate(&store, &all, 3);
    assert!(rate >= 0.4, "overall BM25 Hit@3 = {rate} (want >= 0.4)");
}

// =============================================================================
// Vector + Hybrid (RRF) Tests — require the embedding model
// =============================================================================

/// Run by default; skip when `RQMD_SKIP_LLM_TESTS` is set (CI sets it).
fn skip_llm() -> bool {
    std::env::var("RQMD_SKIP_LLM_TESTS").is_ok()
}

fn to_ranked(r: &SearchResult) -> RankedResult {
    RankedResult {
        file: r.doc.filepath.clone(),
        display_path: r.doc.display_path.clone(),
        title: r.doc.title.clone(),
        body: r.doc.body.clone().unwrap_or_default(),
        score: r.score,
    }
}

/// Vector-search hit-rate: `search_vec` (limit 5), hit when any of the
/// top-`top_k` results matches. Mirrors the inline loops in qmd's
/// `describe("Vector Search")`.
async fn vec_hit_rate(
    store: &Store,
    llm: Arc<dyn Llm>,
    model: &str,
    queries: &[&EvalQuery],
    top_k: usize,
) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let results = search_vec(store, llm.clone(), q.query, model, 5, None, None)
            .await
            .unwrap();
        if results
            .iter()
            .take(top_k)
            .any(|r| matches_expected(&r.doc.filepath, q.expected))
        {
            hits += 1;
        }
    }
    hits as f64 / queries.len() as f64
}

/// Hybrid search with RRF fusion. Mirrors qmd's `hybridSearch`: FTS (limit 20) +
/// vector (limit 20) → RRF (weights [1,1], k=60) → top `limit`.
async fn hybrid_search(
    store: &Store,
    llm: Arc<dyn Llm>,
    model: &str,
    query: &str,
    limit: usize,
) -> Vec<RankedResult> {
    let mut lists: Vec<Vec<RankedResult>> = Vec::new();

    let fts = store
        .with_connection(|c| search_fts(c, query, Some(20), None))
        .unwrap();
    if !fts.is_empty() {
        lists.push(fts.iter().map(to_ranked).collect());
    }

    let vec = search_vec(store, llm.clone(), query, model, 20, None, None)
        .await
        .unwrap();
    if !vec.is_empty() {
        lists.push(vec.iter().map(to_ranked).collect());
    }

    if lists.is_empty() {
        return Vec::new();
    }
    let fused = reciprocal_rank_fusion(&lists, &[1.0, 1.0], None);
    fused.into_iter().take(limit).collect()
}

/// Hybrid hit-rate. `top_k = Some(k)` counts a hit when any of the top-k fused
/// results matches; `None` checks the full fused list (qmd's `.some()` for the
/// hard bucket).
async fn hybrid_hit_rate(
    store: &Store,
    llm: Arc<dyn Llm>,
    model: &str,
    queries: &[&EvalQuery],
    top_k: Option<usize>,
) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let r = hybrid_search(store, llm.clone(), model, q.query, 10).await;
        let hit = match top_k {
            Some(k) => r
                .iter()
                .take(k)
                .any(|x| matches_expected(&x.file, q.expected)),
            None => r.iter().any(|x| matches_expected(&x.file, q.expected)),
        };
        if hit {
            hits += 1;
        }
    }
    hits as f64 / queries.len() as f64
}

/// Returns a failure message when `rate` is below `threshold`, else `None`.
/// Lets the LLM test collect every regressed metric instead of aborting on the
/// first `assert!` (the TS suite reports all failures via separate `test()`s).
fn threshold_failure(name: &str, rate: f64, threshold: f64) -> Option<String> {
    (rate < threshold).then(|| format!("{name} = {rate:.3} (want >= {threshold})"))
}

#[tokio::test(flavor = "multi_thread")]
async fn eval_vector_and_hybrid_search() {
    if skip_llm() {
        eprintln!("RQMD_SKIP_LLM_TESTS set — skipping Vector/Hybrid eval suite");
        return;
    }

    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    index_docs(&store);

    // Real (non-CI) embed model. First run downloads embeddinggemma-300M (~300 MB).
    let llm = Arc::new(LlamaCpp::new(LlamaCppConfig::default()));
    let llm_dyn: Arc<dyn Llm> = llm.clone();
    // Pin the model string used for BOTH the stored embeddings and the query
    // side so kNN formatting stays consistent.
    let model_uri = llm.embed_model_uri().to_string();
    let now = now_rfc3339();

    // Seed embeddings: chunk each doc, embed each chunk (raw text — embed()
    // formats internally), create the vec table from the first vector's dim,
    // then insert. Mirrors qmd's `beforeAll` in `describe("Vector Search")`.
    let mut vec_table_ready = false;
    for (file, content) in read_eval_docs() {
        let title = title_of(&content, &file);
        let hash = hash_content(&content);
        let chunks = chunk_document_by_tokens(
            llm_dyn.clone(),
            &content,
            None,
            None,
            None,
            None,
            ChunkStrategy::Auto,
            None,
        )
        .await
        .unwrap();
        let total = chunks.len() as i64;
        for (seq, chunk) in chunks.iter().enumerate() {
            let result = llm
                .embed(
                    &chunk.text,
                    EmbedOptions {
                        model: None,
                        is_query: false,
                        title: Some(title.clone()),
                    },
                )
                .await
                .unwrap();
            let Some(emb) = result else { continue };
            if !vec_table_ready {
                store
                    .with_connection(|c| ensure_vec_table(c, emb.embedding.len()))
                    .unwrap();
                vec_table_ready = true;
            }
            store
                .with_connection(|c| {
                    insert_embedding(
                        c,
                        &hash,
                        seq as i64,
                        chunk.pos as i64,
                        &emb.embedding,
                        &model_uri,
                        &now,
                        total,
                    )
                })
                .unwrap();
        }
    }
    assert!(
        vec_table_ready,
        "expected at least one embedding to be seeded"
    );

    // Compute every metric (single model load) and collect failures so one
    // regressed threshold doesn't mask the others — a Rust `assert!` aborts on
    // the first failure, whereas the TS suite reports all via separate tests.
    let easy = by_difficulty(Easy);
    let medium = by_difficulty(Medium);
    let hard = by_difficulty(Hard);
    let fusion = by_difficulty(Fusion);
    let all: Vec<&EvalQuery> = EVAL_QUERIES.iter().collect();
    let non_fusion: Vec<&EvalQuery> = EVAL_QUERIES
        .iter()
        .filter(|q| q.difficulty != Fusion)
        .collect();

    let mut failures: Vec<String> = Vec::new();

    // ---- Vector Search (describe("Vector Search")) ----
    // easy: vector should match keywords too.
    let r = vec_hit_rate(&store, llm_dyn.clone(), &model_uri, &easy, 3).await;
    failures.extend(threshold_failure("easy vector Hit@3", r, 0.6));
    // medium: vector excels at semantic.
    let r = vec_hit_rate(&store, llm_dyn.clone(), &model_uri, &medium, 3).await;
    failures.extend(threshold_failure("medium vector Hit@3", r, 0.4));
    // hard: vector helps with vague queries — Hit@5 (top_k = limit = 5).
    let r = vec_hit_rate(&store, llm_dyn.clone(), &model_uri, &hard, 5).await;
    failures.extend(threshold_failure("hard vector Hit@5", r, 0.3));
    // overall: vector baseline.
    let r = vec_hit_rate(&store, llm_dyn.clone(), &model_uri, &all, 3).await;
    failures.extend(threshold_failure("overall vector Hit@3", r, 0.5));

    // ---- Hybrid Search (RRF) (describe("Hybrid Search (RRF)")) ----
    // The real model is always present here, so qmd's `hasVectors=false`
    // fallback thresholds collapse to the with-vectors values.
    // easy: hybrid should match BM25.
    let r = hybrid_hit_rate(&store, llm_dyn.clone(), &model_uri, &easy, Some(3)).await;
    failures.extend(threshold_failure("easy hybrid Hit@3", r, 0.8));
    // medium: hybrid should outperform both BM25 and vector.
    let r = hybrid_hit_rate(&store, llm_dyn.clone(), &model_uri, &medium, Some(3)).await;
    failures.extend(threshold_failure("medium hybrid Hit@3", r, 0.5));
    // hard: `.some()` over the full fused list (≤10), per qmd's exact code.
    let r = hybrid_hit_rate(&store, llm_dyn.clone(), &model_uri, &hard, None).await;
    failures.extend(threshold_failure("hard hybrid Hit@5", r, 0.35));
    // overall (non-fusion queries; fusion is tested separately).
    let r = hybrid_hit_rate(&store, llm_dyn.clone(), &model_uri, &non_fusion, Some(3)).await;
    failures.extend(threshold_failure("overall hybrid Hit@3", r, 0.6));

    // fusion: RRF combines weak signals — must clear 50% AND beat the best
    // individual method. One pass computes all three rates.
    let mut hybrid_hits = 0usize;
    let mut bm25_hits = 0usize;
    let mut vec_hits = 0usize;
    for q in &fusion {
        let hybrid = hybrid_search(&store, llm_dyn.clone(), &model_uri, q.query, 10).await;
        if hybrid
            .iter()
            .take(3)
            .any(|x| matches_expected(&x.file, q.expected))
        {
            hybrid_hits += 1;
        }

        let bm25 = store
            .with_connection(|c| search_fts(c, q.query, Some(5), None))
            .unwrap();
        if bm25
            .iter()
            .take(3)
            .any(|r| matches_expected(&r.doc.filepath, q.expected))
        {
            bm25_hits += 1;
        }

        let vec = search_vec(&store, llm_dyn.clone(), q.query, &model_uri, 5, None, None)
            .await
            .unwrap();
        if vec
            .iter()
            .take(3)
            .any(|r| matches_expected(&r.doc.filepath, q.expected))
        {
            vec_hits += 1;
        }
    }
    let n = fusion.len() as f64;
    let hybrid_rate = hybrid_hits as f64 / n;
    let bm25_rate = bm25_hits as f64 / n;
    let vec_rate = vec_hits as f64 / n;
    failures.extend(threshold_failure("fusion hybrid Hit@3", hybrid_rate, 0.5));
    if hybrid_rate < bm25_rate.max(vec_rate) {
        failures.push(format!(
            "fusion hybrid {hybrid_rate:.3} should match/beat best of bm25 {bm25_rate:.3} / vec {vec_rate:.3}"
        ));
    }

    // Release native resources before asserting (mirrors qmd's
    // `disposeDefaultLlamaCpp` in afterAll — runs even when thresholds fail).
    llm.dispose().await;

    assert!(
        failures.is_empty(),
        "Vector/Hybrid eval thresholds failed:\n  {}",
        failures.join("\n  ")
    );
}
