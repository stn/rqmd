//! `MockLlm` — a hand-rolled `Llm` impl for orchestrator unit tests.
//!
//! Behaviour:
//! * `embed` / `embed_batch` — deterministic per-text vectors. SHA-256 of
//!   the text seeds a `dim`-length `f32` slice in `[-1, 1]`. Tests that need
//!   custom vectors can pre-populate `embed_overrides`.
//! * `expand_query` — returns the canned response for `query` if registered,
//!   otherwise a synthetic `[hyde, vec]` pair.
//! * `rerank` — uses `rerank_overrides` keyed on chunk text; falls back to
//!   keyword-overlap scoring.
//! * `tokenize` — `ceil(len / chars_per_token)` synthetic tokens. Stable so
//!   `chunk_document_by_tokens` tests can reason about the count.
//! * `detokenize` — reverses tokenize at the byte-length level (`"a"` repeated).
//! * `generate` — returns a fixed stub. Not exercised by orchestrator tests.
//! * `model_exists` — always `true`.
//! * `dispose` — no-op.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use rmd_llm::traits::{LlamaToken, Llm};
use rmd_llm::types::{
    EmbedOptions, EmbeddingResult, ExpandQueryOptions, GenerateOptions, GenerateResult, ModelInfo,
    QueryType, Queryable, RerankDocument, RerankDocumentResult, RerankOptions, RerankResult,
};

pub struct MockLlm {
    pub embed_dim: usize,
    pub chars_per_token: f64,

    pub embed_overrides: Mutex<std::collections::HashMap<String, Vec<f32>>>,
    pub expand_overrides: Mutex<std::collections::HashMap<String, Vec<Queryable>>>,
    pub rerank_overrides: Mutex<std::collections::HashMap<String, f32>>,

    pub embed_calls: AtomicUsize,
    pub embed_batch_calls: AtomicUsize,
    pub expand_calls: AtomicUsize,
    pub rerank_calls: AtomicUsize,
    pub tokenize_calls: AtomicUsize,
    pub detokenize_calls: AtomicUsize,
}

impl MockLlm {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            embed_dim,
            chars_per_token: 3.0,
            embed_overrides: Mutex::new(Default::default()),
            expand_overrides: Mutex::new(Default::default()),
            rerank_overrides: Mutex::new(Default::default()),
            embed_calls: AtomicUsize::new(0),
            embed_batch_calls: AtomicUsize::new(0),
            expand_calls: AtomicUsize::new(0),
            rerank_calls: AtomicUsize::new(0),
            tokenize_calls: AtomicUsize::new(0),
            detokenize_calls: AtomicUsize::new(0),
        }
    }

    pub fn with_chars_per_token(mut self, c: f64) -> Self {
        self.chars_per_token = c;
        self
    }

    pub fn set_embed(&self, text: impl Into<String>, embedding: Vec<f32>) {
        self.embed_overrides
            .lock()
            .unwrap()
            .insert(text.into(), embedding);
    }

    pub fn set_expand(&self, query: impl Into<String>, results: Vec<Queryable>) {
        self.expand_overrides
            .lock()
            .unwrap()
            .insert(query.into(), results);
    }

    pub fn set_rerank_score(&self, chunk: impl Into<String>, score: f32) {
        self.rerank_overrides
            .lock()
            .unwrap()
            .insert(chunk.into(), score);
    }
}

fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let seed = h.finalize();
    (0..dim)
        .map(|i| (seed[i % 32] as f32 / 128.0) - 1.0)
        .collect()
}

fn keyword_overlap_score(query: &str, text: &str) -> f32 {
    let q_lower = query.to_lowercase();
    let t_lower = text.to_lowercase();
    let qs: Vec<&str> = q_lower.split_whitespace().filter(|t| t.len() > 1).collect();
    if qs.is_empty() {
        return 0.0;
    }
    let hits = qs.iter().filter(|t| t_lower.contains(*t)).count();
    hits as f32 / qs.len() as f32
}

#[async_trait]
impl Llm for MockLlm {
    async fn embed(&self, text: &str, _opts: EmbedOptions) -> rmd_llm::Result<Option<EmbeddingResult>> {
        self.embed_calls.fetch_add(1, Ordering::Relaxed);
        let v = self
            .embed_overrides
            .lock()
            .unwrap()
            .get(text)
            .cloned()
            .unwrap_or_else(|| deterministic_embedding(text, self.embed_dim));
        Ok(Some(EmbeddingResult {
            embedding: v,
            model: "mock".into(),
        }))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        opts: EmbedOptions,
    ) -> rmd_llm::Result<Vec<Option<EmbeddingResult>>> {
        self.embed_batch_calls.fetch_add(1, Ordering::Relaxed);
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t, opts.clone()).await?);
        }
        Ok(out)
    }

    async fn generate(
        &self,
        _prompt: &str,
        _opts: GenerateOptions,
    ) -> rmd_llm::Result<Option<GenerateResult>> {
        Ok(Some(GenerateResult {
            text: "mock-generation".into(),
            model: "mock".into(),
            logprobs: None,
            done: true,
        }))
    }

    async fn model_exists(&self, name: &str) -> rmd_llm::Result<ModelInfo> {
        Ok(ModelInfo {
            name: name.into(),
            exists: true,
            path: None,
        })
    }

    async fn expand_query(
        &self,
        query: &str,
        _opts: ExpandQueryOptions,
    ) -> rmd_llm::Result<Vec<Queryable>> {
        self.expand_calls.fetch_add(1, Ordering::Relaxed);
        let v = self
            .expand_overrides
            .lock()
            .unwrap()
            .get(query)
            .cloned()
            .unwrap_or_else(|| {
                vec![
                    Queryable {
                        type_: QueryType::Hyde,
                        text: format!("doc about {query}"),
                    },
                    Queryable {
                        type_: QueryType::Vec,
                        text: format!("semantic {query}"),
                    },
                ]
            });
        Ok(v)
    }

    async fn rerank(
        &self,
        query: &str,
        docs: &[RerankDocument],
        _opts: RerankOptions,
    ) -> rmd_llm::Result<RerankResult> {
        self.rerank_calls.fetch_add(1, Ordering::Relaxed);
        let overrides = self.rerank_overrides.lock().unwrap();
        let mut results: Vec<RerankDocumentResult> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let score = overrides
                    .get(&d.text)
                    .copied()
                    .unwrap_or_else(|| keyword_overlap_score(query, &d.text));
                RerankDocumentResult {
                    file: d.file.clone(),
                    score,
                    index: i,
                }
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(RerankResult {
            results,
            model: "mock".into(),
        })
    }

    async fn tokenize(&self, text: &str) -> rmd_llm::Result<Vec<LlamaToken>> {
        self.tokenize_calls.fetch_add(1, Ordering::Relaxed);
        let n = ((text.len() as f64) / self.chars_per_token).ceil() as usize;
        Ok((0..n).map(|i| LlamaToken::new(i as i32)).collect())
    }

    async fn detokenize(&self, tokens: &[LlamaToken]) -> rmd_llm::Result<String> {
        self.detokenize_calls.fetch_add(1, Ordering::Relaxed);
        let n = ((tokens.len() as f64) * self.chars_per_token).round() as usize;
        Ok("a".repeat(n))
    }

    async fn dispose(&self) {}
}
