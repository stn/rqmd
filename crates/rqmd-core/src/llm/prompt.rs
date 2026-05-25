//! Prompt construction helpers shared between PR1 (pure string building)
//! and PR2 (which will combine these with the model's chat template via
//! `LlamaModel::apply_chat_template`).
//!
//! Two prompt shapes live here:
//!
//! * Qwen3-Reranker pairwise judgment prompt. Spike #3 measured
//!   `tokio=0.9985 > rails=0.0007 > json=0.0001` (1400× separation) with
//!   this prompt vs. unusable ordering for the generic
//!   `{query}</s><s>{doc}` placeholder. The yes/no judgment isn't read
//!   under `pooling=Rank`, but the prompt *shape* is what the rank head
//!   was trained on.
//! * `expand_query` user message. The fine-tuned `qmd-query-expansion-1.7B`
//!   model was trained in qmd's exact setup: *no system message*, only the
//!   `/no_think Expand this search query: ...` user turn. Adding a system
//!   prompt (the previous rqmd approach) put the model out of distribution
//!   and made it drift to its Qwen3 base Chinese prior on some backends; we
//!   mirror qmd verbatim instead. Output is also constrained with the same
//!   GBNF grammar qmd uses ([`EXPAND_QUERY_GRAMMAR`]); `sample_token` in
//!   `llama_cpp.rs` applies it via llama.cpp's reference grammar-first flow,
//!   which sidesteps the grammar-sampler abort the naive chain hit.
//!   [`parse_expand_query_output`] + [`fallback_queryables`] stay as a lenient
//!   recovery layer.

/// The instruction string that goes into the Qwen3-Reranker user message.
/// Pulled out as a constant so it can be tweaked centrally if/when we
/// support different rerank tasks (currently only general web search).
pub const QWEN3_RERANKER_INSTRUCT: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Build the Qwen3-Reranker chat-formatted prompt for one (query, doc)
/// pair, ready for tokenization with `AddBos::Never` (the chat template
/// brackets already include the equivalent of BOS).
///
/// See `the original tobi/qmd spike_03_rerank (since removed)` for the measurement
/// that justifies this exact format.
pub fn build_qwen3_rerank_prompt(query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
         Query and the Instruct provided. Note that the answer can only be \"yes\" or \
         \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruct}\n<Query>: {query}\n\
         <Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        instruct = QWEN3_RERANKER_INSTRUCT,
        query = query,
        document = document,
    )
}

/// Build the user-side message for `expand_query`. Includes the `/no_think`
/// prefix so Qwen3 (and similar) skips the chain-of-thought block — we
/// don't want `<think>...</think>` content interleaved with our lines.
///
/// The optional `intent` mirrors `expandQuery(..., { intent })` in TS
/// (lines 1309–1312): an extra hint about *why* the user is searching.
pub fn build_expand_query_user_message(query: &str, intent: Option<&str>) -> String {
    match intent {
        Some(intent) => {
            format!("/no_think Expand this search query: {query}\nQuery intent: {intent}")
        }
        None => format!("/no_think Expand this search query: {query}"),
    }
}

/// GBNF grammar mirroring qmd (`src/llm.ts`): force the expansion model to emit
/// one or more `type: content` lines so the output is always parseable. The
/// `root ::= line+` rule reaches an end-state after every completed line, so the
/// model can stop (EOG) at a line boundary. Kept verbatim with qmd — keep in
/// sync. Applied via `sample_token` (grammar-first), not by appending it to the
/// sampler chain (which aborts; see [`crate::llm`] module docs).
pub const EXPAND_QUERY_GRAMMAR: &str = r#"root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+
"#;

/// Parse the raw model output from `expand_query` into structured
/// `Queryable`s. Skips any lines that don't begin with one of the three
/// known prefixes (this is how we recover from `<think>...</think>`
/// noise without grammar enforcement).
///
/// Returns an empty Vec if no lines match; the caller is responsible for
/// applying fallback behavior (TS lines 1362–1373).
pub fn parse_expand_query_output(raw: &str) -> Vec<crate::llm::types::Queryable> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("lex:") {
            out.push(make_queryable(crate::llm::types::QueryType::Lex, rest));
        } else if let Some(rest) = line.strip_prefix("vec:") {
            out.push(make_queryable(crate::llm::types::QueryType::Vec, rest));
        } else if let Some(rest) = line.strip_prefix("hyde:") {
            out.push(make_queryable(crate::llm::types::QueryType::Hyde, rest));
        }
    }
    out
}

fn make_queryable(type_: crate::llm::types::QueryType, rest: &str) -> crate::llm::types::Queryable {
    // The strict TS grammar required `": "` (colon-space) after the type.
    // We're looser here — accept any leading whitespace after the colon,
    // matching what spike #4 actually observed.
    crate::llm::types::Queryable {
        type_,
        text: rest.trim().to_owned(),
    }
}

/// Filter `Queryable`s to those that mention at least one term from the
/// original query. Mirrors TS `hasQueryTerm` (lines 1339–1356); the goal
/// is to reject hallucinated expansions that wander off-topic.
///
/// Tokenization is the same as TS: lowercase, replace non-alphanumeric
/// with whitespace, split on whitespace.
pub fn filter_with_query_terms(
    query: &str,
    candidates: Vec<crate::llm::types::Queryable>,
) -> Vec<crate::llm::types::Queryable> {
    let terms: Vec<String> = query
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_owned())
        .collect();

    if terms.is_empty() {
        return candidates;
    }

    candidates
        .into_iter()
        .filter(|q| {
            let lower = q.text.to_lowercase();
            terms.iter().any(|t| lower.contains(t))
        })
        .collect()
}

/// Build the fallback `Queryable` list when expansion produces nothing
/// usable. Mirrors TS lines 1362–1373.
pub fn fallback_queryables(
    query: &str,
    include_lexical: bool,
) -> Vec<crate::llm::types::Queryable> {
    let mut out = vec![
        crate::llm::types::Queryable {
            type_: crate::llm::types::QueryType::Hyde,
            text: format!("Information about {query}"),
        },
        crate::llm::types::Queryable {
            type_: crate::llm::types::QueryType::Lex,
            text: query.to_owned(),
        },
        crate::llm::types::Queryable {
            type_: crate::llm::types::QueryType::Vec,
            text: query.to_owned(),
        },
    ];
    if !include_lexical {
        out.retain(|q| q.type_ != crate::llm::types::QueryType::Lex);
    }
    out
}
