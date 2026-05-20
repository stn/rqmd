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
//! * `expand_query` system message that biases the model toward emitting
//!   `lex: ... / vec: ... / hyde: ...` lines without relying on GBNF
//!   grammar enforcement (spike #4 ruled grammar out as brittle).
//!
//! No `LlamaModel` dependency yet — PR2 will add the chat-template
//! wrappers that combine these strings into the final tokenizable form.

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

/// System prompt used by `expand_query`. Asks the model to emit exactly
/// three lines of the form `lex: ...` / `vec: ...` / `hyde: ...`, with
/// no commentary. Spike #4 confirmed this prompt + a chat-tuned model
/// reliably produces parseable output without GBNF enforcement.
pub const EXPAND_QUERY_SYSTEM_PROMPT: &str =
    "You expand search queries. Output ONLY 3 lines, each starting with one of \
     `lex: `, `vec: `, or `hyde: ` followed by a query variation. \
     No commentary, no other text.";

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
pub fn fallback_queryables(query: &str, include_lexical: bool) -> Vec<crate::llm::types::Queryable> {
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
