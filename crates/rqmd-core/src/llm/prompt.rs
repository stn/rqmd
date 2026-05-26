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

/// Build the user-side message for `expand_query`.
///
/// `prefix` is caller-provided (typically resolved by
/// [`crate::llm::config::resolve_expand_user_message_prefix`]). The crate
/// default is `/no_think` (Qwen3 chain-of-thought suppression). A separator
/// space is auto-inserted iff `prefix` is non-empty and does not already end
/// with whitespace — callers do not need to remember a trailing space.
///
/// The optional `intent` mirrors `expandQuery(..., { intent })` in TS
/// (lines 1309–1312): an extra hint about *why* the user is searching.
///
/// # qmd parity divergence — do not revert
///
/// Upstream qmd `src/llm.ts:1449-1451` hardcodes `/no_think` unconditionally,
/// which only works for Qwen3-based finetunes. qmd itself recognises this
/// limitation in `finetune/dataset/prepare_data_lfm2.py:10`
/// (`"No /no_think needed (that's Qwen3-specific)."`). This function's
/// `prefix` argument exists so non-Qwen finetunes (Llama Swallow, LFM2, …)
/// can pass `""` and get a clean user message. Future maintainers: do not
/// "fix" this by hardcoding the prefix back in the name of qmd parity.
///
/// The `EXPAND_QUERY_GRAMMAR` (see below) constrains *output* to the
/// `lex|vec|hyde: ...` line schema for every model, so finetunes are expected
/// to emit qmd-compatible output regardless of the prefix choice.
pub fn build_expand_query_user_message(query: &str, intent: Option<&str>, prefix: &str) -> String {
    let sep = match prefix.chars().last() {
        None => "",                         // empty prefix
        Some(c) if c.is_whitespace() => "", // already separated
        Some(_) => " ",                     // needs separator
    };
    match intent {
        Some(intent) => {
            format!("{prefix}{sep}Expand this search query: {query}\nQuery intent: {intent}")
        }
        None => format!("{prefix}{sep}Expand this search query: {query}"),
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
///
/// `hyde_template` is caller-provided (typically resolved by
/// [`crate::llm::config::resolve_expand_fallback_hyde_template`]); the crate
/// default is `"Information about {query}"`. `{query}` is substituted with
/// the original query via plain `str::replace`. If the template contains no
/// `{query}` placeholder the template is used verbatim (intentional — not a
/// bug).
///
/// # qmd parity divergence — do not revert
///
/// Upstream qmd `src/llm.ts:1502-1504` hardcodes the English "Information
/// about" template. Making it configurable here lets non-English (e.g.
/// Japanese) deployments produce coherent fallback text. Future maintainers:
/// do not hardcode this back.
pub fn fallback_queryables(
    query: &str,
    include_lexical: bool,
    hyde_template: &str,
) -> Vec<crate::llm::types::Queryable> {
    let hyde_text = hyde_template.replace("{query}", query);
    let mut out = vec![
        crate::llm::types::Queryable {
            type_: crate::llm::types::QueryType::Hyde,
            text: hyde_text,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::QueryType;

    // -------------------------------------------------------------------
    // build_expand_query_user_message — separator auto-insertion
    // -------------------------------------------------------------------

    #[test]
    fn user_message_default_prefix_auto_inserts_space() {
        assert_eq!(
            build_expand_query_user_message("foo", None, "/no_think"),
            "/no_think Expand this search query: foo",
        );
    }

    #[test]
    fn user_message_prefix_with_trailing_space_is_idempotent() {
        // Pre-feature behaviour passed a literal `"/no_think "` (trailing
        // space) — the new auto-separator must not double-space.
        assert_eq!(
            build_expand_query_user_message("foo", None, "/no_think "),
            "/no_think Expand this search query: foo",
        );
    }

    #[test]
    fn user_message_prefix_ending_in_newline_skips_separator() {
        assert_eq!(
            build_expand_query_user_message("foo", None, "/no_think\n"),
            "/no_think\nExpand this search query: foo",
        );
    }

    #[test]
    fn user_message_prefix_ending_in_tab_skips_separator() {
        // Locks down `char::is_whitespace` semantics — tab counts.
        assert_eq!(
            build_expand_query_user_message("foo", None, "/no_think\t"),
            "/no_think\tExpand this search query: foo",
        );
    }

    #[test]
    fn user_message_empty_prefix_emits_bare_body() {
        // Llama Swallow / non-Qwen path: prefix `""` produces a clean message
        // with no leading space.
        assert_eq!(
            build_expand_query_user_message("foo", None, ""),
            "Expand this search query: foo",
        );
    }

    #[test]
    fn user_message_prefix_ending_in_colon_gets_separator() {
        // Non-whitespace, non-empty trailing char → insert space.
        assert_eq!(
            build_expand_query_user_message("foo", None, "### Task:"),
            "### Task: Expand this search query: foo",
        );
    }

    #[test]
    fn user_message_with_intent_appends_intent_line() {
        assert_eq!(
            build_expand_query_user_message("foo", Some("research"), "/no_think"),
            "/no_think Expand this search query: foo\nQuery intent: research",
        );
    }

    // -------------------------------------------------------------------
    // fallback_queryables — {query} placeholder substitution
    // -------------------------------------------------------------------

    #[test]
    fn fallback_default_template_matches_pre_feature_behaviour() {
        let out = fallback_queryables("日本の首都", true, "Information about {query}");
        let hyde = out.iter().find(|q| q.type_ == QueryType::Hyde).unwrap();
        assert_eq!(hyde.text, "Information about 日本の首都");
        // Lex + Vec slots are the raw query.
        assert!(
            out.iter()
                .any(|q| q.type_ == QueryType::Lex && q.text == "日本の首都")
        );
        assert!(
            out.iter()
                .any(|q| q.type_ == QueryType::Vec && q.text == "日本の首都")
        );
    }

    #[test]
    fn fallback_japanese_template_produces_coherent_text() {
        let out = fallback_queryables("日本の首都", true, "{query}に関する情報");
        let hyde = out.iter().find(|q| q.type_ == QueryType::Hyde).unwrap();
        assert_eq!(hyde.text, "日本の首都に関する情報");
    }

    #[test]
    fn fallback_excludes_lex_when_not_requested() {
        let out = fallback_queryables("日本の首都", false, "Information about {query}");
        assert!(out.iter().all(|q| q.type_ != QueryType::Lex));
        assert!(out.iter().any(|q| q.type_ == QueryType::Hyde));
        assert!(out.iter().any(|q| q.type_ == QueryType::Vec));
    }

    #[test]
    fn fallback_template_without_placeholder_used_verbatim() {
        // Intentional spec: missing `{query}` is not an error; the template
        // is used as-is. This is documented in the doc-comment.
        let out = fallback_queryables("foo", true, "no placeholder here");
        let hyde = out.iter().find(|q| q.type_ == QueryType::Hyde).unwrap();
        assert_eq!(hyde.text, "no placeholder here");
    }
}
