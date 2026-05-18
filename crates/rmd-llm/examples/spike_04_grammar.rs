//! Spike #4 — verify critical assumption #4 (grammar-constrained sampling)
//! and the manual decode/sample loop more broadly.
//!
//! Findings while writing this spike (recorded here for the v2 plan):
//!   * `LlamaGrammar::from_str` does NOT exist. Grammar is part of the
//!     sampler chain via `LlamaSampler::grammar(model, src, root)` /
//!     `LlamaSampler::grammar_lazy(...)`.
//!   * Strict grammar enforcement on a chat-tuned model crashes
//!     `GGML_ASSERT(!stacks.empty())` in `llama-grammar.cpp:940`. Two
//!     mismatches feed this:
//!       1. The very first generated tokens are chat-template / `<think>` /
//!          whitespace tokens that no `root` rule accepts.
//!       2. `grammar_lazy` does activate on a trigger word, but it expects
//!          the *next* token to start matching `root` fresh — fighting BPE
//!          merges like ` :`, `:\n`, ` lex:` and so on. Tweaking the
//!          grammar to allow these merges is possible but brittle, and a
//!          mistake silently aborts the process.
//!   * Conclusion: for `expandQuery`, do NOT rely on GBNF to enforce the
//!     output schema. Use the model's chat template, prompt clearly for
//!     the desired format, generate without grammar, and parse the output
//!     in Rust. This is what this spike now validates end-to-end.
//!
//! Pass criteria:
//!   * Sampler chain (top_k → top_p → temp → penalties → dist) drives a
//!     manual decode loop to completion via `model.is_eog_token`.
//!   * Chat template is applied via `model.chat_template` + `apply_chat_template`.
//!   * At least one output line parses as `^(lex|vec|hyde): .+$`.
//!   * Record the exact LlamaSampler chain API and decode loop shape.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

const HF_REPO: &str = "ggml-org/Qwen3-0.6B-GGUF";
const HF_FILE: &str = "Qwen3-0.6B-Q8_0.gguf";

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("backend init"))
}

fn resolve_model_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("SPIKE_MODEL_PATH") {
        let p = PathBuf::from(path);
        anyhow::ensure!(
            p.exists(),
            "SPIKE_MODEL_PATH does not exist: {}",
            p.display()
        );
        return Ok(p);
    }
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("failed to build hf-hub API")?;
    api.model(HF_REPO.to_string())
        .get(HF_FILE)
        .with_context(|| format!("failed to download {HF_REPO}/{HF_FILE}"))
}

fn main() -> Result<()> {
    let backend = backend();
    let model_path = resolve_model_path()?;
    eprintln!("model path = {}", model_path.display());

    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
        .context("LlamaModel::load_from_file")?;

    let ctx_params =
        LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).expect("non-zero")));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("model.new_context")?;

    // Sampler chain (no grammar — see top-of-file commentary on why).
    // Order matters: candidate filtering (top_k/top_p) → temperature →
    // penalties → terminal selector (dist).
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(20),
        LlamaSampler::top_p(0.8, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::penalties(64, 1.0, 0.0, 0.5),
        LlamaSampler::dist(1234),
    ]);

    // Build a proper Qwen3 chat prompt via the model's embedded chat template,
    // rather than feeding raw text. Without this, Qwen3 produces unstructured
    // output that the grammar can't latch onto.
    let chat_template = model
        .chat_template(None)
        .context("model has no embedded chat template")?;
    let user_msg = LlamaChatMessage::new(
        "user".to_string(),
        "/no_think Expand this search query into 3 lines. Each line MUST start with `lex: `, \
         `vec: `, or `hyde: ` and contain a query variation.\n\n\
         Query: rust async runtime"
            .to_string(),
    )?;
    let prompt = model
        .apply_chat_template(&chat_template, &[user_msg], true)
        .context("apply_chat_template")?;
    eprintln!("--- chat-formatted prompt ---\n{prompt}\n--- end prompt ---");

    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .with_context(|| format!("failed to tokenize {prompt:?}"))?;
    eprintln!("prompt tokens = {}", tokens.len());

    // Prime the context with the prompt. Only the last token needs logits
    // because that's where we'll sample the next token from.
    let mut batch = LlamaBatch::new(2048, 1);
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        batch.add(*token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch).context("initial ctx.decode")?;

    // token_to_piece needs a stateful decoder so multi-byte UTF-8 sequences
    // spanning multiple tokens are reassembled correctly.
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    let max_new_tokens = 200_i32;

    for _ in 0..max_new_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model
            .token_to_piece(
                token,
                &mut decoder,
                /* special */ false,
                /* lstrip */ None,
            )
            .unwrap_or_else(|_| String::from("\u{FFFD}"));
        output.push_str(&piece);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch).context("loop ctx.decode")?;
    }

    println!("--- output ---");
    println!("{output}");
    println!("--- end ---");

    // Post-hoc validation: count how many lines parse as `^(lex|vec|hyde): .+$`.
    // Without grammar enforcement, the model may produce extra `<think>` or
    // commentary lines — we just need at least one match to prove the loop
    // produced usable output.
    let mut matched = 0_u32;
    let mut total = 0_u32;
    for line in output.lines().filter(|l| !l.trim().is_empty()) {
        total += 1;
        let trimmed = line.trim_start();
        let is_match = trimmed.starts_with("lex:")
            || trimmed.starts_with("vec:")
            || trimmed.starts_with("hyde:");
        if is_match {
            matched += 1;
        }
    }
    println!("matched {matched}/{total} non-empty lines");
    anyhow::ensure!(matched > 0, "expected at least one lex:/vec:/hyde: line");
    println!("OK: sampler chain + manual decode loop produced parseable output.");

    println!();
    println!("Notes for v2 plan:");
    println!("  - LlamaSampler::grammar / grammar_lazy exist and compile, BUT");
    println!("    are brittle in practice: chat-tuned models emit chat-template");
    println!("    and `<think>` tokens that easily exhaust the grammar candidate");
    println!("    set and abort the process via GGML_ASSERT(!stacks.empty()).");
    println!("    Recommendation: do not rely on grammar for expandQuery output");
    println!("    enforcement. Prompt clearly and parse output in Rust instead.");
    println!("  - Use model.chat_template(None) + model.apply_chat_template(...)");
    println!("    to format the prompt. AddBos::Never after the template (template");
    println!("    embeds BOS already).");
    println!("  - chain_simple takes IntoIterator<Item = LlamaSampler> and moves them.");
    println!("  - Manual decode loop: clear() -> add(token, pos, &[0], true) -> decode.");
    println!("  - Termination: model.is_eog_token(token) OR a token budget.");
    println!("  - token_to_piece(token, &mut Decoder, special, lstrip) replaces the");
    println!("    deprecated token_to_str/token_to_bytes; needs a stateful UTF-8 decoder.");
    Ok(())
}
