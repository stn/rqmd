//! `rqmd doctor` — index / runtime / device diagnostics.
//!
//! Port of qmd's `showDoctor` (`src/cli/qmd.ts`, origin/main / v2.5.x). Output
//! wording mirrors qmd; tool name is rqmd (sanctioned branding) and all env
//! vars use the `RQMD_` prefix. Two checks (`legacy fingerprint adoption`,
//! `embedding vector sample`) load the embed model, so this command is async;
//! they are gated on the model being cached + a `vectors_vec` table existing so
//! a model-less / CI environment skips them instead of downloading.

use std::collections::{HashMap, HashSet};
use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use rqmd_core::Store;
use rqmd_core::collections::ModelsConfig;
use rqmd_core::env_keys;
use rqmd_core::llm::config::{
    DEFAULT_EMBED_MODEL, DEFAULT_GENERATE_MODEL, DEFAULT_RERANK_MODEL, ResolvedModels,
};
use rqmd_core::llm::format::{embedding_fingerprint, format_doc_for_embedding};
use rqmd_core::llm::llama_cpp::LlamaCpp;
use rqmd_core::llm::pull::inspect_cached_model;
use rqmd_core::llm::session::{LlmSession, LlmSessionOptions};
use rqmd_core::llm::types::EmbedOptions;
use rqmd_core::llm::{LlamaBackendDeviceType, probe_devices};
use rqmd_core::store::chunking::ChunkStrategy;
use rqmd_core::store::context::list_collections;
use rqmd_core::store::doctor as docsql;
use rqmd_core::store::documents::extract_title;
use rqmd_core::store::embeddings::{
    cosine_distance, get_hashes_needing_embedding, get_stored_embedding, nearest_vector,
    vec_table_exists,
};
use rqmd_core::store_ops::chunk_tokens::chunk_document_by_tokens;

use crate::color::Palette;
use crate::format_helpers::format_bytes;
use crate::state::IndexState;

/// Cosine-distance threshold for "reproduces the stored vector" (qmd parity).
const VECTOR_MATCH_THRESHOLD: f64 = 0.0001;
/// Wall-clock budget for the LLM-backed checks.
const LLM_SESSION_MAX: Duration = Duration::from_secs(600);

pub async fn run(state: &mut IndexState, p: &Palette) -> Result<()> {
    // Resolve everything that needs `&mut state` up front (owned values), so we
    // can hold the `&mut Store` borrow for the rest of the command. `model` /
    // `fingerprint` are computed once and shared by checks 10/11/12/13.
    let db_path = state.db_path()?;
    let resolved = state.resolved_model_uris()?;
    let configured: Option<ModelsConfig> = state.config_mut()?.data().models.clone();
    let model = resolved.embed.clone();
    let fingerprint = embedding_fingerprint(&model);
    // Cheap (no model load until first embed); fetched before the store borrow.
    let llm = state.llama_cpp()?;
    let store = state.store_mut()?;

    let mut next_steps: Vec<String> = Vec::new();

    println!("{}rqmd Doctor{}\n", p.bold(), p.reset());
    println!("Index: {}", db_path.display());
    println!("Runtime: rusqlite (bundled SQLite)");

    // --- SQLite runtime / sqlite-vec versions ------------------------------
    match store
        .with_connection(|c| c.query_row("SELECT sqlite_version()", [], |r| r.get::<_, String>(0)))
    {
        Ok(v) => check(p, "SQLite runtime", true, &v),
        Err(e) => check(p, "SQLite runtime", false, &e.to_string()),
    }
    match store
        .with_connection(|c| c.query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0)))
    {
        Ok(v) => check(p, "sqlite-vec", true, &v),
        Err(e) => check(p, "sqlite-vec", false, &e.to_string()),
    }

    // --- index config (collections) ----------------------------------------
    let collection_count = match store.with_connection(list_collections) {
        Ok(cols) => cols.len(),
        Err(e) => {
            check(p, "index config", false, &e.to_string());
            0
        }
    };
    if collection_count == 0 {
        check(
            p,
            "index config",
            false,
            "no collections configured. Next: `rqmd collection add .`",
        );
        next_steps.push(
            "Run `rqmd collection add . --name <name>` from the folder you want to index, or edit the index config manually."
                .into(),
        );
    } else {
        check(
            p,
            "index config",
            true,
            &format!(
                "{} {} configured",
                format_count(collection_count as i64),
                if collection_count == 1 {
                    "collection"
                } else {
                    "collections"
                }
            ),
        );
    }

    // --- environment overrides / model defaults / model cache --------------
    check_environment_overrides(p, &resolved, &configured);
    check_model_defaults(p, &resolved, &configured);
    check_model_cache(p, &resolved, &mut next_steps);

    // --- device mode + probe ----------------------------------------------
    check_device(p, &mut next_steps);

    // --- legacy fingerprint adoption (LLM) --------------------------------
    if let Some((ok, details)) =
        legacy_fingerprint_adoption(store, llm.clone(), &model, &fingerprint).await
    {
        check(p, "legacy fingerprint adoption", ok, &details);
    }

    // --- embedding freshness ----------------------------------------------
    match store.with_connection(|c| get_hashes_needing_embedding(c, None, &model, &fingerprint)) {
        Ok(pending) => {
            let ok = pending == 0;
            let details = if ok {
                "all active documents match current fingerprint".to_string()
            } else {
                format!(
                    "{} active documents need embeddings. Next: `rqmd embed`",
                    format_count(pending)
                )
            };
            check(p, "embedding freshness", ok, &details);
            if pending > 0 {
                next_steps.push(format!(
                    "Run `rqmd embed` to generate {} missing/stale document embeddings.",
                    format_count(pending)
                ));
            }
        }
        Err(e) => check(p, "embedding freshness", false, &e.to_string()),
    }

    // --- embedding fingerprints (+ mixed named) ---------------------------
    check_embedding_fingerprints(p, store, &model, &fingerprint, &mut next_steps);

    // --- embedding vector sample (LLM) ------------------------------------
    let (ok, details) = embedding_vector_sample(store, llm.clone(), &model, &fingerprint).await;
    check(p, "embedding vector sample", ok, &details);
    if !ok {
        next_steps.push(
            "Run `rqmd embed --force` to rebuild existing vectors that no longer reproduce under the current embedding pipeline."
                .into(),
        );
    }

    // --- recommended next steps -------------------------------------------
    let steps = normalized_doctor_next_steps(next_steps);
    if !steps.is_empty() {
        println!(
            "\n{}Recommended next step{}{}",
            p.bold(),
            if steps.len() == 1 { "" } else { "s" },
            p.reset()
        );
        for s in steps {
            println!("  - {s}");
        }
    }

    Ok(())
}

// ============================================================================
// Output helpers (mirror qmd doctorCheck / formatters)
// ============================================================================

fn check(p: &Palette, label: &str, ok: bool, details: &str) {
    let mark = if ok {
        format!("{}✓{}", p.green(), p.reset())
    } else {
        format!("{}⚠{}", p.yellow(), p.reset())
    };
    println!("{mark} {label}: {details}");
}

/// `1234567` -> `1,234,567` (qmd `formatCount`, en-US grouping).
fn format_count(n: i64) -> String {
    let neg = n < 0;
    let digits = n.unsigned_abs().to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if neg { format!("-{out}") } else { out }
}

/// qmd `shortModelName`: `hf:` URIs collapse to their filename; long plain
/// names are truncated to 56 chars.
fn short_model_name(model: &str) -> String {
    if model.starts_with("hf:") {
        return model.rsplit('/').next().unwrap_or(model).to_string();
    }
    if model.chars().count() > 56 {
        let head: String = model.chars().take(53).collect();
        format!("{head}...")
    } else {
        model.to_string()
    }
}

/// qmd `shortHashSeq`: `<hash12>_<seq>`.
fn short_hash_seq(hash_seq: &str) -> String {
    match hash_seq.rfind('_') {
        None => {
            if hash_seq.chars().count() > 18 {
                let head: String = hash_seq.chars().take(18).collect();
                format!("{head}...")
            } else {
                hash_seq.to_string()
            }
        }
        Some(idx) => {
            let head = &hash_seq[..hash_seq.len().min(12)];
            format!("{head}_{}", &hash_seq[idx + 1..])
        }
    }
}

/// qmd `summarizeDeviceNames`: collapse duplicates to `N× name`.
fn summarize_device_names(names: &[String]) -> String {
    let mut order: Vec<&String> = Vec::new();
    let mut counts: HashMap<&String, usize> = HashMap::new();
    for n in names {
        if !counts.contains_key(n) {
            order.push(n);
        }
        *counts.entry(n).or_insert(0) += 1;
    }
    order
        .into_iter()
        .map(|n| {
            let c = counts[n];
            if c > 1 {
                format!("{c}× {n}")
            } else {
                n.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// qmd `normalizedDoctorNextSteps`: dedup, and when a `rqmd embed --force` step
/// exists, drop the milder `rqmd embed` steps it subsumes.
fn normalized_doctor_next_steps(steps: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let unique: Vec<String> = steps
        .into_iter()
        .filter(|s| seen.insert(s.clone()))
        .collect();
    let has_force = unique.iter().any(|s| s.contains("rqmd embed --force"));
    if !has_force {
        return unique;
    }
    unique
        .into_iter()
        .filter(|s| !s.contains("rqmd embed") || s.starts_with("Run `rqmd embed --force`"))
        .collect()
}

/// Configured device mode (moved from `status.rs`). Mirrors qmd
/// `configuredGpuModeLabel`: `CPU forced` when `RQMD_FORCE_CPU` is truthy
/// (rqmd's `--no-gpu` sets it), else explicit `RQMD_LLAMA_GPU`, else `auto`.
fn device_mode() -> String {
    if is_force_cpu() {
        return "CPU forced (RQMD_FORCE_CPU)".to_string();
    }
    if let Ok(v) = std::env::var(env_keys::LLAMA_GPU) {
        let t = v.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    "auto".to_string()
}

fn is_force_cpu() -> bool {
    match std::env::var(env_keys::FORCE_CPU) {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !t.is_empty()
                && !matches!(
                    t.as_str(),
                    "false" | "off" | "none" | "disable" | "disabled" | "0"
                )
        }
        Err(_) => false,
    }
}

/// qmd `envValueForDisplay`: truncate to 96 chars.
fn env_value_for_display(value: &str) -> String {
    if value.chars().count() > 96 {
        let head: String = value.chars().take(93).collect();
        format!("{head}...")
    } else {
        value.to_string()
    }
}

// ============================================================================
// Checks
// ============================================================================

fn check_environment_overrides(
    p: &Palette,
    resolved: &ResolvedModels,
    configured: &Option<ModelsConfig>,
) {
    let overrides = collect_environment_overrides(resolved, configured);
    if overrides.is_empty() {
        check(p, "environment overrides", true, "none");
        return;
    }
    check(
        p,
        "environment overrides",
        false,
        &format!("{} set", overrides.len()),
    );
    for (name_value, consequence) in &overrides {
        println!("  - {name_value}: {consequence}");
    }
}

/// `(name=value, consequence)` for every override env var rqmd actually reads.
fn collect_environment_overrides(
    resolved: &ResolvedModels,
    configured: &Option<ModelsConfig>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();

    macro_rules! add {
        ($name:expr, $consequence:expr) => {
            if let Ok(raw) = std::env::var($name) {
                let raw = raw.trim();
                if !raw.is_empty() {
                    out.push((
                        format!("{}={}", $name, env_value_for_display(raw)),
                        ($consequence).to_string(),
                    ));
                }
            }
        };
    }

    add!(
        "RQMD_INDEX_PATH",
        "overrides the SQLite index path; rqmd reads/writes a different database"
    );
    add!(
        "RQMD_CONFIG_DIR",
        "overrides the rqmd config directory and takes precedence over XDG_CONFIG_HOME"
    );
    add!(
        "RQMD_CACHE_DIR",
        "overrides the rqmd cache directory (index cache and model cache)"
    );
    add!(
        "XDG_CONFIG_HOME",
        "moves rqmd config to $XDG_CONFIG_HOME/qmd when RQMD_CONFIG_DIR is not set"
    );
    add!(
        "XDG_CACHE_HOME",
        "moves the default index cache and model cache"
    );

    push_model_override(
        &mut out,
        env_keys::EMBED_MODEL,
        "embed",
        &resolved.embed,
        model_config(configured, |m| &m.embed),
    );
    push_model_override(
        &mut out,
        env_keys::GENERATE_MODEL,
        "generate",
        &resolved.generate,
        model_config(configured, |m| &m.generate),
    );
    push_model_override(
        &mut out,
        env_keys::RERANK_MODEL,
        "rerank",
        &resolved.rerank,
        model_config(configured, |m| &m.rerank),
    );

    add!(
        env_keys::FORCE_CPU,
        "forces llama.cpp to bypass GPU backends; embeddings/query will be slower but GPU crashes are avoided"
    );
    add!(
        env_keys::LLAMA_GPU,
        "selects llama.cpp GPU backend (metal/cuda/vulkan) or disables GPU when set to false/off/0"
    );
    add!(
        env_keys::DOCTOR_DEVICE_PROBE,
        "controls rqmd doctor native device probing; 0/off skips GPU probing"
    );
    add!(
        env_keys::EMBED_PARALLELISM,
        "overrides embedding parallel context count; too high can exhaust RAM/VRAM"
    );
    add!(
        env_keys::RERANK_PARALLELISM,
        "overrides reranker parallel context count; too high can exhaust RAM/VRAM"
    );
    add!(
        env_keys::EXPAND_CONTEXT_SIZE,
        "overrides query expansion context size; larger values use more memory"
    );
    add!(
        env_keys::RERANK_CONTEXT_SIZE,
        "overrides reranker context size; larger values use more memory"
    );
    add!(
        env_keys::EMBED_CONTEXT_SIZE,
        "overrides embed context size; larger values use more memory"
    );
    add!(
        "RQMD_EDITOR_URI",
        "overrides clickable editor link template in terminal output"
    );
    add!(
        "RQMD_SKILLS_DIR",
        "overrides where rqmd skills are discovered from"
    );
    add!("NO_COLOR", "disables colored terminal output");
    add!(
        "CI",
        "disables real LLM operations inside rqmd's LlamaCpp wrapper"
    );
    add!(
        "HF_ENDPOINT",
        "changes Hugging Face download endpoint used when pulling models"
    );
    add!("WSL_DISTRO_NAME", "enables WSL path handling heuristics");
    add!("WSL_INTEROP", "enables WSL path handling heuristics");

    out
}

fn model_config<'a>(
    configured: &'a Option<ModelsConfig>,
    pick: impl Fn(&'a ModelsConfig) -> &'a Option<String>,
) -> Option<&'a str> {
    configured.as_ref().and_then(|m| pick(m).as_deref())
}

fn push_model_override(
    out: &mut Vec<(String, String)>,
    name: &str,
    key: &str,
    active: &str,
    configured: Option<&str>,
) {
    let Ok(raw) = std::env::var(name) else {
        return;
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return;
    }
    let consequence = match configured {
        Some(c) if c != raw => {
            format!("set but ignored because index models.{key} is configured as {c}")
        }
        _ => format!(
            "sets the active {key} model to {active}; changes embedding/search semantics and may require `rqmd pull` plus `rqmd embed`"
        ),
    };
    out.push((
        format!("{name}={}", env_value_for_display(raw)),
        consequence,
    ));
}

fn check_model_defaults(p: &Palette, resolved: &ResolvedModels, configured: &Option<ModelsConfig>) {
    let checks: [(&str, &str, &str, Option<&str>, &str); 3] = [
        (
            "embedding",
            env_keys::EMBED_MODEL,
            &resolved.embed,
            model_config(configured, |m| &m.embed),
            DEFAULT_EMBED_MODEL,
        ),
        (
            "generation",
            env_keys::GENERATE_MODEL,
            &resolved.generate,
            model_config(configured, |m| &m.generate),
            DEFAULT_GENERATE_MODEL,
        ),
        (
            "reranking",
            env_keys::RERANK_MODEL,
            &resolved.rerank,
            model_config(configured, |m| &m.rerank),
            DEFAULT_RERANK_MODEL,
        ),
    ];

    let mut notes: Vec<String> = Vec::new();
    for (role, env_name, active, configured_val, default_model) in checks {
        let env_value = std::env::var(env_name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        if let Some(ev) = &env_value
            && active == ev
        {
            notes.push(format!(
                "{role}: env {env_name}={active} (default {default_model}; might be ok)"
            ));
            continue;
        }
        if let Some(cv) = configured_val
            && cv != default_model
        {
            notes.push(format!(
                "{role}: index {cv} (default {default_model}; might be ok)"
            ));
            continue;
        }
        if let Some(ev) = &env_value
            && active != ev
        {
            notes.push(format!(
                "{role}: {env_name} is set to {ev} but index config uses {active}"
            ));
        }
    }

    if notes.is_empty() {
        check(p, "model defaults", true, "using rqmd codebase defaults");
    } else {
        check(
            p,
            "model defaults",
            false,
            &format!("non-default model configuration: {}", notes.join("; ")),
        );
    }
}

fn check_model_cache(p: &Palette, resolved: &ResolvedModels, next_steps: &mut Vec<String>) {
    let roles: [(&str, &str); 3] = [
        ("embedding", &resolved.embed),
        ("generation", &resolved.generate),
        ("reranking", &resolved.rerank),
    ];

    // Dedup by model URI, preserving first-seen order, collecting roles.
    let mut order: Vec<String> = Vec::new();
    let mut roles_by: HashMap<String, Vec<&str>> = HashMap::new();
    for (role, uri) in roles {
        if !roles_by.contains_key(uri) {
            order.push(uri.to_string());
        }
        roles_by.entry(uri.to_string()).or_default().push(role);
    }
    let unique_count = order.len();

    let mut missing: Vec<String> = Vec::new();
    let mut cached: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    for uri in &order {
        let label = format!("{}: {}", roles_by[uri].join("+"), uri);
        let inspection = inspect_cached_model(uri);
        for detail in &inspection.invalid {
            invalid.push(format!("{label} ({detail})"));
        }
        if inspection.path.is_some() {
            cached.push(label);
        } else {
            missing.push(label);
        }
    }

    if missing.is_empty() && invalid.is_empty() {
        check(
            p,
            "model cache",
            true,
            &format!(
                "{} active {} downloaded and valid GGUF",
                cached.len(),
                if cached.len() == 1 {
                    "model is"
                } else {
                    "models are"
                }
            ),
        );
        return;
    }

    let mut parts: Vec<String> = Vec::new();
    if !invalid.is_empty() {
        parts.push(format!("invalid {}: {}", invalid.len(), invalid.join("; ")));
    }
    if !missing.is_empty() {
        parts.push(format!(
            "missing {}/{}: {}",
            missing.len(),
            unique_count,
            missing.join("; ")
        ));
    }
    let next = if !invalid.is_empty() {
        "Next: run `rqmd pull --refresh` (or remove the bad cached file)"
    } else {
        "Next: run `rqmd pull`"
    };
    check(
        p,
        "model cache",
        false,
        &format!("{}. {next}", parts.join("; ")),
    );
    if !invalid.is_empty() {
        next_steps.push(
            "Run `rqmd pull --refresh` to replace invalid cached model files, or delete the listed file and rerun `rqmd pull`."
                .into(),
        );
    } else {
        next_steps.push(
            "Run `rqmd pull` to download missing embedding/generation/reranking models before `rqmd embed` or `rqmd query`."
                .into(),
        );
    }
}

fn check_device(p: &Palette, next_steps: &mut Vec<String>) {
    check(p, "device mode", true, &device_mode());

    let skip = matches!(
        std::env::var(env_keys::DOCTOR_DEVICE_PROBE)
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off" | "no" | "skip")
    );
    if skip {
        check(
            p,
            "device probe",
            false,
            "skipped by RQMD_DOCTOR_DEVICE_PROBE=0. Next: unset it and rerun `rqmd doctor` to verify GPU/CPU acceleration",
        );
        next_steps.push(
            "Unset `RQMD_DOCTOR_DEVICE_PROBE` and rerun `rqmd doctor` when you want to verify llama.cpp device acceleration."
                .into(),
        );
        return;
    }

    let crash_hint = "Probing native llama backend now. If rqmd crashes here, rerun with `RQMD_FORCE_CPU=1 rqmd doctor` (or `RQMD_DOCTOR_DEVICE_PROBE=0 rqmd doctor` to skip this probe).";
    let is_tty = std::io::stderr().is_terminal();
    if is_tty {
        eprint!("{}{}{}", p.dim(), crash_hint, p.reset());
    }

    let probed = probe_devices();
    if is_tty {
        eprint!("\r{}\r", " ".repeat(crash_hint.chars().count()));
    }

    let devices = match probed {
        Ok(d) => d,
        Err(e) => {
            check(
                p,
                "device probe",
                false,
                &format!(
                    "probe failed: {e}. Next: run with RQMD_FORCE_CPU=1 to bypass GPU probing, or set RQMD_LLAMA_GPU=metal|cuda|vulkan and retry"
                ),
            );
            next_steps.push(
                "GPU probe failed; try `RQMD_FORCE_CPU=1 rqmd doctor` to confirm CPU fallback, then fix GPU drivers/backend if acceleration is expected."
                    .into(),
            );
            return;
        }
    };

    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let gpu_devices: Vec<_> = devices
        .iter()
        .filter(|d| !matches!(d.device_type, LlamaBackendDeviceType::Cpu))
        .collect();

    if gpu_devices.is_empty() {
        check(
            p,
            "device probe",
            false,
            &format!(
                "running on CPU ({cpu_cores} math cores). Next: install/configure Metal, CUDA, or Vulkan for faster embeddings, or set RQMD_FORCE_CPU=1 to make CPU mode explicit"
            ),
        );
        next_steps.push(
            "Vector operations are running on CPU; install/configure Metal, CUDA, or Vulkan if embedding/query performance is too slow."
                .into(),
        );
        return;
    }

    // `gpuOffloading` is approximated as "a GPU device exists and CPU is not
    // forced" (qmd's exact offloading probe is not exposed here).
    let offloading = !is_force_cpu();
    let backend = gpu_devices[0].backend.clone();
    let names: Vec<String> = gpu_devices
        .iter()
        .map(|d| {
            if d.description.is_empty() {
                d.name.clone()
            } else {
                d.description.clone()
            }
        })
        .collect();
    let free: usize = gpu_devices.iter().map(|d| d.memory_free).sum();
    let total: usize = gpu_devices.iter().map(|d| d.memory_total).sum();

    let mut parts = vec![
        format!("GPU {backend}"),
        format!(
            "offloading {}",
            if offloading { "enabled" } else { "disabled" }
        ),
        format!("devices: {}", summarize_device_names(&names)),
    ];
    if total > 0 {
        parts.push(format!(
            "VRAM {} free / {} total",
            format_bytes(free as u64),
            format_bytes(total as u64)
        ));
    }
    parts.push(format!("{cpu_cores} CPU math cores"));

    if offloading {
        check(p, "device probe", true, &parts.join("; "));
    } else {
        check(
            p,
            "device probe",
            false,
            &format!(
                "{}. Next: check RQMD_LLAMA_GPU and llama.cpp backend support",
                parts.join("; ")
            ),
        );
        next_steps.push(
            "GPU was detected but offloading is disabled; check `RQMD_LLAMA_GPU=metal|cuda|vulkan` and rerun `rqmd doctor`."
                .into(),
        );
    }
}

fn check_embedding_fingerprints(
    p: &Palette,
    store: &mut Store,
    model: &str,
    fingerprint: &str,
    next_steps: &mut Vec<String>,
) {
    let rows = match store.with_connection(docsql::fingerprint_groups) {
        Ok(r) => r,
        Err(e) => {
            check(p, "embedding fingerprints", false, &e.to_string());
            return;
        }
    };

    let unique_fps: HashSet<&str> = rows.iter().map(|r| r.fingerprint.as_str()).collect();
    let off_current = rows
        .iter()
        .any(|r| r.model == model && r.fingerprint != fingerprint);
    let ok = rows.is_empty()
        || (unique_fps.len() == 1
            && rows
                .first()
                .map(|r| r.fingerprint == fingerprint)
                .unwrap_or(false)
            && !off_current);
    let current_docs: i64 = rows
        .iter()
        .filter(|r| r.model == model && r.fingerprint == fingerprint)
        .map(|r| r.docs)
        .sum();
    let other_docs: i64 = rows.iter().map(|r| r.docs).sum::<i64>() - current_docs;
    let groups = rows
        .iter()
        .map(|r| {
            let label = if r.fingerprint == fingerprint {
                "current".to_string()
            } else if r.fingerprint.is_empty() {
                "legacy".to_string()
            } else {
                r.fingerprint.clone()
            };
            format!(
                "{}:{label} {} docs/{} chunks",
                short_model_name(&r.model),
                format_count(r.docs),
                format_count(r.chunks)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    // Mixed *named* fingerprints (legacy empty strings excluded): a hard warn.
    let named: Vec<_> = rows.iter().filter(|r| !r.fingerprint.is_empty()).collect();
    let named_fps: HashSet<&str> = named.iter().map(|r| r.fingerprint.as_str()).collect();
    if named_fps.len() > 1 {
        let named_groups = named
            .iter()
            .map(|r| {
                format!(
                    "{}{}: {} {} docs/{} chunks",
                    r.fingerprint,
                    if r.fingerprint == fingerprint {
                        " (current)"
                    } else {
                        ""
                    },
                    short_model_name(&r.model),
                    format_count(r.docs),
                    format_count(r.chunks)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        check(
            p,
            "mixed named embedding fingerprints",
            false,
            &format!(
                "content_vectors contains {} named fingerprints: {named_groups}. Next: `rqmd embed` or `rqmd embed --force`",
                named_fps.len()
            ),
        );
        next_steps.push(
            "Run `rqmd embed` to converge mixed named embedding fingerprints; use `rqmd embed --force` if old named fingerprints or vector sample mismatches remain."
                .into(),
        );
    }

    let details = if rows.is_empty() {
        format!("no vectors yet; current fingerprint {fingerprint}")
    } else if ok {
        format!(
            "{} docs on current fingerprint ({fingerprint})",
            format_count(current_docs)
        )
    } else {
        format!(
            "{} docs current, {} docs legacy/stale. {groups}. Next: `rqmd embed`",
            format_count(current_docs),
            format_count(other_docs)
        )
    };
    check(p, "embedding fingerprints", ok, &details);
    if !ok {
        next_steps.push(
            "Run `rqmd embed` to migrate active documents to the current embedding fingerprint; use `rqmd embed --force` if vector samples still fail afterward."
                .into(),
        );
    }
}

// ============================================================================
// LLM-backed checks (gated on cached model + vectors_vec)
// ============================================================================

/// qmd `maybeAdoptLegacyEmbeddingFingerprint`. Returns `None` (print nothing)
/// when there is nothing to check (no legacy rows / no vector table / no active
/// sample), mirroring qmd which only prints when it actually ran or adopted.
async fn legacy_fingerprint_adoption(
    store: &mut Store,
    llm: Arc<LlamaCpp>,
    model: &str,
    fingerprint: &str,
) -> Option<(bool, String)> {
    let legacy = store
        .with_connection(|c| docsql::count_legacy_distinct_hashes(c, model))
        .ok()?;
    if legacy == 0 {
        return None;
    }
    if !store.with_connection(vec_table_exists).unwrap_or(false) {
        return None;
    }
    // Avoid loading / downloading a model in a model-less or CI environment.
    // Suppress (return None) rather than print a redundant nudge: the `model
    // cache` check already reports the missing model and pushes the `rqmd pull`
    // next-step, so it stays the single source of truth for that advice.
    inspect_cached_model(model).path.as_ref()?;
    let sample = store
        .with_connection(|c| docsql::sample_legacy_chunk(c, model))
        .ok()
        .flatten()?;

    let expected = format!("{}_{}", sample.hash, sample.seq);
    let title = extract_title(&sample.body, &sample.path);
    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            max_duration: Some(LLM_SESSION_MAX),
            name: Some("doctorLegacyAdoption".into()),
        },
    );

    let result = async {
        let chunks = chunk_document_by_tokens(
            session.clone(),
            &sample.body,
            None,
            None,
            None,
            Some(&sample.path),
            ChunkStrategy::Auto,
            Some(session.signal()),
        )
        .await
        .ok()?;
        let chunk = chunks.get(sample.seq as usize)?;
        let formatted = format_doc_for_embedding(&chunk.text, Some(&title), model);
        let embedding = session
            .embed(
                &formatted,
                EmbedOptions {
                    model: Some(model.to_string()),
                    is_query: false,
                    title: None,
                },
            )
            .await
            .ok()??;
        Some(embedding.embedding)
    }
    .await;
    session.release();

    let embedding = match result {
        Some(e) => e,
        None => return Some((false, "failed to embed legacy sample".into())),
    };

    let nearest = store
        .with_connection(|c| nearest_vector(c, &embedding))
        .ok()
        .flatten();
    let Some((hash_seq, distance)) = nearest else {
        return Some((false, "legacy sample vector not found".into()));
    };
    if hash_seq != expected || distance > VECTOR_MATCH_THRESHOLD {
        return Some((
            false,
            format!(
                "legacy sample differs from current fingerprint (nearest {}, distance {distance:.6})",
                short_hash_seq(&hash_seq)
            ),
        ));
    }

    let adopted = store
        .with_connection_mut(|c| docsql::adopt_legacy_fingerprint(c, model, fingerprint))
        .unwrap_or(0);
    let reason = format!(
        "sample {} matched current fingerprint at distance {distance:.6}",
        short_hash_seq(&expected)
    );
    if adopted > 0 {
        Some((true, format!("adopted {adopted} legacy chunks; {reason}")))
    } else {
        Some((false, reason))
    }
}

/// qmd `checkEmbeddingVectorSamples`: re-chunk + re-embed up to 3 random
/// current-fingerprint chunks and compare to the stored vectors.
async fn embedding_vector_sample(
    store: &mut Store,
    llm: Arc<LlamaCpp>,
    model: &str,
    fingerprint: &str,
) -> (bool, String) {
    let active = store
        .with_connection(docsql::count_active_documents)
        .unwrap_or(0);
    if active == 0 {
        return (true, "no active documents indexed".into());
    }
    if !store.with_connection(vec_table_exists).unwrap_or(false) {
        return (
            false,
            "no vector table to test; please run rqmd embed again".into(),
        );
    }
    if inspect_cached_model(model).path.is_none() {
        return (
            false,
            "embed model not downloaded; cannot verify vectors. Next: `rqmd pull`".into(),
        );
    }
    let samples = store
        .with_connection(|c| docsql::sample_current_chunks(c, model, fingerprint, 3))
        .unwrap_or_default();
    if samples.is_empty() {
        return (
            false,
            "no current embedded chunks to test; please run rqmd embed again".into(),
        );
    }

    let session = LlmSession::new(
        llm,
        LlmSessionOptions {
            max_duration: Some(LLM_SESSION_MAX),
            name: Some("doctorEmbeddingVectorSample".into()),
        },
    );

    let total = samples.len();
    let mut mismatches: Vec<String> = Vec::new();
    for sample in samples {
        let hash_seq = format!("{}_{}", sample.hash, sample.seq);
        let short = short_hash_seq(&hash_seq);

        let chunks = match chunk_document_by_tokens(
            session.clone(),
            &sample.body,
            None,
            None,
            None,
            Some(&sample.path),
            ChunkStrategy::Auto,
            Some(session.signal()),
        )
        .await
        {
            Ok(c) => c,
            Err(_) => {
                mismatches.push(format!("{short}: chunk no longer exists"));
                continue;
            }
        };
        let Some(chunk) = chunks.get(sample.seq as usize) else {
            mismatches.push(format!("{short}: chunk no longer exists"));
            continue;
        };
        let title = extract_title(&sample.body, &sample.path);
        let formatted = format_doc_for_embedding(&chunk.text, Some(&title), model);
        let embedding = match session
            .embed(
                &formatted,
                EmbedOptions {
                    model: Some(model.to_string()),
                    is_query: false,
                    title: None,
                },
            )
            .await
        {
            Ok(Some(r)) => r.embedding,
            _ => {
                mismatches.push(format!("{short}: embedding failed"));
                continue;
            }
        };
        let stored = store
            .with_connection(|c| get_stored_embedding(c, &hash_seq))
            .ok()
            .flatten();
        let Some(stored) = stored else {
            mismatches.push(format!("{short}: stored vector missing"));
            continue;
        };
        let distance = cosine_distance(&embedding, &stored);
        if distance > VECTOR_MATCH_THRESHOLD {
            mismatches.push(format!("{short}: stored vector distance {distance:.6}"));
        }
    }
    session.release();

    if !mismatches.is_empty() {
        (
            false,
            format!(
                "{}/{total} sampled chunks differ from stored vectors ({}). Rebuild with `rqmd embed --force`",
                mismatches.len(),
                mismatches[0]
            ),
        )
    } else {
        (
            true,
            format!(
                "{total} sampled {} reproduce stored vectors",
                if total == 1 { "chunk" } else { "chunks" }
            ),
        )
    }
}
