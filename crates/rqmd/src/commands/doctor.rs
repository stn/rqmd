//! `rqmd doctor` — index / runtime / device diagnostics.
//!
//! Port of qmd's `showDoctor` (`src/cli/qmd.ts`, origin/main / v2.5.x). All
//! diagnostic logic lives in [`rqmd_core::RqmdStore::doctor_report`] /
//! [`rqmd_core::RqmdStore::adopt_legacy_embeddings`]; this file is purely the
//! qmd-parity formatter on top.

use std::collections::HashMap;
use std::io::IsTerminal;

use anyhow::{Context, Result};

use rqmd_core::llm::config::{DEFAULT_EMBED_MODEL, DEFAULT_GENERATE_MODEL, DEFAULT_RERANK_MODEL};
use rqmd_core::llm::device::LlamaBackendDeviceType;
use rqmd_core::{
    CachedModelEntry, DoctorReport, EnvOverride, FingerprintGroup, LegacyAdoptionOutcome,
    ModelsConfig, ResolvedModels, RqmdStore, VectorSampleCheck, VectorSampleStatus,
};

use crate::color::Palette;
use crate::format_helpers::format_bytes;
use crate::state::IndexState;

pub async fn run(state: &mut IndexState, p: &Palette) -> Result<()> {
    let options = state.rqmd_store_options()?;
    let mut store = RqmdStore::open(options).context("opening index for doctor")?;

    let report = store.doctor_report(None).await?;

    let mut next_steps: Vec<String> = Vec::new();
    print_doctor_report(&report, p, &mut next_steps);

    let adopt_outcome = if let Some(pending) = &report.legacy_pending {
        if pending.adoption_possible {
            Some(store.adopt_legacy_embeddings(None).await?)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(Some(outcome)) = adopt_outcome.as_ref() {
        print_legacy_adoption(outcome, p);
    }

    print_vector_sample(&report.vector_sample, p, &mut next_steps);

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

    store.close().await;
    Ok(())
}

// ============================================================================
// Formatter
// ============================================================================

fn print_doctor_report(r: &DoctorReport, p: &Palette, next_steps: &mut Vec<String>) {
    println!("{}rqmd Doctor{}\n", p.bold(), p.reset());
    println!("Index: {}", r.db_path.display());
    println!("Runtime: rusqlite (bundled SQLite)");

    check(p, "SQLite runtime", true, &r.sqlite_version);
    match &r.vec_version {
        Some(v) => check(p, "sqlite-vec", true, v),
        None => check(p, "sqlite-vec", false, "unavailable"),
    }

    if r.collection_count == 0 {
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
                format_count(r.collection_count as i64),
                if r.collection_count == 1 {
                    "collection"
                } else {
                    "collections"
                }
            ),
        );
    }

    print_environment_overrides(&r.env_overrides, p);
    print_model_defaults(&r.resolved_models, &r.configured_models, p);
    print_model_cache(&r.model_cache, p, next_steps);
    print_device(r, p, next_steps);

    // Embedding freshness (needs_embedding) — printed before the
    // fingerprint groups for qmd parity.
    let ok = r.needs_embedding == 0;
    let details = if ok {
        "all active documents match current fingerprint".to_string()
    } else {
        format!(
            "{} active documents need embeddings. Next: `rqmd embed`",
            format_count(r.needs_embedding)
        )
    };
    check(p, "embedding freshness", ok, &details);
    if r.needs_embedding > 0 {
        next_steps.push(format!(
            "Run `rqmd embed` to generate {} missing/stale document embeddings.",
            format_count(r.needs_embedding)
        ));
    }

    print_embedding_fingerprints(r, p, next_steps);
}

fn print_environment_overrides(overrides: &[EnvOverride], p: &Palette) {
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
    for o in overrides {
        println!("  - {}={}: {}", o.name, o.value, o.consequence);
    }
}

fn print_model_defaults(resolved: &ResolvedModels, configured: &Option<ModelsConfig>, p: &Palette) {
    let checks: [(&str, &str, &str, Option<&str>, &str); 3] = [
        (
            "embedding",
            rqmd_core::env_keys::EMBED_MODEL,
            &resolved.embed,
            model_config(configured, |m| &m.embed),
            DEFAULT_EMBED_MODEL,
        ),
        (
            "generation",
            rqmd_core::env_keys::GENERATE_MODEL,
            &resolved.generate,
            model_config(configured, |m| &m.generate),
            DEFAULT_GENERATE_MODEL,
        ),
        (
            "reranking",
            rqmd_core::env_keys::RERANK_MODEL,
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

fn print_model_cache(entries: &[CachedModelEntry], p: &Palette, next_steps: &mut Vec<String>) {
    let mut missing: Vec<String> = Vec::new();
    let mut cached: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    let unique_count = entries.len();
    for e in entries {
        let label = format!("{}: {}", roles_label(e), e.model_uri);
        for detail in &e.invalid {
            invalid.push(format!("{label} ({detail})"));
        }
        if e.path.is_some() {
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

fn roles_label(e: &CachedModelEntry) -> String {
    let mut roles = Vec::new();
    if e.used_for_embed {
        roles.push("embedding");
    }
    if e.used_for_generate {
        roles.push("generation");
    }
    if e.used_for_rerank {
        roles.push("reranking");
    }
    roles.join("+")
}

fn print_device(r: &DoctorReport, p: &Palette, next_steps: &mut Vec<String>) {
    check(p, "device mode", true, &r.device_mode);

    if r.device_probe_skipped {
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
        eprint!("\r{}\r", " ".repeat(crash_hint.chars().count()));
    }

    if let Some(err) = &r.device_probe_error {
        check(
            p,
            "device probe",
            false,
            &format!(
                "probe failed: {err}. Next: run with RQMD_FORCE_CPU=1 to bypass GPU probing, or set RQMD_LLAMA_GPU=metal|cuda|vulkan and retry"
            ),
        );
        next_steps.push(
            "GPU probe failed; try `RQMD_FORCE_CPU=1 rqmd doctor` to confirm CPU fallback, then fix GPU drivers/backend if acceleration is expected."
                .into(),
        );
        return;
    }

    let gpu_devices: Vec<_> = r
        .devices
        .iter()
        .filter(|d| !matches!(d.device_type, LlamaBackendDeviceType::Cpu))
        .collect();

    if gpu_devices.is_empty() {
        check(
            p,
            "device probe",
            false,
            &format!(
                "running on CPU ({} math cores). Next: install/configure Metal, CUDA, or Vulkan for faster embeddings, or set RQMD_FORCE_CPU=1 to make CPU mode explicit",
                r.cpu_cores
            ),
        );
        next_steps.push(
            "Vector operations are running on CPU; install/configure Metal, CUDA, or Vulkan if embedding/query performance is too slow."
                .into(),
        );
        return;
    }

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
    parts.push(format!("{} CPU math cores", r.cpu_cores));

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

fn print_embedding_fingerprints(r: &DoctorReport, p: &Palette, next_steps: &mut Vec<String>) {
    let rows: &[FingerprintGroup] = &r.fingerprint_groups;
    let model = &r.active_embed_model;
    let fingerprint = &r.active_embed_fingerprint;

    let unique_fps: std::collections::HashSet<&str> =
        rows.iter().map(|r| r.fingerprint.as_str()).collect();
    let off_current = rows
        .iter()
        .any(|r| &r.model == model && &r.fingerprint != fingerprint);
    let ok = rows.is_empty()
        || (unique_fps.len() == 1
            && rows
                .first()
                .map(|r| &r.fingerprint == fingerprint)
                .unwrap_or(false)
            && !off_current);
    let current_docs: i64 = rows
        .iter()
        .filter(|r| &r.model == model && &r.fingerprint == fingerprint)
        .map(|r| r.docs)
        .sum();
    let other_docs: i64 = rows.iter().map(|r| r.docs).sum::<i64>() - current_docs;
    let groups = rows
        .iter()
        .map(|r| {
            let label = if &r.fingerprint == fingerprint {
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

    let named: Vec<_> = rows.iter().filter(|r| !r.fingerprint.is_empty()).collect();
    let named_fps: std::collections::HashSet<&str> =
        named.iter().map(|r| r.fingerprint.as_str()).collect();
    if named_fps.len() > 1 {
        let named_groups = named
            .iter()
            .map(|r| {
                format!(
                    "{}{}: {} {} docs/{} chunks",
                    r.fingerprint,
                    if &r.fingerprint == fingerprint {
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

fn print_legacy_adoption(o: &LegacyAdoptionOutcome, p: &Palette) {
    let details = if o.adopted {
        format!(
            "adopted {} legacy chunks; sample {} matched current fingerprint at distance {:.6}",
            o.adopted_rows,
            short_hash_seq(&o.sample_hash_seq),
            o.sample_distance,
        )
    } else {
        o.reason.clone()
    };
    check(p, "legacy fingerprint adoption", o.adopted, &details);
}

fn print_vector_sample(s: &VectorSampleCheck, p: &Palette, next_steps: &mut Vec<String>) {
    match &s.status {
        VectorSampleStatus::NoActiveDocuments => {
            check(
                p,
                "embedding vector sample",
                true,
                "no active documents indexed",
            );
        }
        VectorSampleStatus::NoVectorTable => {
            check(
                p,
                "embedding vector sample",
                false,
                "no vector table to test; please run rqmd embed again",
            );
        }
        VectorSampleStatus::ModelNotCached => {
            check(
                p,
                "embedding vector sample",
                false,
                "embed model not downloaded; cannot verify vectors. Next: `rqmd pull`",
            );
        }
        VectorSampleStatus::NoCurrentChunks => {
            check(
                p,
                "embedding vector sample",
                false,
                "no current embedded chunks to test; please run rqmd embed again",
            );
        }
        VectorSampleStatus::Sampled {
            sampled,
            passed: _,
            failures,
        } => {
            let total = *sampled;
            if !failures.is_empty() {
                let first = &failures[0];
                let details = format!(
                    "{}/{total} sampled chunks differ from stored vectors ({}: {}). Rebuild with `rqmd embed --force`",
                    failures.len(),
                    short_hash_seq(&first.hash_seq),
                    first.reason
                );
                check(p, "embedding vector sample", false, &details);
                next_steps.push(
                    "Run `rqmd embed --force` to rebuild existing vectors that no longer reproduce under the current embedding pipeline."
                        .into(),
                );
            } else {
                let details = format!(
                    "{total} sampled {} reproduce stored vectors",
                    if total == 1 { "chunk" } else { "chunks" }
                );
                check(p, "embedding vector sample", true, &details);
            }
        }
    }
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

fn normalized_doctor_next_steps(steps: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
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

fn is_force_cpu() -> bool {
    match std::env::var(rqmd_core::env_keys::FORCE_CPU) {
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

fn model_config<'a>(
    configured: &'a Option<ModelsConfig>,
    pick: impl Fn(&'a ModelsConfig) -> &'a Option<String>,
) -> Option<&'a str> {
    configured.as_ref().and_then(|m| pick(m).as_deref())
}
