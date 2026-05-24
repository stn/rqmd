//! E2E CLI integration tests — Rust port of qmd's `test/cli.test.ts`.
//!
//! Each test spawns the built `rqmd` binary against an isolated temp index +
//! config (see `common`), mirroring qmd's `runQmd()` process-spawn approach.
//!
//! Scope: a near-complete port of `cli.test.ts`. The editor-URI / `termLink`
//! helpers are implemented in `search_view.rs` and unit-tested there, so they
//! are not re-asserted here. Genuinely out of scope: qmd's MCP *stdio launcher*
//! test (a Node/bash wrapper-script concern with no rqmd analogue) and its two
//! absolute-path `ls` tests (on Windows the drive letter collides with `qmd://`
//! collection-name parsing, and a hand-written-YAML collection is not synced
//! into the DB registry `ls` reads). Deliberate behavioural divergences are
//! kept and tested as rqmd's *actual* behaviour (noted at each call site).
//!
//! Where rqmd's wording / exit codes diverge from qmd, the assertions follow
//! rqmd's actual output (clap parse errors exit 2; app errors exit 1; the
//! default search format uses bare `collection/path` while JSON uses `qmd://`).

mod common;

// ===========================================================================
// CLI Help
// ===========================================================================
mod cli_help {
    use crate::common::*;

    #[test]
    fn shows_help_with_help_flag() {
        let e = env();
        let out = e.run_bare(&["--help"]);
        out.assert_ok();
        assert!(out.stdout.contains("Usage:"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("collection"));
        assert!(out.stdout.contains("search"));
        assert!(out.stdout.contains("--no-gpu"));
        // (qmd's "qmd collection add" / "qmd skill show/install" lines have no
        // rqmd equivalent — clap renders a `Commands:` list instead.)
    }

    #[test]
    fn shows_usage_with_no_arguments() {
        let e = env();
        // clap requires a subcommand → parse error on stderr, exit code 2
        // (qmd printed usage to stdout with exit 1).
        let out = e.run_bare(&[]);
        out.assert_err();
        assert!(out.stderr.contains("Usage:"), "stderr: {}", out.stderr);
    }
}

// ===========================================================================
// CLI Init (`rqmd init` — project-local `.rqmd` index)
// ===========================================================================
mod cli_init {
    use crate::common::*;

    #[test]
    fn creates_local_index_with_empty_collections_and_seeded_models() {
        let e = env();
        let project = e.root.join("proj");
        std::fs::create_dir_all(&project).unwrap();

        let out = e.run_in_env(&project, &["init"], &[]);
        out.assert_ok();
        assert!(
            out.stdout.contains("ready to go with new local index"),
            "stdout: {}",
            out.stdout
        );

        let cfg = project.join(".rqmd").join("index.yml");
        let db = project.join(".rqmd").join("index.sqlite");
        assert!(cfg.is_file(), "config not created");
        assert!(db.is_file(), "sqlite not created");

        let yaml = std::fs::read_to_string(&cfg).unwrap();
        // Parity with qmd `init`: a fresh config serializes an empty collections
        // map. Guards against a future `skip_serializing_if` on the field that
        // would silently break drop-in parity.
        assert!(yaml.contains("collections: {}"), "yaml:\n{yaml}");
        // Models seeded (env > crate default): all three keys present.
        assert!(yaml.contains("embed:"), "yaml:\n{yaml}");
        assert!(yaml.contains("generate:"), "yaml:\n{yaml}");
        assert!(yaml.contains("rerank:"), "yaml:\n{yaml}");
    }

    #[test]
    fn seeds_models_from_env_override() {
        let e = env();
        let project = e.root.join("proj");
        std::fs::create_dir_all(&project).unwrap();

        let out = e.run_in_env(
            &project,
            &["init"],
            &[("QMD_EMBED_MODEL", "hf:example/custom-embed.gguf")],
        );
        out.assert_ok();

        let yaml = std::fs::read_to_string(project.join(".rqmd").join("index.yml")).unwrap();
        assert!(
            yaml.contains("hf:example/custom-embed.gguf"),
            "env override not seeded; yaml:\n{yaml}"
        );
    }

    #[test]
    fn refuses_to_initialize_in_home() {
        let e = env();
        let home = e.root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let home_str = home.to_str().unwrap();

        // cwd == HOME == USERPROFILE → the $HOME guard trips and nothing is made.
        let out = e.run_in_env(
            &home,
            &["init"],
            &[("HOME", home_str), ("USERPROFILE", home_str)],
        );
        out.assert_err();
        assert!(
            out.stderr
                .contains("Refusing to initialize a local index in $HOME"),
            "stderr: {}",
            out.stderr
        );
        assert!(
            !home.join(".rqmd").exists(),
            ".rqmd must not be created in $HOME"
        );
    }
}

// ===========================================================================
// CLI Skills (bundled runtime skill — `skills list/get/path`)
// ===========================================================================
mod cli_skills {
    use crate::common::*;

    #[test]
    fn lists_bundled_runtime_skills() {
        let e = env();
        let out = e.run_bare(&["skills", "list"]);
        out.assert_ok();
        assert_eq!(out.stderr, "", "stderr: {}", out.stderr);
        assert!(out.stdout.contains("rqmd"), "stdout: {}", out.stdout);
        assert!(
            out.stdout.contains("Search local markdown knowledge bases"),
            "stdout: {}",
            out.stdout
        );
    }

    #[test]
    fn gets_runtime_skill_content() {
        let e = env();
        let out = e.run_bare(&["skills", "get", "rqmd"]);
        out.assert_ok();
        // rqmd identity strings (sanctioned parity exception): em-dash H1.
        assert!(
            out.stdout.contains("# rqmd — Query Markdown Documents"),
            "stdout: {}",
            out.stdout
        );
        assert!(out.stdout.contains("## MCP Tool: `query`"));
        assert!(!out.stdout.contains("discovery stub"));
    }

    #[test]
    fn gets_runtime_skill_with_references() {
        let e = env();
        let out = e.run_bare(&["skills", "get", "rqmd", "--full"]);
        out.assert_ok();
        assert!(
            out.stdout.contains("--- references/mcp-setup.md ---"),
            "stdout: {}",
            out.stdout
        );
        assert!(out.stdout.contains("# rqmd MCP Server Setup"));
    }

    #[test]
    fn prints_canonical_skill_path() {
        let e = env();
        let out = e.run_bare(&["skills", "path", "rqmd"]);
        out.assert_ok();
        assert_eq!(out.stderr, "", "stderr: {}", out.stderr);
        let p = out.stdout.trim().replace('\\', "/");
        assert!(p.ends_with("skills/rqmd"), "path: {p}");
    }
}

// ===========================================================================
// CLI Skill (legacy `skill show/install` + `--skill` alias)
// ===========================================================================
mod cli_skill {
    use crate::common::*;

    #[test]
    fn shows_skill_with_skill_alias() {
        let e = env();
        let out = e.run_bare(&["--skill"]);
        out.assert_ok();
        assert!(out.stdout.contains("rqmd Skill"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("name: rqmd"));
        assert!(
            out.stdout
                .contains("allowed-tools: Bash(rqmd:*), mcp__rqmd__*")
        );
    }

    #[test]
    fn legacy_skill_show_prints_canonical_skill() {
        let e = env();
        let out = e.run_bare(&["skill", "show"]);
        out.assert_ok();
        assert!(out.stdout.contains("# rqmd — Query Markdown Documents"));
        assert!(out.stdout.contains("## MCP Tool: `query`"));
        assert!(!out.stdout.contains("discovery stub"));
    }

    #[test]
    fn shows_skill_help() {
        let e = env();
        // clap renders `skill -h` with a Commands list (show/install); qmd's
        // exact "Usage: qmd skill <show|install>" wording has no rqmd analogue.
        let out = e.run_bare(&["skill", "-h"]);
        out.assert_ok();
        assert!(out.stdout.contains("Usage:"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("install"));
        assert!(out.stdout.contains("show"));
    }

    #[test]
    fn installs_into_the_current_project() {
        let e = env();
        let proj = e.root.join("skill-project");
        std::fs::create_dir_all(&proj).unwrap();
        let out = e.run_in(&proj, &["skill", "install"]);
        out.assert_ok();
        assert!(
            out.stdout.contains("Installed rqmd skill"),
            "stdout: {}",
            out.stdout
        );
        let installed = proj
            .join(".agents")
            .join("skills")
            .join("rqmd")
            .join("SKILL.md");
        assert!(installed.is_file(), "missing {}", installed.display());
        let body = std::fs::read_to_string(&installed).unwrap();
        assert!(body.contains("# rqmd — Query Markdown Documents"));
    }

    #[test]
    fn refuses_to_overwrite_without_force() {
        let e = env();
        let proj = e.root.join("skill-project-force");
        std::fs::create_dir_all(&proj).unwrap();
        e.run_in(&proj, &["skill", "install"]).assert_ok();

        let second = e.run_in(&proj, &["skill", "install"]);
        second.assert_code(1);
        assert!(
            second.stderr.contains("Skill already exists"),
            "stderr: {}",
            second.stderr
        );
        assert!(
            second.stderr.contains("--force"),
            "stderr: {}",
            second.stderr
        );
    }

    // The `--global --yes` path creates a directory symlink, which on Windows
    // requires Developer Mode / admin; gate it to Unix (the project install
    // above exercises the cross-platform path). Mirrors the documented
    // platform constraint on the `ls` absolute-path tests.
    #[cfg(unix)]
    #[test]
    fn installs_globally_and_creates_claude_symlink_with_yes() {
        let e = env();
        let fake_home = e.root.join("skill-home");
        std::fs::create_dir_all(&fake_home).unwrap();
        let out = e.run_env(
            &["skill", "install", "--global", "--yes"],
            &[("HOME", fake_home.to_str().unwrap())],
        );
        out.assert_ok();
        assert!(
            out.stdout.contains("Linked Claude skill at"),
            "stdout: {}",
            out.stdout
        );

        let skill_dir = fake_home.join(".agents").join("skills").join("rqmd");
        assert!(skill_dir.join("SKILL.md").is_file());

        let link = fake_home.join(".claude").join("skills").join("rqmd");
        let meta = std::fs::symlink_metadata(&link).expect("claude link");
        assert!(
            meta.file_type().is_symlink(),
            "expected symlink at {}",
            link.display()
        );
        // Reading through the link resolves to the installed SKILL.md.
        let via_link = std::fs::read_to_string(link.join("SKILL.md")).unwrap();
        assert!(via_link.contains("# rqmd — Query Markdown Documents"));
    }
}

// ===========================================================================
// CLI Embed (flag validation only — model-free, fails before any load)
// ===========================================================================
mod cli_embed {
    use crate::common::*;

    #[test]
    fn rejects_invalid_max_docs_per_batch() {
        let e = env();
        let out = e.run(&["embed", "--max-docs-per-batch", "0"]);
        out.assert_err();
        assert!(
            out.stderr.contains("maxDocsPerBatch"),
            "stderr: {}",
            out.stderr
        );
    }

    #[test]
    fn rejects_invalid_max_batch_mb() {
        let e = env();
        let out = e.run(&["embed", "--max-batch-mb", "0"]);
        out.assert_err();
        assert!(
            out.stderr.contains("maxBatchBytes"),
            "stderr: {}",
            out.stderr
        );
    }
}

// ===========================================================================
// CLI Add Command
// ===========================================================================
mod cli_add {
    use crate::common::*;

    #[test]
    fn adds_files_from_current_directory() {
        let e = env();
        let out = e.run(&["collection", "add", "."]);
        out.assert_ok();
        // qmd asserts "Collection:" / "Indexed:"; rqmd prints "Creating
        // collection 'fixtures'..." + "Indexed: …".
        assert!(out.stdout.contains("Creating collection"));
        assert!(out.stdout.contains("Indexed:"));
    }

    #[test]
    fn adds_files_with_custom_glob_pattern() {
        let e = env();
        let out = e.run(&["collection", "add", ".", "--mask", "notes/*.md"]);
        out.assert_ok();
        assert!(out.stdout.contains("Indexed:"));
        // rqmd's add output does not echo the mask; verify via `collection list`.
        let list = e.run(&["collection", "list"]);
        list.assert_ok();
        assert!(list.stdout.contains("notes/*.md"), "list: {}", list.stdout);
    }

    #[test]
    fn can_recreate_collection_with_remove_and_add() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e.run(&["collection", "remove", "fixtures"]).assert_ok();
        let out = e.run(&["collection", "add", "."]);
        out.assert_ok();
        assert!(
            out.stdout
                .contains("Collection 'fixtures' created successfully"),
            "stdout: {}",
            out.stdout
        );
    }
}

// ===========================================================================
// CLI Status Command
// ===========================================================================
mod cli_status {
    use crate::common::*;

    #[test]
    fn shows_index_status() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        let out = e.run(&["status"]);
        out.assert_ok();
        assert!(out.stdout.contains("Collection"), "stdout: {}", out.stdout);
    }
    #[test]
    fn status_omits_device_section_doctor_owns_it() {
        // qmd v2.5.0 parity: GPU/device diagnostics moved out of `status` into
        // `doctor`. `status` must no longer print a Device section or mention
        // the retired QMD_STATUS_DEVICE_PROBE knob.
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        let out = e.run(&["status"]);
        out.assert_ok();
        assert!(!out.stdout.contains("Device"), "stdout: {}", out.stdout);
        assert!(
            !out.stdout.contains("QMD_STATUS_DEVICE_PROBE"),
            "stdout: {}",
            out.stdout
        );
        assert!(!out.stdout.contains("not probed"), "stdout: {}", out.stdout);
    }

    #[test]
    fn shows_mcp_daemon_section() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        let out = e.run(&["status"]);
        out.assert_ok();
        // qmd parity (qmd.ts:423-437): status reports MCP daemon health. No
        // daemon runs in this isolated env, so it reads "not running".
        assert!(out.stdout.contains("MCP"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("Daemon:"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("not running"), "stdout: {}", out.stdout);
    }
}

// ===========================================================================
// CLI Doctor Command (port of qmd's `qmd doctor` cli tests)
// ===========================================================================
mod cli_doctor {
    use crate::common::*;

    /// Run `rqmd doctor` hermetically: skip the native device probe and point
    /// the model cache at an empty dir so the model-gated LLM checks (legacy
    /// adoption / vector sample) skip instead of attempting a model load.
    fn doctor(e: &Env) -> Out {
        let cache = e.root.to_string_lossy().to_string();
        e.run_env(
            &["doctor"],
            &[
                ("QMD_DOCTOR_DEVICE_PROBE", "0"),
                ("XDG_CACHE_HOME", cache.as_str()),
            ],
        )
    }

    #[test]
    fn reports_core_index_health_checks() {
        let e = env();
        let out = doctor(&e);
        out.assert_ok();
        for needle in [
            "rqmd Doctor",
            "SQLite runtime",
            "sqlite-vec",
            "environment overrides",
            "model defaults",
            "model cache",
            "device mode",
            "device probe",
            "embedding freshness",
            "embedding fingerprints",
            "embedding vector sample",
        ] {
            assert!(
                out.stdout.contains(needle),
                "missing `{needle}`\n{}",
                out.stdout
            );
        }
        // device probe was disabled via the env knob.
        assert!(
            out.stdout.contains("skipped by QMD_DOCTOR_DEVICE_PROBE=0"),
            "stdout: {}",
            out.stdout
        );
    }

    #[test]
    fn warns_when_no_collections_configured() {
        let e = env();
        let out = doctor(&e);
        out.assert_ok();
        assert!(
            out.stdout.contains("no collections configured"),
            "stdout: {}",
            out.stdout
        );
        assert!(
            out.stdout.contains("rqmd collection add ."),
            "stdout: {}",
            out.stdout
        );
    }

    #[test]
    fn flags_invalid_gguf_in_model_cache() {
        let e = env();
        // Point the embed model at a local non-GGUF (HTML) file — the cleanest
        // way to exercise the model-cache check without seeding hf-hub's
        // snapshot layout (a plain path is inspected directly).
        let bad = e.root.join("bad-model.gguf");
        std::fs::write(&bad, "<!doctype html><html>blocked by proxy</html>").unwrap();
        let bad_str = bad.to_string_lossy().to_string();
        let cache = e.root.to_string_lossy().to_string();
        let out = e.run_env(
            &["doctor"],
            &[
                ("QMD_DOCTOR_DEVICE_PROBE", "0"),
                ("XDG_CACHE_HOME", cache.as_str()),
                ("QMD_EMBED_MODEL", bad_str.as_str()),
            ],
        );
        out.assert_ok();
        assert!(out.stdout.contains("model cache"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("invalid 1"), "stdout: {}", out.stdout);
        assert!(
            out.stdout.contains("HTML page, not a GGUF model"),
            "stdout: {}",
            out.stdout
        );
        assert!(
            out.stdout.contains("rqmd pull --refresh"),
            "stdout: {}",
            out.stdout
        );
    }

    #[test]
    fn flags_mixed_named_fingerprints() {
        let e = env();
        // Seed content_vectors with two distinct *named* fingerprints by writing
        // straight to the index DB (the mixed-fingerprint query reads
        // content_vectors alone, so no documents/embeddings are needed).
        {
            let mut store = rqmd_core::Store::open(&e.db).expect("open store");
            store.with_connection_mut(|c| {
                c.execute(
                    "INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
                     VALUES ('h1', 0, 0, 'm', 'aaaaaa', 1, 'ts')",
                    [],
                )
                .unwrap();
                c.execute(
                    "INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
                     VALUES ('h2', 0, 0, 'm', 'bbbbbb', 1, 'ts')",
                    [],
                )
                .unwrap();
            });
        }
        let out = doctor(&e);
        out.assert_ok();
        assert!(
            out.stdout.contains("mixed named embedding fingerprints"),
            "stdout: {}",
            out.stdout
        );
    }
}

// ===========================================================================
// CLI Search Command
// ===========================================================================
mod cli_search {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn searches_for_documents_with_bm25() {
        let e = seeded();
        let out = e.run(&["search", "meeting"]);
        out.assert_ok();
        assert!(out.stdout.to_lowercase().contains("meeting"));
    }

    #[test]
    fn searches_with_limit_option() {
        let e = seeded();
        e.run(&["search", "-n", "1", "test"]).assert_ok();
    }

    // qmd parity (arg-order fix): flags may follow the query. Before the fix the
    // trailing `-n 1` was swallowed into the query string. Here `meeting` reliably
    // matches notes/meeting.md, so a successful run with the limit applied proves
    // the flag was parsed rather than treated as a query token.
    #[test]
    fn searches_with_trailing_limit_option() {
        let e = seeded();
        let out = e.run(&["search", "meeting", "-n", "1"]);
        out.assert_ok();
        assert!(out.stdout.to_lowercase().contains("meeting"));
    }

    // A value-taking option immediately before the query: `-c` consumes `fixtures`
    // and `meeting` remains the query (not swallowed by `-c`).
    #[test]
    fn searches_with_collection_flag_before_query() {
        let e = seeded();
        let out = e.run(&["search", "-c", "fixtures", "meeting"]);
        out.assert_ok();
        assert!(out.stdout.to_lowercase().contains("meeting"));
    }

    // `--json` after the query must be parsed as a format flag, yielding a JSON
    // array — not folded into the query (which would print the default CLI format).
    #[test]
    fn searches_with_trailing_json_flag() {
        let e = seeded();
        let out = e.run(&["search", "meeting", "--json"]);
        out.assert_ok();
        let v: serde_json::Value = serde_json::from_str(&out.stdout).expect("valid json");
        assert!(v.is_array());
    }

    // A `-`-leading query token needs the `--` escape (otherwise clap rejects it
    // as an unknown flag). `xyznonexistent` won't match, so we only assert it runs.
    #[test]
    fn searches_hyphen_leading_query_after_double_dash() {
        let e = seeded();
        e.run(&["search", "--", "-xyznonexistent123"]).assert_ok();
    }

    #[test]
    fn searches_with_all_results_option() {
        let e = seeded();
        e.run(&["search", "--all", "the"]).assert_ok();
    }

    #[test]
    fn no_results_message_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123"]);
        out.assert_ok();
        // qmd parity: the cli "No results found." message goes to stdout.
        assert!(out.stdout.contains("No results"), "stdout: {}", out.stdout);
    }

    #[test]
    fn empty_json_array_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123", "--json"]);
        out.assert_ok();
        assert_eq!(out.stdout.trim(), "[]");
    }

    #[test]
    fn csv_header_only_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123", "--csv"]);
        out.assert_ok();
        assert_eq!(
            out.stdout.trim(),
            "docid,score,file,title,context,line,snippet"
        );
    }

    #[test]
    fn empty_xml_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123", "--xml"]);
        out.assert_ok();
        // rqmd emits no `<results>` wrapper for empty xml (qmd emitted
        // `<results></results>`).
        assert_eq!(out.stdout.trim(), "");
    }

    #[test]
    fn empty_md_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123", "--md"]);
        out.assert_ok();
        assert_eq!(out.stdout.trim(), "");
    }

    #[test]
    fn empty_files_for_non_matching_query() {
        let e = seeded();
        let out = e.run(&["search", "xyznonexistent123", "--files"]);
        out.assert_ok();
        assert_eq!(out.stdout.trim(), "");
    }

    #[test]
    fn min_score_filters_default_output() {
        let e = seeded();
        // Scores are normalised 0..1, so --min-score 2 filters everything.
        let out = e.run(&["search", "test", "--min-score", "2"]);
        out.assert_ok();
        // qmd parity: the cli "No results found." message goes to stdout.
        assert!(out.stdout.contains("No results"), "stdout: {}", out.stdout);
    }

    #[test]
    fn min_score_format_safe_empty_output() {
        let e = seeded();

        let json = e.run(&["search", "test", "--json", "--min-score", "2"]);
        json.assert_ok();
        assert_eq!(json.stdout.trim(), "[]");

        let csv = e.run(&["search", "test", "--csv", "--min-score", "2"]);
        csv.assert_ok();
        assert_eq!(
            csv.stdout.trim(),
            "docid,score,file,title,context,line,snippet"
        );

        let xml = e.run(&["search", "test", "--xml", "--min-score", "2"]);
        xml.assert_ok();
        assert_eq!(xml.stdout.trim(), "");

        let md = e.run(&["search", "test", "--md", "--min-score", "2"]);
        md.assert_ok();
        assert_eq!(md.stdout.trim(), "");

        let files = e.run(&["search", "test", "--files", "--min-score", "2"]);
        files.assert_ok();
        assert_eq!(files.stdout.trim(), "");
    }

    // Divergence (tested, not dropped): qmd's `search` errors on a missing
    // query argument (exit 1, "Usage:"); rqmd deliberately treats an empty
    // query as a no-match so `rqmd search "$VAR" --json` with an empty $VAR
    // yields `[]` rather than erroring (search.rs documents this).
    #[test]
    fn empty_query_is_no_match_not_error() {
        let e = seeded();
        let out = e.run(&["search"]);
        out.assert_ok();
        assert!(out.stdout.contains("No results"), "stdout: {}", out.stdout);

        let json = e.run(&["search", "--json"]);
        json.assert_ok();
        assert_eq!(json.stdout.trim(), "[]");
    }

    #[test]
    fn json_full_includes_line_field() {
        let e = seeded();
        let out = e.run(&["search", "meeting", "--json", "--full", "-n", "1"]);
        out.assert_ok();
        let v: serde_json::Value = serde_json::from_str(&out.stdout).expect("valid json");
        let arr = v.as_array().expect("array");
        assert!(!arr.is_empty(), "expected at least one result");
        assert!(arr[0]["line"].is_number());
        assert!(arr[0]["line"].as_u64().unwrap() > 0);
        assert!(arr[0]["body"].is_string());
    }
}

// ===========================================================================
// CLI Get Command
// ===========================================================================
mod cli_get {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn retrieves_document_content_by_path() {
        let e = seeded();
        let out = e.run(&["get", "README.md"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Project"));
    }

    #[test]
    fn retrieves_document_from_subdirectory() {
        let e = seeded();
        let out = e.run(&["get", "notes/meeting.md"]);
        out.assert_ok();
        assert!(out.stdout.contains("Team Meeting"));
    }

    #[test]
    fn handles_non_existent_file() {
        let e = seeded();
        let out = e.run(&["get", "nonexistent.md"]);
        out.assert_code(1);
    }

    // Divergence (tested, not dropped): qmd clamps a negative `--from` to the
    // top of the file; rqmd types `--from` as `Option<usize>`, so `--from -19`
    // is a clap parse error (exit 2) rather than a clamp.
    #[test]
    fn rejects_negative_from() {
        let e = seeded();
        let out = e.run(&["get", "README.md", "--from", "-19"]);
        out.assert_err();
    }

    // `--full` は qmd parity のための no-op。素の `get` と同一出力で、
    // スライス挙動も変えてはならない（将来 `full` をスライスに配線する誤りを防ぐ）。
    #[test]
    fn accepts_full_flag_as_noop_for_qmd_parity() {
        let e = seeded();
        let plain = e.run(&["get", "README.md"]);
        let full = e.run(&["get", "README.md", "--full"]);
        plain.assert_ok();
        full.assert_ok();
        assert_eq!(plain.stdout, full.stdout);
        assert!(full.stderr.is_empty());
    }
}

// ===========================================================================
// CLI Multi-Get Command
// ===========================================================================
mod cli_multi_get {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn retrieves_multiple_documents_by_pattern() {
        let e = seeded();
        let out = e.run(&["multi-get", "notes/*.md"]);
        out.assert_ok();
        assert!(out.stdout.contains("Meeting"));
        assert!(out.stdout.contains("Ideas"));
    }

    #[test]
    fn retrieves_documents_by_comma_separated_paths() {
        let e = seeded();
        let out = e.run(&["multi-get", "README.md,notes/meeting.md"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Project"));
        assert!(out.stdout.contains("Team Meeting"));
    }
}

// ===========================================================================
// CLI Update Command
// ===========================================================================
mod cli_update {
    use crate::common::*;

    #[test]
    fn updates_all_collections() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        let out = e.run(&["update"]);
        out.assert_ok();
        assert!(out.stdout.contains("Updating"));
    }

    #[test]
    fn deactivates_stale_docs_when_collection_has_zero_matching_files() {
        let e = env();
        let coll_dir = e.root.join("stale-coll");
        std::fs::create_dir_all(&coll_dir).unwrap();
        let doc = coll_dir.join("only.md");
        let token = "stale-proof-token";
        std::fs::write(
            &doc,
            format!("---\ndate: 2026-03-06\n---\n# Empty Collection Deactivation\n{token}\n"),
        )
        .unwrap();

        let abs = e.yaml_path(&coll_dir);
        e.run(&["collection", "add", &abs, "--name", "empty-check"])
            .assert_ok();

        let before = e.run(&["get", "qmd://empty-check/only.md"]);
        before.assert_ok();
        assert!(before.stdout.contains(token));

        std::fs::remove_file(&doc).unwrap();

        let update = e.run(&["update"]);
        update.assert_ok();
        assert!(
            update
                .stdout
                .contains("0 new, 0 updated, 0 unchanged, 1 removed"),
            "stdout: {}",
            update.stdout
        );

        let after = e.run(&["get", "qmd://empty-check/only.md"]);
        after.assert_code(1);
    }
}

// ===========================================================================
// CLI Add-Context Command
// ===========================================================================
mod cli_add_context {
    use crate::common::*;

    #[test]
    fn adds_context_to_a_path() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        let out = e.run(&[
            "context",
            "add",
            "qmd://fixtures/",
            "Personal notes and meeting logs",
        ]);
        out.assert_ok();
        assert!(out.stdout.contains("✓ Added context"));
    }

    #[test]
    fn requires_path_and_text_arguments() {
        let e = env();
        // clap: `args` is a required positional → parse error (exit 2) with usage.
        let out = e.run(&["context", "add"]);
        out.assert_err();
        assert!(out.stderr.contains("Usage:"), "stderr: {}", out.stderr);
    }
}

// ===========================================================================
// CLI Cleanup Command
// ===========================================================================
mod cli_cleanup {
    use crate::common::*;

    #[test]
    fn cleans_up_orphaned_entries() {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e.run(&["cleanup"]).assert_ok();
    }
}

// ===========================================================================
// CLI Error Handling
// ===========================================================================
mod cli_error_handling {
    use crate::common::*;

    #[test]
    fn handles_unknown_command() {
        let e = env();
        let out = e.run(&["unknowncommand"]);
        out.assert_err();
        // clap's wording is "unrecognized subcommand" (qmd: "Unknown command").
        assert!(
            out.stderr.contains("unrecognized subcommand"),
            "stderr: {}",
            out.stderr
        );
    }

    #[test]
    fn uses_rqmd_index_path_environment_variable() {
        let e = env();
        let custom = e.root.join("custom.sqlite");
        let out = e.run_env(
            &["collection", "add", "."],
            &[("RQMD_INDEX_PATH", custom.to_str().unwrap())],
        );
        out.assert_ok();
        assert!(custom.exists(), "expected {} to exist", custom.display());
    }
}

// ===========================================================================
// CLI Output Formats
// ===========================================================================
mod cli_output_formats {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn search_json_outputs_json_array() {
        let e = seeded();
        let out = e.run(&["search", "--json", "test"]);
        out.assert_ok();
        let v: serde_json::Value = serde_json::from_str(&out.stdout).expect("valid json");
        assert!(v.is_array());
    }

    #[test]
    fn search_files_outputs_file_paths() {
        let e = seeded();
        let out = e.run(&["search", "--files", "meeting"]);
        out.assert_ok();
        assert!(out.stdout.contains(".md"));
    }

    #[test]
    fn search_output_includes_snippets_by_default() {
        let e = seeded();
        let out = e.run(&["search", "API"]);
        out.assert_ok();
        // "API" reliably matches docs/api.md, so results are non-empty and the
        // default output names the path (containing "api"). "No results" would
        // go to stderr; asserting unconditionally here catches an empty-stdout
        // regression (qmd's original guarded this with `if !No results`).
        assert!(!out.stdout.trim().is_empty(), "stderr: {}", out.stderr);
        assert!(out.stdout.to_lowercase().contains("api"));
    }
}

// ===========================================================================
// CLI Search with Collection Filter
// ===========================================================================
mod cli_search_collection_filter {
    use crate::common::*;

    #[test]
    fn filters_search_by_collection_name() {
        let e = env();
        e.run(&[
            "collection",
            "add",
            ".",
            "--name",
            "notes",
            "--mask",
            "notes/*.md",
        ])
        .assert_ok();
        e.run(&[
            "collection",
            "add",
            ".",
            "--name",
            "docs",
            "--mask",
            "docs/*.md",
        ])
        .assert_ok();
        let out = e.run(&["search", "-c", "notes", "meeting"]);
        out.assert_ok();
    }
}

// ===========================================================================
// CLI Context Management
// ===========================================================================
mod cli_context_management {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn add_global_context() {
        let e = seeded();
        let out = e.run(&["context", "add", "/", "Global system context"]);
        out.assert_ok();
        assert!(out.stdout.contains("✓ Set global context"));
        assert!(out.stdout.contains("Global system context"));
    }

    #[test]
    fn list_contexts() {
        let e = seeded();
        e.run(&["context", "add", "/", "Test context"]).assert_ok();
        let out = e.run(&["context", "list"]);
        out.assert_ok();
        assert!(out.stdout.contains("Configured Contexts"));
        assert!(out.stdout.contains("Test context"));
    }

    #[test]
    fn add_context_to_virtual_path() {
        let e = seeded();
        let out = e.run(&[
            "context",
            "add",
            "qmd://fixtures/notes",
            "Context for notes subdirectory",
        ]);
        out.assert_ok();
        assert!(
            out.stdout
                .contains("✓ Added context for: qmd://fixtures/notes")
        );
    }

    #[test]
    fn remove_global_context() {
        let e = seeded();
        e.run(&["context", "add", "/", "Global context to remove"])
            .assert_ok();
        let out = e.run(&["context", "rm", "/"]);
        out.assert_ok();
        assert!(out.stdout.contains("✓ Removed"));
    }

    #[test]
    fn remove_virtual_path_context() {
        let e = seeded();
        e.run(&[
            "context",
            "add",
            "qmd://fixtures/notes",
            "Context to remove",
        ])
        .assert_ok();
        let out = e.run(&["context", "rm", "qmd://fixtures/notes"]);
        out.assert_ok();
        assert!(
            out.stdout
                .contains("✓ Removed context for: qmd://fixtures/notes")
        );
    }

    #[test]
    fn fails_to_remove_non_existent_context() {
        let e = seeded();
        let out = e.run(&["context", "rm", "qmd://nonexistent/path"]);
        out.assert_code(1);
        // rqmd's message is "No context found for: …" (contains "context found",
        // not the bare "not found" qmd used).
        assert!(
            out.stderr.contains("No context found"),
            "stderr: {}",
            out.stderr
        );
    }
}

// ===========================================================================
// CLI ls Command
// ===========================================================================
mod cli_ls {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn lists_all_collections() {
        let e = seeded();
        let out = e.run(&["ls"]);
        out.assert_ok();
        assert!(out.stdout.contains("Collections:"));
        assert!(out.stdout.contains("qmd://fixtures/"));
    }

    #[test]
    fn lists_files_in_a_collection() {
        let e = seeded();
        let out = e.run(&["ls", "fixtures"]);
        out.assert_ok();
        assert!(out.stdout.contains("qmd://fixtures/README.md"));
        assert!(out.stdout.contains("qmd://fixtures/notes/meeting.md"));
    }

    #[test]
    fn lists_files_with_path_prefix() {
        let e = seeded();
        let out = e.run(&["ls", "fixtures/notes"]);
        out.assert_ok();
        assert!(out.stdout.contains("qmd://fixtures/notes/meeting.md"));
        assert!(out.stdout.contains("qmd://fixtures/notes/ideas.md"));
        assert!(!out.stdout.contains("qmd://fixtures/README.md"));
    }

    #[test]
    fn lists_files_with_virtual_path() {
        let e = seeded();
        let out = e.run(&["ls", "qmd://fixtures/docs"]);
        out.assert_ok();
        assert!(out.stdout.contains("qmd://fixtures/docs/api.md"));
    }

    #[test]
    fn normalizes_extra_slashes_for_virtual_paths() {
        let e = seeded();
        let out = e.run(&["ls", "qmd:///fixtures/docs"]);
        out.assert_ok();
        assert_eq!(out.stderr, "");
        assert!(out.stdout.contains("qmd://fixtures/docs/api.md"));
    }

    #[test]
    fn handles_non_existent_collection() {
        let e = seeded();
        let out = e.run(&["ls", "nonexistent"]);
        out.assert_code(1);
        assert!(out.stderr.contains("Collection not found"));
    }

    // qmd's two absolute-path ls tests (`ls qmd://<abs>/…` and the
    // longest-prefix raw-path variant) are intentionally omitted:
    //  * On Windows the absolute path's drive letter (`C:/…`) collides with
    //    virtual-path parsing (it would be read as the collection name).
    //  * The collection is registered only in YAML; `rqmd update` does not
    //    sync a hand-written collection into the DB registry that `ls` reads,
    //    so `ls` would report "Collection not found" without an extra resync
    //    step (see `collection_ignore_patterns::status_shows_ignore_patterns`).
}

// ===========================================================================
// CLI Collection Commands
// ===========================================================================
mod cli_collection {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn lists_collections() {
        let e = seeded();
        let out = e.run(&["collection", "list"]);
        out.assert_ok();
        assert!(out.stdout.contains("Collections"));
        assert!(out.stdout.contains("fixtures"));
        assert!(out.stdout.contains("qmd://fixtures/"));
        assert!(out.stdout.contains("Pattern:"));
        assert!(out.stdout.contains("Files:"));
    }

    #[test]
    fn removes_a_collection() {
        let e = seeded();
        let before = e.run(&["collection", "list"]);
        assert!(before.stdout.contains("fixtures"));

        let out = e.run(&["collection", "remove", "fixtures"]);
        out.assert_ok();
        assert!(out.stdout.contains("✓ Removed collection 'fixtures'"));
        assert!(out.stdout.contains("Deleted"));

        let after = e.run(&["collection", "list"]);
        assert!(!after.stdout.contains("fixtures"));
    }

    #[test]
    fn handles_removing_non_existent_collection() {
        let e = seeded();
        let out = e.run(&["collection", "remove", "nonexistent"]);
        out.assert_code(1);
        assert!(out.stderr.contains("Collection not found"));
    }

    #[test]
    fn handles_missing_remove_argument() {
        let e = seeded();
        let out = e.run(&["collection", "remove"]);
        out.assert_err();
        assert!(out.stderr.contains("Usage:"), "stderr: {}", out.stderr);
    }

    #[test]
    fn handles_unknown_subcommand() {
        let e = seeded();
        let out = e.run(&["collection", "invalid"]);
        out.assert_err();
        assert!(
            out.stderr.contains("unrecognized subcommand"),
            "stderr: {}",
            out.stderr
        );
    }

    #[test]
    fn renames_a_collection() {
        let e = seeded();
        let before = e.run(&["collection", "list"]);
        assert!(before.stdout.contains("qmd://fixtures/"));

        let out = e.run(&["collection", "rename", "fixtures", "my-fixtures"]);
        out.assert_ok();
        assert!(
            out.stdout
                .contains("✓ Renamed collection 'fixtures' to 'my-fixtures'")
        );
        assert!(out.stdout.contains("qmd://fixtures/"));
        assert!(out.stdout.contains("qmd://my-fixtures/"));

        let after = e.run(&["collection", "list"]);
        assert!(after.stdout.contains("qmd://my-fixtures/"));
        assert!(!after.stdout.contains("qmd://fixtures/"));
    }

    #[test]
    fn handles_renaming_non_existent_collection() {
        let e = seeded();
        let out = e.run(&["collection", "rename", "nonexistent", "newname"]);
        out.assert_code(1);
        assert!(out.stderr.contains("Collection not found"));
    }

    #[test]
    fn handles_renaming_to_existing_collection_name() {
        let e = seeded();
        let second = e.root.join("second-coll");
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(second.join("test.md"), "# Test\n").unwrap();
        let abs = e.yaml_path(&second);
        e.run(&["collection", "add", &abs, "--name", "second"])
            .assert_ok();

        let both = e.run(&["collection", "list"]);
        assert!(both.stdout.contains("qmd://fixtures/"));
        assert!(both.stdout.contains("qmd://second/"));

        let out = e.run(&["collection", "rename", "fixtures", "second"]);
        out.assert_code(1);
        assert!(out.stderr.contains("Collection name already exists"));
    }

    #[test]
    fn handles_missing_rename_arguments() {
        let e = seeded();
        let none = e.run(&["collection", "rename"]);
        none.assert_err();
        assert!(none.stderr.contains("Usage:"));

        let one = e.run(&["collection", "rename", "fixtures"]);
        one.assert_err();
        assert!(one.stderr.contains("Usage:"));
    }
}

// ===========================================================================
// collection ignore patterns
// ===========================================================================
mod collection_ignore_patterns {
    use crate::common::*;
    use std::path::PathBuf;

    /// Build the ignore-fixtures tree under the env root and return its path.
    fn make_tree(e: &Env) -> PathBuf {
        let dir = e.root.join("ignore-fixtures");
        std::fs::create_dir_all(dir.join("notes")).unwrap();
        std::fs::create_dir_all(dir.join("sessions").join("2026-03")).unwrap();
        std::fs::create_dir_all(dir.join("archive")).unwrap();

        std::fs::write(
            dir.join("readme.md"),
            "# Main readme\nThis should be indexed.",
        )
        .unwrap();
        std::fs::write(
            dir.join("notes").join("note1.md"),
            "# Note 1\nThis is a personal note.",
        )
        .unwrap();
        std::fs::write(
            dir.join("sessions").join("session1.md"),
            "# Session 1\nThis session should be ignored.",
        )
        .unwrap();
        std::fs::write(
            dir.join("sessions").join("2026-03").join("session2.md"),
            "# Session 2\nNested session should also be ignored.",
        )
        .unwrap();
        std::fs::write(
            dir.join("archive").join("old.md"),
            "# Old stuff\nThis archive file should be ignored.",
        )
        .unwrap();
        dir
    }

    /// Seed an env with the ignore-fixtures tree indexed via a hand-written
    /// config that ignores `sessions/**` + `archive/**`, then `update`.
    fn seeded_with_ignore() -> Env {
        let e = env();
        let dir = make_tree(&e);
        let abs = e.yaml_path(&dir);
        e.write_config(&format!(
            "collections:\n  ignoretst:\n    path: \"{abs}\"\n    pattern: \"**/*.md\"\n    ignore:\n      - \"sessions/**\"\n      - \"archive/**\"\n"
        ));
        e.run_in(&dir, &["update"]).assert_ok();
        e
    }

    #[test]
    fn ignore_patterns_exclude_matching_files() {
        let e = env();
        let dir = make_tree(&e);
        let abs = e.yaml_path(&dir);
        e.write_config(&format!(
            "collections:\n  ignoretst:\n    path: \"{abs}\"\n    pattern: \"**/*.md\"\n    ignore:\n      - \"sessions/**\"\n      - \"archive/**\"\n"
        ));
        let out = e.run_in(&dir, &["update"]);
        out.assert_ok();
        // 2 indexed (readme.md + notes/note1.md), not 5.
        assert!(out.stdout.contains("2 new"), "stdout: {}", out.stdout);
    }

    #[test]
    fn ignored_files_are_not_searchable() {
        let e = seeded_with_ignore();
        let out = e.run(&["search", "session", "-n", "10"]);
        out.assert_ok();
        assert!(!out.stdout.contains("session1"));
        assert!(!out.stdout.contains("session2"));
    }

    #[test]
    fn non_ignored_files_are_searchable() {
        let e = seeded_with_ignore();
        let out = e.run(&["search", "personal note", "-n", "10"]);
        out.assert_ok();
        assert!(out.stdout.contains("note1"), "stdout: {}", out.stdout);
    }

    #[test]
    fn status_shows_ignore_patterns() {
        let e = seeded_with_ignore();
        // `collection list` reads the collection set from the DB registry
        // (store_collections), which `update` does not populate for a
        // hand-written YAML collection — only config mutations sync it. A
        // no-op `collection include` triggers that sync while preserving the
        // YAML `ignore:` list (Collection serializes `ignore`).
        e.run(&["collection", "include", "ignoretst"]).assert_ok();
        let out = e.run(&["collection", "list"]);
        out.assert_ok();
        assert!(out.stdout.contains("Ignore:"), "stdout: {}", out.stdout);
        assert!(out.stdout.contains("sessions/**"));
        assert!(out.stdout.contains("archive/**"));
    }

    #[test]
    fn collection_without_ignore_indexes_all_files() {
        let e = env();
        let dir = make_tree(&e);
        let abs = e.yaml_path(&dir);
        e.write_config(&format!(
            "collections:\n  allfiles:\n    path: \"{abs}\"\n    pattern: \"**/*.md\"\n"
        ));
        let out = e.run_in(&dir, &["update"]);
        out.assert_ok();
        assert!(out.stdout.contains("5 new"), "stdout: {}", out.stdout);
    }
}

// ===========================================================================
// search output formats — qmd:// URIs, context, docid
// ===========================================================================
//
// qmd's "custom-index search links include ?index= and can be passed back to
// qmd get" (cli.test.ts:1314) is ported in `mod custom_index_links` below:
// search output now annotates `qmd://` links with `?index=<name>` for a
// non-default index, and `get` honours that suffix to reopen the right index.
mod search_output_formats {
    use crate::common::*;

    const CTX: &str = "Test fixtures for QMD";

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e.run(&["context", "add", "qmd://fixtures/", CTX])
            .assert_ok();
        e
    }

    #[test]
    fn json_includes_qmd_path_docid_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "--json", "-n", "1"]);
        out.assert_ok();
        let v: serde_json::Value = serde_json::from_str(&out.stdout).expect("json");
        let r = &v.as_array().expect("array")[0];

        let file = r["file"].as_str().unwrap();
        assert!(file.starts_with("qmd://fixtures/"), "file: {file}");
        // rqmd's JSON `docid` is the bare 6-hex id (no leading '#').
        assert!(is_hex6(r["docid"].as_str().unwrap()), "docid: {r}");
        assert_eq!(r["context"].as_str().unwrap(), CTX);
        assert!(!file.contains("/Users/"));
        assert!(!file.contains("/home/"));
    }

    #[test]
    fn files_includes_path_docid_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "--files", "-n", "1"]);
        out.assert_ok();
        // Format: #docid,score,qmd://collection/path,"context" (qmd parity:
        // `outputResults` files branch uses the qmd:// URI, not a bare path).
        let line = first_line(&out.stdout);
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        assert!(
            parts[0].starts_with('#') && is_hex6(&parts[0][1..]),
            "line: {line}"
        );
        assert!(parts[2].starts_with("qmd://fixtures/"), "line: {line}");
        assert!(out.stdout.contains(CTX), "stdout: {}", out.stdout);
        assert!(!out.stdout.contains("/Users/"));
        assert!(!out.stdout.contains("/home/"));
    }

    #[test]
    fn csv_includes_path_docid_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "--csv", "-n", "1"]);
        out.assert_ok();
        assert!(
            out.stdout
                .contains("docid,score,file,title,context,line,snippet")
        );
        // Data row: #docid,score,qmd://collection/path,... (qmd parity).
        let row = out
            .stdout
            .lines()
            .find(|l| l.starts_with('#'))
            .expect("data row");
        let parts: Vec<&str> = row.splitn(4, ',').collect();
        assert!(is_hex6(&parts[0][1..]), "row: {row}");
        assert!(parts[2].starts_with("qmd://fixtures/"), "row: {row}");
        assert!(out.stdout.contains(CTX));
        assert!(!out.stdout.contains("/Users/"));
        assert!(!out.stdout.contains("/home/"));
    }

    #[test]
    fn md_includes_docid_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "--md", "-n", "1"]);
        out.assert_ok();
        assert!(
            out.stdout.contains("**docid:** `#"),
            "stdout: {}",
            out.stdout
        );
        assert!(out.stdout.contains(&format!("**context:** {CTX}")));
    }

    #[test]
    fn xml_includes_path_docid_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "--xml", "-n", "1"]);
        out.assert_ok();
        assert!(
            out.stdout.contains("<file docid=\"#"),
            "stdout: {}",
            out.stdout
        );
        assert!(out.stdout.contains("name=\"qmd://fixtures/"));
        assert!(out.stdout.contains(&format!("context=\"{CTX}\"")));
        assert!(!out.stdout.contains("/Users/"));
        assert!(!out.stdout.contains("/home/"));
    }

    #[test]
    fn default_cli_format_includes_path_and_context() {
        let e = seeded();
        let out = e.run(&["search", "test", "-n", "1"]);
        out.assert_ok();
        // qmd-parity cli format (non-TTY): a `qmd://` path line with docid, then
        // `Context:` and `Score:` lines. NO_COLOR + non-TTY ⇒ no ANSI / OSC-8.
        assert!(
            out.stdout.contains("qmd://fixtures/"),
            "stdout: {}",
            out.stdout
        );
        assert!(out.stdout.contains(&format!("Context: {CTX}")));
        assert!(out.stdout.contains("Score:"), "stdout: {}", out.stdout);
        assert!(!out.stdout.contains('\u{1b}')); // no ANSI / OSC-8
        assert!(!out.stdout.contains("/Users/"));
        assert!(!out.stdout.contains("/home/"));
    }
}

// ===========================================================================
// custom-index ?index= links — search annotation + get round-trip
// (port of qmd cli.test.ts:1314)
// ===========================================================================
mod custom_index_links {
    use crate::common::*;

    #[test]
    fn search_links_include_index_and_get_round_trips() {
        let e = env();
        // One cache dir shared across all three invocations so the named-index
        // DB (`<cache>/release-notes.sqlite`) persists between them. Using
        // `spawn_cache` (no RQMD_INDEX_PATH) lets `--index <name>` and a link's
        // `?index=<name>` both resolve to that file.
        let cache = tempfile::tempdir().expect("mkdtemp cache");
        let coll = "fixtures-alt";
        let idx = "release-notes";

        // 1. Index the fixtures under a non-default named index.
        spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &["--index", idx, "collection", "add", ".", "--name", coll],
        )
        .assert_ok();

        // 2. Search that index: the JSON `file` carries `?index=<name>`.
        //    Flags may sit on either side of the query now; here they follow it.
        let s = spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &["--index", idx, "search", "test", "--json", "-n", "1"],
        );
        s.assert_ok();
        let v: serde_json::Value = serde_json::from_str(&s.stdout).expect("json");
        let file = v.as_array().expect("array")[0]["file"]
            .as_str()
            .expect("file field");
        assert!(file.starts_with(&format!("qmd://{coll}/")), "file: {file}");
        assert!(file.ends_with(&format!("?index={idx}")), "file: {file}");

        // 3. Round-trip: feed the link to `get` WITHOUT `--index`; the `?index=`
        //    suffix alone must reopen the right index DB.
        let g = spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &["get", file, "-l", "2"],
        );
        g.assert_ok();
        assert!(!g.stdout.trim().is_empty(), "stdout: {}", g.stdout);
    }
}

// ===========================================================================
// get command path normalization
// ===========================================================================
mod get_path_normalization {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn get_with_qmd_collection_path() {
        let e = seeded();
        let out = e.run(&["get", "qmd://fixtures/test1.md", "-l", "3"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Document 1"));
    }

    #[test]
    fn get_with_collection_path_no_scheme() {
        let e = seeded();
        let out = e.run(&["get", "fixtures/test1.md", "-l", "3"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Document 1"));
    }

    #[test]
    fn get_with_double_slash_path() {
        let e = seeded();
        let out = e.run(&["get", "//fixtures/test1.md", "-l", "3"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Document 1"));
    }

    // Previously dropped on the assumption rqmd couldn't normalize the 4-slash
    // form — but it can, so port qmd's original test verbatim: `qmd:////` is
    // normalized like `qmd://`/`//` and returns the document.
    #[test]
    fn get_with_quadruple_slash_path() {
        let e = seeded();
        let out = e.run(&["get", "qmd:////fixtures/test1.md", "-l", "3"]);
        out.assert_ok();
        assert!(out.stdout.contains("Test Document 1"));
    }

    #[test]
    fn get_with_path_line_suffix() {
        let e = seeded();
        let out = e.run(&["get", "fixtures/test1.md:3", "-l", "2"]);
        out.assert_ok();
        // Starts at line 3, so the line-1 heading is not at the top.
        assert!(!out.stdout.lines().any(|l| l == "# Test Document 1"));
    }

    #[test]
    fn get_with_qmd_path_line_suffix() {
        let e = seeded();
        let out = e.run(&["get", "qmd://fixtures/test1.md:3", "-l", "2"]);
        out.assert_ok();
        assert!(!out.stdout.lines().any(|l| l == "# Test Document 1"));
    }
}

// ===========================================================================
// status & collection list hide filesystem paths
// ===========================================================================
mod hide_filesystem_paths {
    use crate::common::*;

    fn seeded() -> Env {
        let e = env();
        e.run(&["collection", "add", "."]).assert_ok();
        e
    }

    #[test]
    fn status_does_not_show_collection_filesystem_paths() {
        let e = seeded();
        let out = e.run(&["status"]);
        out.assert_ok();
        assert!(out.stdout.contains("qmd://fixtures/"));
        // status only ever prints the collection as a qmd:// URI; the on-disk
        // fixtures path must not leak. (The `Index:` line legitimately shows the
        // db path, which is the temp db, not the fixtures dir.)
        let fwd = e.yaml_path(&e.fixtures);
        assert!(
            !out.stdout.contains(&fwd),
            "leaked path: {fwd}\n{}",
            out.stdout
        );
        assert!(!out.stdout.contains(e.fixtures.to_str().unwrap()));
    }

    #[test]
    fn collection_list_does_not_show_filesystem_paths() {
        let e = seeded();
        let out = e.run(&["collection", "list"]);
        out.assert_ok();
        assert!(out.stdout.contains("qmd://fixtures/"));
        // `collection list` prints `Pattern:` but never a `Path:` line
        // (only `collection show` does).
        assert!(!out.stdout.contains("Path:"), "stdout: {}", out.stdout);
    }
}

// ===========================================================================
// MCP foreground HTTP server (`rqmd mcp --http`)
// ===========================================================================
//
// Port of qmd's "foreground HTTP server" cases (cli.test.ts:1652-1732). These
// spawn the real `rqmd mcp --http` binary, poll `/health`, and POST `/query`.
// The MCP *daemon* lifecycle (`--daemon`/`stop`/PID) is covered separately.
mod cli_mcp_http {
    use crate::common::*;
    use std::time::Duration;

    #[test]
    fn foreground_http_server_responds_to_health_check() {
        let e = env();
        let cache = tempfile::tempdir().expect("mkdtemp cache");
        let idx = "mcp-health";
        // Seed an index so the server opens a real store.
        spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index",
                idx,
                "collection",
                "add",
                ".",
                "--name",
                "fixtures",
            ],
        )
        .assert_ok();

        let port = free_port();
        let port_s = port.to_string();
        let _server = ServerChild {
            child: spawn_cache_child(
                &e.fixtures,
                cache.path(),
                &e.config_dir,
                &["--index", idx, "mcp", "--http", "--port", &port_s],
            ),
            port,
        };

        let body = wait_for_health(port, Duration::from_secs(15)).expect("server health");
        assert_eq!(body["status"], "ok", "health body: {body}");
        // `_server` dropped here → child killed + reaped.
    }

    #[test]
    fn foreground_http_server_honors_index_and_serves_query() {
        let e = env();
        let cache = tempfile::tempdir().expect("mkdtemp cache");
        let idx = "mcp-alt";
        let coll = "mcp-fixtures";
        spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &["--index", idx, "collection", "add", ".", "--name", coll],
        )
        .assert_ok();

        let port = free_port();
        let port_s = port.to_string();
        let _server = ServerChild {
            child: spawn_cache_child(
                &e.fixtures,
                cache.path(),
                &e.config_dir,
                &["--index", idx, "mcp", "--http", "--port", &port_s],
            ),
            port,
        };
        wait_for_health(port, Duration::from_secs(15)).expect("server health");

        // "authentication" matches notes/meeting.md ("Bob to fix authentication
        // bug"). A `lex` sub-query with rerank:false keeps this LLM-free.
        let resp = post_query(
            port,
            serde_json::json!({
                "searches": [{ "type": "lex", "query": "authentication" }],
                "limit": 5,
                "rerank": false
            }),
        );
        let results = resp["results"].as_array().expect("results array");
        let files: Vec<&str> = results.iter().filter_map(|r| r["file"].as_str()).collect();
        assert!(
            files
                .iter()
                .any(|f| f.contains(&format!("{coll}/notes/meeting.md"))),
            "expected a {coll}/notes/meeting.md hit, got: {files:?}"
        );
    }
}

// ===========================================================================
// MCP HTTP daemon lifecycle (`mcp --http --daemon`, `mcp stop`)
// ===========================================================================
//
// Port of qmd's "mcp http daemon" cases (cli.test.ts:1738-1847). The detached
// daemon is killed via `mcp stop` (a `DaemonGuard` does this on drop, since the
// child is not tracked by any handle). PID/log live at `<RQMD_CACHE_DIR>/mcp.*`.
mod cli_mcp_daemon {
    use crate::common::*;
    use std::time::Duration;

    /// Seed a collection under `idx` in `cache`, returning nothing — callers
    /// reuse the same `cache`/`cfg` for the daemon so the store matches.
    fn seed(e: &Env, cache: &std::path::Path, idx: &str) {
        spawn_cache(
            &e.fixtures,
            cache,
            &e.config_dir,
            &[
                "--index",
                idx,
                "collection",
                "add",
                ".",
                "--name",
                "fixtures",
            ],
        )
        .assert_ok();
    }

    #[test]
    fn daemon_writes_pid_file_and_serves() {
        let e = env();
        let cache = tempfile::tempdir().expect("cache");
        let idx = "daemon-serves";
        seed(&e, cache.path(), idx);
        let _guard = DaemonGuard {
            cwd: e.fixtures.clone(),
            cache: cache.path().to_path_buf(),
            cfg: e.config_dir.clone(),
        };

        let port = free_port();
        let port_s = port.to_string();
        let out = spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index", idx, "mcp", "--http", "--daemon", "--port", &port_s,
            ],
        );
        out.assert_ok();
        assert!(
            out.stdout.contains(&format!("http://localhost:{port}/mcp")),
            "stdout: {}",
            out.stdout
        );

        let pid_file = cache.path().join("mcp.pid");
        assert!(
            pid_file.exists(),
            "PID file missing: {}",
            pid_file.display()
        );
        assert!(
            wait_for_health(port, Duration::from_secs(15)).is_some(),
            "daemon did not become healthy"
        );
    }

    #[test]
    fn stop_kills_daemon_and_removes_pid_file() {
        let e = env();
        let cache = tempfile::tempdir().expect("cache");
        let idx = "daemon-stop";
        seed(&e, cache.path(), idx);
        let _guard = DaemonGuard {
            cwd: e.fixtures.clone(),
            cache: cache.path().to_path_buf(),
            cfg: e.config_dir.clone(),
        };

        let port = free_port();
        let port_s = port.to_string();
        spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index", idx, "mcp", "--http", "--daemon", "--port", &port_s,
            ],
        )
        .assert_ok();
        wait_for_health(port, Duration::from_secs(15)).expect("daemon health");

        let pid_file = cache.path().join("mcp.pid");
        let stop = spawn_cache(&e.fixtures, cache.path(), &e.config_dir, &["mcp", "stop"]);
        stop.assert_ok();
        assert!(stop.stdout.contains("Stopped"), "stdout: {}", stop.stdout);
        assert!(!pid_file.exists(), "PID file should be removed");
    }

    #[test]
    fn stop_handles_dead_pid_as_stale() {
        let e = env();
        let cache = tempfile::tempdir().expect("cache");
        let pid_file = cache.path().join("mcp.pid");
        std::fs::write(&pid_file, "999999999").unwrap();

        let out = spawn_cache(&e.fixtures, cache.path(), &e.config_dir, &["mcp", "stop"]);
        out.assert_ok();
        assert!(out.stdout.contains("stale"), "stdout: {}", out.stdout);
        assert!(!pid_file.exists(), "stale PID file should be removed");
    }

    #[test]
    fn daemon_rejects_when_already_running() {
        let e = env();
        let cache = tempfile::tempdir().expect("cache");
        let idx = "daemon-dup";
        seed(&e, cache.path(), idx);
        let _guard = DaemonGuard {
            cwd: e.fixtures.clone(),
            cache: cache.path().to_path_buf(),
            cfg: e.config_dir.clone(),
        };

        let port = free_port();
        let port_s = port.to_string();
        spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index", idx, "mcp", "--http", "--daemon", "--port", &port_s,
            ],
        )
        .assert_ok();
        wait_for_health(port, Duration::from_secs(15)).expect("daemon health");

        let port2 = free_port().to_string();
        let second = spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index", idx, "mcp", "--http", "--daemon", "--port", &port2,
            ],
        );
        second.assert_code(1);
        assert!(
            second.stderr.contains("Already running"),
            "stderr: {}",
            second.stderr
        );
    }

    #[test]
    fn daemon_cleans_stale_pid_and_starts_fresh() {
        let e = env();
        let cache = tempfile::tempdir().expect("cache");
        let idx = "daemon-stale";
        seed(&e, cache.path(), idx);
        let pid_file = cache.path().join("mcp.pid");
        std::fs::write(&pid_file, "999999999").unwrap();
        let _guard = DaemonGuard {
            cwd: e.fixtures.clone(),
            cache: cache.path().to_path_buf(),
            cfg: e.config_dir.clone(),
        };

        let port = free_port();
        let port_s = port.to_string();
        let out = spawn_cache(
            &e.fixtures,
            cache.path(),
            &e.config_dir,
            &[
                "--index", idx, "mcp", "--http", "--daemon", "--port", &port_s,
            ],
        );
        out.assert_ok();
        assert!(
            out.stdout.contains(&format!("http://localhost:{port}/mcp")),
            "stdout: {}",
            out.stdout
        );
        let new_pid = std::fs::read_to_string(&pid_file).unwrap();
        assert_ne!(new_pid.trim(), "999999999", "stale PID should be replaced");
        assert!(
            wait_for_health(port, Duration::from_secs(15)).is_some(),
            "fresh daemon did not become healthy"
        );
    }
}
