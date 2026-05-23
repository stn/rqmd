//! Search-quality evaluation harness — faithful port of `tobi/qmd`'s
//! `test/eval-harness.ts`.
//!
//! Unlike the in-process `rqmd-core/tests/eval.rs` (which ports qmd's
//! `eval.test.ts` and drives the store functions directly), this is the port of
//! qmd's *standalone CLI harness*: it shells out to the built `rqmd` binary —
//! exactly as the upstream `execSync`s `bun src/cli/qmd.ts …` — and exercises
//! the two end-to-end CLI modes:
//!
//! 1. **`search`** — FTS / BM25 only, no model. Built by `collection add`.
//! 2. **`query`**  — the full RAG pipeline (LLM expansion + hybrid + LLM
//!    rerank). This pipeline is *not* covered anywhere in `eval.rs`, so this
//!    harness adds genuinely new end-to-end coverage.
//!
//! It runs 18 known-answer queries (easy / medium / hard, 6 each) through each
//! mode and prints a Hit@1 / Hit@3 / Hit@5 report per difficulty plus an overall
//! line. Like the upstream harness it is **report-only**: there are no hit-rate
//! thresholds, so run with `-- --nocapture` to see the table. The setup steps
//! (`collection add`, `embed`) *are* asserted so a broken binary fails loudly;
//! per-query command/parse errors are swallowed and counted as a miss, mirroring
//! the upstream `try/catch → return []`.
//!
//! The `query` mode needs the ~embed + generate (expansion) + rerank models, so
//! the whole suite runs by default and skips when `RQMD_SKIP_LLM_TESTS` is set —
//! the same gate `eval.rs` uses (CI must export `RQMD_SKIP_LLM_TESTS=1`).

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use serde::Deserialize;

// =============================================================================
// Eval queries with expected documents (eval-harness.ts:11-130, verbatim incl.
// the `description` field)
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

use Difficulty::{Easy, Hard, Medium};

struct EvalQuery {
    query: &'static str,
    /// Substring expected to appear in the matching document's path (lowercase).
    expected: &'static str,
    difficulty: Difficulty,
    /// Free-text note carried through to the per-query report line.
    description: &'static str,
}

static EVAL_QUERIES: &[EvalQuery] = &[
    // EASY: Exact keyword matches
    EvalQuery {
        query: "API versioning",
        expected: "api-design",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    EvalQuery {
        query: "Series A fundraising",
        expected: "fundraising",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    EvalQuery {
        query: "CAP theorem",
        expected: "distributed-systems",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    EvalQuery {
        query: "overfitting machine learning",
        expected: "machine-learning",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    EvalQuery {
        query: "remote work VPN",
        expected: "remote-work",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    EvalQuery {
        query: "Project Phoenix retrospective",
        expected: "product-launch",
        difficulty: Easy,
        description: "Direct keyword match",
    },
    // MEDIUM: Semantic/conceptual queries
    EvalQuery {
        query: "how to structure REST endpoints",
        expected: "api-design",
        difficulty: Medium,
        description: "Conceptual - no exact match",
    },
    EvalQuery {
        query: "raising money for startup",
        expected: "fundraising",
        difficulty: Medium,
        description: "Conceptual - synonyms",
    },
    EvalQuery {
        query: "consistency vs availability tradeoffs",
        expected: "distributed-systems",
        difficulty: Medium,
        description: "Conceptual understanding",
    },
    EvalQuery {
        query: "how to prevent models from memorizing data",
        expected: "machine-learning",
        difficulty: Medium,
        description: "Conceptual - overfitting",
    },
    EvalQuery {
        query: "working from home guidelines",
        expected: "remote-work",
        difficulty: Medium,
        description: "Synonym match",
    },
    EvalQuery {
        query: "what went wrong with the launch",
        expected: "product-launch",
        difficulty: Medium,
        description: "Conceptual query",
    },
    // HARD: Vague, partial memory, indirect
    EvalQuery {
        query: "nouns not verbs",
        expected: "api-design",
        difficulty: Hard,
        description: "Partial phrase recall",
    },
    EvalQuery {
        query: "Sequoia investor pitch",
        expected: "fundraising",
        difficulty: Hard,
        description: "Indirect reference",
    },
    EvalQuery {
        query: "Raft algorithm leader election",
        expected: "distributed-systems",
        difficulty: Hard,
        description: "Specific detail in long doc",
    },
    EvalQuery {
        query: "F1 score precision recall",
        expected: "machine-learning",
        difficulty: Hard,
        description: "Technical detail",
    },
    EvalQuery {
        query: "quarterly team gathering travel",
        expected: "remote-work",
        difficulty: Hard,
        description: "Specific policy detail",
    },
    EvalQuery {
        query: "beta program 47 bugs",
        expected: "product-launch",
        difficulty: Hard,
        description: "Specific number recall",
    },
];

fn difficulty_label(d: Difficulty) -> &'static str {
    match d {
        Easy => "easy",
        Medium => "medium",
        Hard => "hard",
    }
}

/// Mirrors qmd's `r.file.toLowerCase().includes(expectedDoc)`. The `expected`
/// substrings are already lowercase, so only the haystack is lowered.
fn matches_expected(file: &str, expected: &str) -> bool {
    file.to_lowercase().contains(expected)
}

// =============================================================================
// Binary harness — one isolated `rqmd` invocation per call
// =============================================================================

/// Minimal field set of `rqmd {search,query} --json` (a plain array of `Hit`
/// objects without `--explain`). Only `file` is needed for matching.
#[derive(Deserialize)]
struct EvalHit {
    file: String,
}

struct Out {
    stdout: String,
    code: i32,
    stderr: String,
}

/// Isolated test environment: a temp dir holding the copied eval-docs fixtures,
/// a `config/` dir for `index.yml`, and a pinned index db path.
struct Harness {
    root: PathBuf,
    db: PathBuf,
    cfg: PathBuf,
}

impl Harness {
    /// Spawn `rqmd --index index <args>` against the isolated index/config.
    ///
    /// Deliberately **not** the `tests/common/mod.rs` helper: that one hardcodes
    /// `CI=1`, which sets `ci_mode` and makes `embed`/`query` return
    /// `Error::CiDisabled`. Here we instead *remove* `CI` so models load, and we
    /// leave the model cache dirs (`XDG_CACHE_HOME` → `dirs::cache_dir()`)
    /// untouched so the embed/generate/rerank models download once and persist
    /// across runs. `RQMD_INDEX_PATH` pins the DB; `--index index` skips the
    /// `.rqmd/` ancestor-config walk for hermeticity (same combo as
    /// `common/mod.rs::run_in`).
    fn run(&self, args: &[&str]) -> Out {
        let mut full: Vec<&str> = vec!["--index", "index"];
        full.extend_from_slice(args);
        let mut cmd = Command::cargo_bin("rqmd").expect("rqmd binary is built by cargo test");
        cmd.current_dir(&self.root)
            .env_remove("CI")
            .env_remove("XDG_CONFIG_HOME")
            .env("NO_COLOR", "1")
            .env("PWD", &self.root)
            .env("RQMD_INDEX_PATH", &self.db)
            .env("RQMD_CONFIG_DIR", &self.cfg)
            .args(&full);
        let out = cmd.output().expect("spawn rqmd");
        Out {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            code: out.status.code().unwrap_or(-1),
        }
    }

    /// Run one query through `mode` (`"search"` or `"query"`) and return the
    /// `file` of each hit (top 5). Faithful to the upstream `try/catch → []`:
    /// any non-zero exit or JSON-parse failure yields an empty list (a miss).
    fn search(&self, mode: &str, query: &str) -> Vec<String> {
        let out = self.run(&[mode, query, "--json", "-n", "5"]);
        if out.code != 0 {
            return Vec::new();
        }
        let hits: Vec<EvalHit> = serde_json::from_str(&out.stdout).unwrap_or_default();
        hits.into_iter().map(|h| h.file).collect()
    }

    /// `evaluate(mode)` from eval-harness.ts:162-201: run all 18 queries, tally
    /// Hit@1/3/5 per difficulty, and print the per-query lines, the per-difficulty
    /// summary, and the overall line.
    fn evaluate(&self, mode: &str) {
        eprintln!("\n=== Evaluating {} mode ===\n", mode.to_uppercase());

        let mut buckets = [Bucket::default(); 3]; // easy / medium / hard
        for q in EVAL_QUERIES {
            let hits = self.search(mode, q.query);
            // 1-indexed rank of the first matching hit (qmd's `firstHit`).
            let first_hit = hits
                .iter()
                .position(|f| matches_expected(f, q.expected))
                .map(|i| i + 1);

            let b = &mut buckets[bucket_index(q.difficulty)];
            b.total += 1;
            if first_hit == Some(1) {
                b.hit1 += 1;
            }
            if matches!(first_hit, Some(n) if (1..=3).contains(&n)) {
                b.hit3 += 1;
            }
            if matches!(first_hit, Some(n) if (1..=5).contains(&n)) {
                b.hit5 += 1;
            }

            let status = match first_hit {
                Some(1) => "✓".to_string(),
                Some(n) => format!("@{n}"),
                None => "✗".to_string(),
            };
            eprintln!(
                "[{:<6}] {:<3} \"{}\" → {}",
                difficulty_label(q.difficulty),
                status,
                q.query,
                q.description
            );
        }

        eprintln!("\n--- Summary ---");
        for (i, label) in ["easy", "medium", "hard"].iter().enumerate() {
            let b = buckets[i];
            eprintln!(
                "{:<8}: Hit@1={}% Hit@3={}% Hit@5={}% (n={})",
                label,
                pct(b.hit1, b.total),
                pct(b.hit3, b.total),
                pct(b.hit5, b.total),
                b.total
            );
        }

        let total = EVAL_QUERIES.len();
        let total_hit1: usize = buckets.iter().map(|b| b.hit1).sum();
        let total_hit3: usize = buckets.iter().map(|b| b.hit3).sum();
        eprintln!(
            "\nOverall: Hit@1={}% Hit@3={}%",
            pct(total_hit1, total),
            pct(total_hit3, total)
        );
    }
}

#[derive(Default, Clone, Copy)]
struct Bucket {
    total: usize,
    hit1: usize,
    hit3: usize,
    hit5: usize,
}

fn bucket_index(d: Difficulty) -> usize {
    match d {
        Easy => 0,
        Medium => 1,
        Hard => 2,
    }
}

/// Integer percentage `round(hit / total * 100)` (qmd's `.toFixed(0)`).
fn pct(hit: usize, total: usize) -> i64 {
    if total == 0 {
        return 0;
    }
    ((hit as f64 / total as f64) * 100.0).round() as i64
}

// =============================================================================
// Entry point
// =============================================================================

/// Run by default; skip when `RQMD_SKIP_LLM_TESTS` is set (CI sets it).
fn skip_llm() -> bool {
    std::env::var("RQMD_SKIP_LLM_TESTS").is_ok()
}

#[test]
fn eval_harness() {
    if skip_llm() {
        eprintln!("RQMD_SKIP_LLM_TESTS set — skipping eval-harness suite (needs models)");
        return;
    }

    // --- Setup: isolated temp index + config, with the eval-docs fixtures
    //     copied in (mirrors the qmd `collection add` + `embed` prerequisite). ---
    let tmp = tempfile::tempdir().expect("mkdtemp");
    let h = Harness {
        root: tmp.path().to_path_buf(),
        db: tmp.path().join("index.sqlite"),
        cfg: tmp.path().join("config"),
    };
    let docs_dir = h.root.join("eval-docs");
    std::fs::create_dir_all(&h.cfg).unwrap();
    std::fs::create_dir_all(&docs_dir).unwrap();
    std::fs::write(h.cfg.join("index.yml"), "collections: {}\n").unwrap();

    // Reuse the existing fixtures from rqmd-core (no duplication in the tree).
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/../rqmd-core/tests/eval-docs");
    for entry in std::fs::read_dir(src).expect("read eval-docs source dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            let name = path.file_name().unwrap();
            std::fs::copy(&path, docs_dir.join(name)).unwrap();
        }
    }

    // `collection add` builds documents + the FTS index (no model) → `search`
    // works after this step alone.
    let docs_str = docs_dir.to_str().unwrap();
    let add = h.run(&["collection", "add", docs_str, "--name", "eval-docs"]);
    assert_eq!(
        add.code, 0,
        "collection add failed (exit {})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        add.code, add.stdout, add.stderr
    );

    // `embed` adds the vector embeddings needed by `query` (downloads the embed
    // model on first run).
    let emb = h.run(&["embed"]);
    assert_eq!(
        emb.code, 0,
        "embed failed (exit {})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        emb.code, emb.stdout, emb.stderr
    );

    // --- Report (eval-harness.ts:204-223). ---
    eprintln!("rqmd Evaluation Harness");
    eprintln!("{}", "=".repeat(50));
    eprintln!("Testing {} queries across 6 documents", EVAL_QUERIES.len());

    h.evaluate("search");
    h.evaluate("query");
}
