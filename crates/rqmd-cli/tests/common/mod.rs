//! Shared harness for the E2E CLI tests (`tests/cli.rs`).
//!
//! Rust port of the `runQmd()` helper + `beforeAll` fixtures in qmd's
//! `test/cli.test.ts`. Each test gets its own [`Env`] (a fresh `TempDir` with
//! the same markdown fixtures qmd creates) and spawns the built `rqmd` binary
//! with fully isolated state:
//!
//! * `RQMD_INDEX_PATH`  — points the SQLite index at the temp dir
//!   (honoured before the production-mode gate, `store/path.rs`).
//! * `RQMD_CONFIG_DIR`  — points the YAML config (`index.yml`) at the temp dir.
//! * `PWD`              — rqmd's `pwd()` prefers this over `current_dir()`.
//! * `--index index`    — prepended so `IndexState::new` skips the `.rqmd/`
//!   local-config walk up the ancestors of the cwd (which, on Windows, live
//!   under the user profile alongside the OS temp dir). This keeps every run
//!   hermetic regardless of the host's `~/.rqmd` / `~/.config/rqmd`.
//! * `NO_COLOR=1`       — strip ANSI so substring assertions are clean.
//! * `CI=1`             — make any accidental LLM/model path fail fast instead
//!   of hitting the network (none of the ported commands need a model).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;

/// Result of one `rqmd` invocation.
pub struct Out {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
}

impl Out {
    /// Assert exit code 0, surfacing stderr on failure.
    pub fn assert_ok(&self) -> &Self {
        assert_eq!(
            self.code, 0,
            "expected exit 0\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout, self.stderr
        );
        self
    }

    /// Assert a non-zero exit code (clap parse errors use 2; app errors use 1).
    pub fn assert_err(&self) -> &Self {
        assert_ne!(
            self.code, 0,
            "expected non-zero exit\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout, self.stderr
        );
        self
    }

    /// Assert exit code 1 (an application error routed through `main`'s
    /// `eprintln!("error: …"); exit(1)`).
    pub fn assert_code(&self, code: i32) -> &Self {
        assert_eq!(
            self.code, code,
            "expected exit {code}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout, self.stderr
        );
        self
    }
}

/// One isolated test environment: a temp dir holding the markdown fixtures, a
/// `config/` dir for `index.yml`, and a default index db path.
pub struct Env {
    /// Kept alive for the duration of the test; dropping it removes the dir.
    _root: tempfile::TempDir,
    pub root: PathBuf,
    pub fixtures: PathBuf,
    pub config_dir: PathBuf,
    pub db: PathBuf,
}

impl Env {
    /// Spawn `rqmd <args>` from the fixtures dir with the default isolated db
    /// + config. `--index index` is prepended (see module docs).
    pub fn run(&self, args: &[&str]) -> Out {
        self.run_in(&self.fixtures, args)
    }

    /// Like [`run`](Self::run) but from an explicit working directory.
    pub fn run_in(&self, cwd: &Path, args: &[&str]) -> Out {
        let mut full: Vec<&str> = vec!["--index", "index"];
        full.extend_from_slice(args);
        spawn(cwd, &self.db, &self.config_dir, &full, &[])
    }

    /// Like [`run`](Self::run) but with extra environment overrides applied
    /// last (so they win over the defaults — used to point at a custom
    /// `RQMD_INDEX_PATH`).
    pub fn run_env(&self, args: &[&str], extra: &[(&str, &str)]) -> Out {
        let mut full: Vec<&str> = vec!["--index", "index"];
        full.extend_from_slice(args);
        spawn(&self.fixtures, &self.db, &self.config_dir, &full, extra)
    }

    /// Spawn `rqmd <args>` verbatim (no `--index` prepend). Used for the
    /// help / no-args parser tests that don't touch the index.
    pub fn run_bare(&self, args: &[&str]) -> Out {
        spawn(&self.fixtures, &self.db, &self.config_dir, args, &[])
    }

    /// Overwrite `<config_dir>/index.yml` (the global `beforeEach` reset, and
    /// the ignore-pattern tests that hand-write a collections config).
    pub fn write_config(&self, yaml: &str) {
        std::fs::write(self.config_dir.join("index.yml"), yaml).expect("write index.yml");
    }

    /// Forward-slash form of an absolute path under `root`, safe to embed in a
    /// double-quoted YAML scalar on Windows (`C:/Users/...`).
    pub fn yaml_path(&self, p: &Path) -> String {
        p.to_string_lossy().replace('\\', "/")
    }
}

/// Create a fresh isolated [`Env`], mirroring qmd's `beforeAll` fixtures.
/// The fixtures subdir is literally named `fixtures` so `collection add .`
/// derives the collection name `fixtures` (qmd relies on the same basename).
pub fn env() -> Env {
    let root = tempfile::tempdir().expect("mkdtemp");
    let root_path = root.path().to_path_buf();
    let fixtures = root_path.join("fixtures");
    let config_dir = root_path.join("config");
    let db = root_path.join("index.sqlite");

    std::fs::create_dir_all(fixtures.join("notes")).unwrap();
    std::fs::create_dir_all(fixtures.join("docs")).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").unwrap();

    write_fixtures(&fixtures);

    Env {
        _root: root,
        root: root_path,
        fixtures,
        config_dir,
        db,
    }
}

fn write_fixtures(fixtures: &Path) {
    std::fs::write(
        fixtures.join("README.md"),
        "# Test Project\n\n\
         This is a test project for QMD CLI testing.\n\n\
         ## Features\n\n\
         - Full-text search with BM25\n\
         - Vector similarity search\n\
         - Hybrid search with reranking\n",
    )
    .unwrap();

    std::fs::write(
        fixtures.join("notes").join("meeting.md"),
        "# Team Meeting Notes\n\n\
         Date: 2024-01-15\n\n\
         ## Attendees\n\
         - Alice\n\
         - Bob\n\
         - Charlie\n\n\
         ## Discussion Topics\n\
         - Project timeline review\n\
         - Resource allocation\n\
         - Technical debt prioritization\n\n\
         ## Action Items\n\
         1. Alice to update documentation\n\
         2. Bob to fix authentication bug\n\
         3. Charlie to review pull requests\n",
    )
    .unwrap();

    std::fs::write(
        fixtures.join("notes").join("ideas.md"),
        "# Product Ideas\n\n\
         ## Feature Requests\n\
         - Dark mode support\n\
         - Keyboard shortcuts\n\
         - Export to PDF\n\n\
         ## Technical Improvements\n\
         - Improve search performance\n\
         - Add caching layer\n\
         - Optimize database queries\n",
    )
    .unwrap();

    std::fs::write(
        fixtures.join("docs").join("api.md"),
        "# API Documentation\n\n\
         ## Endpoints\n\n\
         ### GET /search\n\
         Search for documents.\n\n\
         Parameters:\n\
         - q: Search query (required)\n\
         - limit: Max results (default: 10)\n\n\
         ### GET /document/:id\n\
         Retrieve a specific document.\n\n\
         ### POST /index\n\
         Index new documents.\n",
    )
    .unwrap();

    std::fs::write(
        fixtures.join("test1.md"),
        "# Test Document 1\n\n\
         This is the first test document.\n\n\
         It has multiple lines for testing line numbers.\n\
         Line 6 is here.\n\
         Line 7 is here.\n",
    )
    .unwrap();

    std::fs::write(
        fixtures.join("test2.md"),
        "# Test Document 2\n\n\
         This is the second test document.\n",
    )
    .unwrap();
}

fn spawn(cwd: &Path, db: &Path, cfg: &Path, args: &[&str], extra: &[(&str, &str)]) -> Out {
    let mut cmd = Command::cargo_bin("rqmd").expect("rqmd binary is built by cargo test");
    cmd.current_dir(cwd)
        .env_remove("XDG_CACHE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("RQMD_CACHE_DIR")
        .env("NO_COLOR", "1")
        .env("CI", "1")
        .env("PWD", cwd)
        .env("RQMD_INDEX_PATH", db)
        .env("RQMD_CONFIG_DIR", cfg)
        .args(args);
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn rqmd");
    Out {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code().unwrap_or(-1),
    }
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// True if `s` is exactly 6 lowercase-hex characters (rqmd's docid shape).
pub fn is_hex6(s: &str) -> bool {
    s.len() == 6
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// First non-empty line of `s` (the first data row for csv/files output).
pub fn first_line(s: &str) -> &str {
    s.lines().find(|l| !l.trim().is_empty()).unwrap_or("")
}
