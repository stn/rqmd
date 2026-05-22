//! `rqmd mcp` — start the Model Context Protocol server (and manage its daemon).
//!
//! Maps to qmd's `mcp` CLI handler, which calls `startMcpServer` /
//! `startMcpHttpServer` from `src/mcp/server.ts`. The protocol + transports live
//! in the `rqmd-mcp` crate; this handler resolves the active index into an
//! [`RqmdStore`] and hands it off, or manages the background daemon.
//!
//! - `rqmd mcp` — stdio transport (default; what MCP clients use).
//! - `rqmd mcp --http [--port N]` — Streamable HTTP at `/mcp` (default port 8181)
//!   plus the REST `/health`, `/query`, `/search` endpoints.
//! - `rqmd mcp --http --daemon [--port N]` — background HTTP server; writes a PID
//!   file under the cache dir (`<cache>/mcp.pid`), logs to `<cache>/mcp.log`.
//! - `rqmd mcp stop` — stop the daemon (kills the PID file's process).
//!
//! PID/log live under [`rqmd_core::paths::cache_dir`] (honours `RQMD_CACHE_DIR`)
//! — the same base `default_db_path` uses, so daemon and store stay consistent.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use rqmd_core::RqmdStore;

use crate::cli::{McpAction, McpArgs};
use crate::state::IndexState;

/// Default HTTP port (matches qmd's `mcp --http` default).
const DEFAULT_HTTP_PORT: u16 = 8181;

pub async fn run(args: McpArgs, state: &mut IndexState) -> Result<()> {
    if let Some(McpAction::Stop) = args.action {
        return stop();
    }

    if args.daemon {
        // The daemon is HTTP-only; spawning is synchronous (no .await).
        return start_daemon(&args, state);
    }

    let options = state.rqmd_store_options()?;
    let store = RqmdStore::open(options).context("opening index for the MCP server")?;

    if args.http {
        let port = args.port.unwrap_or(DEFAULT_HTTP_PORT);
        rqmd_mcp::serve_http(port, store, false).await?;
    } else {
        rqmd_mcp::serve_stdio(store).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// daemon lifecycle
// ---------------------------------------------------------------------------

/// `<cache>/mcp.pid` — the running daemon's PID.
fn pid_path() -> PathBuf {
    rqmd_core::paths::cache_dir().join("mcp.pid")
}

/// `<cache>/mcp.log` — the detached daemon's stdout/stderr.
fn log_path() -> PathBuf {
    rqmd_core::paths::cache_dir().join("mcp.log")
}

/// Start the HTTP server detached in the background. Re-spawns this executable
/// in foreground HTTP mode (re-passing `--index`, which the parent consumed) and
/// records the child PID. Mirrors qmd's `--daemon` path (`qmd.ts:3644-3711`).
fn start_daemon(args: &McpArgs, state: &mut IndexState) -> Result<()> {
    let port = args.port.unwrap_or(DEFAULT_HTTP_PORT);
    let pid_file = pid_path();

    // Reject a live daemon; clean a stale PID file so we can start fresh.
    if let Some(pid) = read_pid(&pid_file) {
        if is_alive(pid) {
            bail!("Already running (PID {pid}). Run 'rqmd mcp stop' first.");
        }
        let _ = fs::remove_file(&pid_file);
    }

    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent).ok();
    }
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
        .context("opening mcp.log")?;
    let log_err = log.try_clone().context("cloning mcp.log handle")?;

    let exe = std::env::current_exe().context("resolving rqmd executable")?;
    let mut cmd = Command::new(exe);
    // Re-pass the active index name explicitly — `--index` is a global flag
    // consumed by this parent process and would not survive arg inheritance.
    cmd.arg("--index").arg(state.index_name());
    cmd.args(["mcp", "--http", "--port"]).arg(port.to_string());
    // Detach the child's stdio to the log file so this parent's captured stdout
    // (e.g. under a test's `.output()`) closes when the parent exits, rather
    // than blocking on the long-lived child. Env is inherited by default, so the
    // child sees the same RQMD_CACHE_DIR/RQMD_CONFIG_DIR and opens the same store.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    detach(&mut cmd);

    // Windows: stop the detached child from inheriting *this* process's stdout/
    // stderr pipes (it would keep them open and hang a `.output()` launcher). The
    // child's own stdio is the log file, set above. No-op on Unix.
    make_std_handles_noninheritable();

    let child = cmd.spawn().context("spawning mcp daemon")?;
    fs::write(&pid_file, child.id().to_string()).context("writing PID file")?;
    println!(
        "Started on http://localhost:{port}/mcp (PID {})",
        child.id()
    );
    Ok(())
}

/// `rqmd mcp stop`: kill the daemon named by the PID file and remove it. A dead
/// PID is treated as a stale file (cleaned, not an error). Mirrors qmd's `stop`.
fn stop() -> Result<()> {
    let pid_file = pid_path();
    let Some(pid) = read_pid(&pid_file) else {
        println!("No MCP daemon running (no PID file).");
        return Ok(());
    };
    if is_alive(pid) {
        kill(pid);
        let _ = fs::remove_file(&pid_file);
        println!("Stopped rqmd MCP server (PID {pid}).");
    } else {
        let _ = fs::remove_file(&pid_file);
        println!("Cleaned up stale PID file (server was not running).");
    }
    Ok(())
}

fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse::<u32>().ok()
}

/// Configure `cmd` to start the child detached from this process's session /
/// console so it survives the parent exiting.
#[cfg(windows)]
fn detach(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(unix)]
fn detach(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    // New process group → detached from the controlling terminal's job control.
    cmd.process_group(0);
}

/// Clear the inherit flag on this process's std handles so a subsequently-spawned
/// child does not inherit them. Required on Windows (see the `windows-sys`
/// dependency note in Cargo.toml); a no-op everywhere else.
#[cfg(windows)]
fn make_std_handles_noninheritable() {
    use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
    use windows_sys::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    // SAFETY: GetStdHandle/SetHandleInformation are simple FFI calls; we only
    // touch the three standard handles and ignore failures on invalid ones.
    unsafe {
        for id in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            let h = GetStdHandle(id);
            if !h.is_null() {
                SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0);
            }
        }
    }
}

#[cfg(not(windows))]
fn make_std_handles_noninheritable() {}

/// True if a process with `pid` currently exists. Cross-platform via sysinfo.
fn is_alive(pid: u32) -> bool {
    use sysinfo::{Pid, System};
    let sys = System::new_all();
    sys.process(Pid::from_u32(pid)).is_some()
}

/// Terminate `pid`. Returns false if the process was not found.
fn kill(pid: u32) -> bool {
    use sysinfo::{Pid, System};
    let sys = System::new_all();
    match sys.process(Pid::from_u32(pid)) {
        Some(p) => p.kill(),
        None => false,
    }
}
