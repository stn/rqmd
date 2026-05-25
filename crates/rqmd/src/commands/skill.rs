//! `rqmd skill` / `rqmd skills` — bundled runtime skill (SKILL.md) management.
//!
//! Port of qmd's skill/skills handlers (`src/cli/qmd.ts:2751-3283`). rqmd ships
//! a single skill named `rqmd`; its `SKILL.md` + `references/mcp-setup.md` are
//! embedded at compile time with `include_str!`. The identity strings
//! deliberately say "rqmd" (not "qmd") — a sanctioned parity exception, like the
//! MCP server name.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::cli::{
    SkillCmd, SkillInstallArgs, SkillsCmd, SkillsGetArgs, SkillsListArgs, SkillsPathArgs,
};
use crate::color::Palette;

/// The single bundled skill's name (and its install / search dir basename).
const SKILL_NAME: &str = "rqmd";
/// Embedded `SKILL.md` (relative to this file: `<crate>/skills/rqmd/…`). The
/// asset lives inside the crate so it ships in the published `.crate` tarball.
const SKILL_MD: &str = include_str!("../../skills/rqmd/SKILL.md");
/// Embedded supplementary reference, appended by `skills get --full`.
const MCP_SETUP_MD: &str = include_str!("../../skills/rqmd/references/mcp-setup.md");
/// Relative path used for the `--full` separator header (qmd parity).
const MCP_SETUP_REL: &str = "references/mcp-setup.md";

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// `rqmd skill <show|install>` (legacy command group).
pub fn run_skill(cmd: SkillCmd, p: &Palette) -> Result<()> {
    match cmd {
        SkillCmd::Show => {
            show(p);
            Ok(())
        }
        SkillCmd::Install(args) => install(args, p),
    }
}

/// `rqmd skills <list|get|path>`.
pub fn run_skills(cmd: SkillsCmd, _p: &Palette) -> Result<()> {
    match cmd {
        SkillsCmd::List(args) => list(args),
        SkillsCmd::Get(args) => get(args),
        SkillsCmd::Path(args) => path(args),
    }
}

/// The `--skill` top-level alias and `skill show`: print the bundled skill.
pub fn show(p: &Palette) {
    println!("{}rqmd Skill{}\n", p.bold(), p.reset());
    print!("{SKILL_MD}");
    if !SKILL_MD.ends_with('\n') {
        println!();
    }
}

// ---------------------------------------------------------------------------
// skills list / get / path
// ---------------------------------------------------------------------------

fn list(args: SkillsListArgs) -> Result<()> {
    let desc = parse_frontmatter_field(SKILL_MD, "description").unwrap_or_default();
    if args.json {
        let v = serde_json::json!({
            "success": true,
            "data": [{ "name": SKILL_NAME, "description": desc }],
        });
        println!("{}", serde_json::to_string(&v)?);
    } else {
        // qmd format: `  <name>  <description>` (name left-padded to a column;
        // rqmd has one skill, so a fixed two-space gutter suffices).
        println!("  {SKILL_NAME}  {desc}");
    }
    Ok(())
}

fn get(args: SkillsGetArgs) -> Result<()> {
    // rqmd ships a single skill, so `--all` and `get rqmd` are equivalent.
    let name = args.name.as_deref().unwrap_or(SKILL_NAME);
    if !args.all && name != SKILL_NAME {
        bail!("Skill not found: {name}");
    }

    let mut body = SKILL_MD.to_string();
    if args.full {
        // Separator format mirrors qmd: `\n--- <relativePath> ---\n`.
        body.push_str(&format!("\n--- {MCP_SETUP_REL} ---\n"));
        body.push_str(MCP_SETUP_MD);
    }

    if args.json {
        let v = serde_json::json!({
            "success": true,
            "data": { "name": SKILL_NAME, "content": body },
        });
        println!("{}", serde_json::to_string(&v)?);
    } else {
        print!("{body}");
        if !body.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

fn path(args: SkillsPathArgs) -> Result<()> {
    match args.name.as_deref() {
        None => {
            // No name: print the skills *search* directory (parent of the skill).
            let dir = skills_dir();
            let search = dir.parent().unwrap_or(&dir);
            println!("{}", search.display());
        }
        Some(n) if n == SKILL_NAME => println!("{}", skills_dir().display()),
        Some(other) => bail!("Skill not found: {other}"),
    }
    Ok(())
}

/// On-disk directory for the bundled skill. `RQMD_SKILLS_DIR` (mirrors qmd's
/// `QMD_SKILLS_DIR`) overrides; otherwise derive it from the crate's in-tree
/// `skills/` dir via `CARGO_MANIFEST_DIR` at compile time. NOTE: the fallback
/// path won't exist on an installed binary — `skills get`/`skill show` serve the
/// *embedded* content regardless, and the `skills path` test only checks the
/// path suffix.
fn skills_dir() -> PathBuf {
    if let Ok(d) = std::env::var("RQMD_SKILLS_DIR") {
        let d = d.trim();
        if !d.is_empty() {
            return PathBuf::from(d).join(SKILL_NAME);
        }
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("skills"); // <crate>/skills (bundled in the published tarball)
    p.push(SKILL_NAME);
    p
}

// ---------------------------------------------------------------------------
// skill install
// ---------------------------------------------------------------------------

fn install(args: SkillInstallArgs, p: &Palette) -> Result<()> {
    let base = if args.global {
        home_dir().ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
    } else {
        rqmd_core::store::path::pwd()
    };
    let install_dir = base.join(".agents").join("skills").join(SKILL_NAME);

    if install_dir.exists() && !args.force {
        bail!(
            "Skill already exists: {} (use --force to replace it)",
            install_dir.display()
        );
    }

    std::fs::create_dir_all(install_dir.join("references"))?;
    std::fs::write(install_dir.join("SKILL.md"), SKILL_MD)?;
    std::fs::write(install_dir.join(MCP_SETUP_REL), MCP_SETUP_MD)?;
    println!(
        "{}\u{2713}{} Installed rqmd skill to {}",
        p.green(),
        p.reset(),
        install_dir.display()
    );

    // Claude integration: `.claude/skills/rqmd` -> `.agents/skills/rqmd`.
    let claude_link = base.join(".claude").join("skills").join(SKILL_NAME);
    if args.yes {
        ensure_symlink(&install_dir, &claude_link, args.force)?;
        println!(
            "{}\u{2713}{} Linked Claude skill at {}",
            p.green(),
            p.reset(),
            claude_link.display()
        );
    } else {
        println!(
            "Tip: create a Claude symlink manually at {}",
            claude_link.display()
        );
    }
    Ok(())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Create a directory symlink `link -> target`, creating parent dirs. If `link`
/// already exists, only replace it when `force`.
fn ensure_symlink(target: &Path, link: &Path, force: bool) -> Result<()> {
    if let Some(parent) = link.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Ok(meta) = link.symlink_metadata() {
        if !force {
            return Ok(());
        }
        remove_existing(link, &meta)?;
    }
    symlink_dir(target, link)
}

/// Remove whatever is at `link` so a fresh symlink can replace it. A symlink is
/// removed *as a link* (never followed); a real directory/file is removed
/// explicitly. This avoids `remove_dir_all` ever recursing into a real
/// directory a user happened to place at the link path.
fn remove_existing(link: &Path, meta: &std::fs::Metadata) -> Result<()> {
    if meta.file_type().is_symlink() {
        // Unix: a symlink (to dir or file) is removed with `remove_file`.
        // Windows: a *directory* symlink must be removed with `remove_dir`.
        #[cfg(windows)]
        std::fs::remove_dir(link).or_else(|_| std::fs::remove_file(link))?;
        #[cfg(not(windows))]
        std::fs::remove_file(link)?;
    } else if meta.is_dir() {
        std::fs::remove_dir_all(link)?;
    } else {
        std::fs::remove_file(link)?;
    }
    Ok(())
}

#[cfg(unix)]
fn symlink_dir(target: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(windows)]
fn symlink_dir(target: &Path, link: &Path) -> Result<()> {
    // Windows directory symlinks need Developer Mode / admin; surface the OS
    // error rather than silently degrading.
    std::os::windows::fs::symlink_dir(target, link)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extract a single-line `key: value` field from a leading `---` YAML
/// frontmatter block. Returns `None` if absent. Sufficient for the bundled
/// skill's one-line `description:`.
fn parse_frontmatter_field(md: &str, key: &str) -> Option<String> {
    let mut lines = md.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let prefix = format!("{key}:");
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some(rest) = line.strip_prefix(&prefix) {
            return Some(rest.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_description_is_parsed() {
        let desc = parse_frontmatter_field(SKILL_MD, "description").expect("description");
        assert!(desc.starts_with("Search local markdown knowledge bases"));
    }

    #[test]
    fn embedded_skill_has_no_discovery_stub() {
        assert!(!SKILL_MD.contains("discovery stub"));
        assert!(SKILL_MD.contains("# rqmd \u{2014} Query Markdown Documents"));
    }

    #[test]
    fn skills_dir_ends_with_skill_name() {
        let s = skills_dir().to_string_lossy().replace('\\', "/");
        assert!(s.ends_with("skills/rqmd"), "skills_dir: {s}");
    }
}
