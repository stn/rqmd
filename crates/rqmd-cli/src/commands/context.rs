//! `rqmd context ...` — attach human-written context summaries.
//!
//! Maps to qmd's `context` subcommands in `src/cli/qmd.ts`
//! (lines 727–917, 3292–3366).

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use rqmd_core::store::path::{homedir, pwd, real_path};
use rqmd_core::store::virtual_path::{is_virtual_path, parse_virtual_path};

use crate::cli::{ContextAddArgs, ContextCmd, ContextRmArgs};
use crate::color::Palette;
use crate::state::IndexState;

pub fn run(cmd: ContextCmd, state: &mut IndexState, p: &Palette) -> Result<()> {
    match cmd {
        ContextCmd::Add(a) => add(a, state, p),
        ContextCmd::List => list(state, p),
        ContextCmd::Rm(a) => remove(a, state, p),
    }
}

/// Resolve qmd's "path or context text" ambiguity:
///   `context add "text"`              → no path (use cwd), text=args[0]
///   `context add path "text" ...`     → path=args[0], text=args[1..].join(" ")
fn split_path_and_text(args: &[String]) -> (Option<String>, String) {
    match args.len() {
        0 => (None, String::new()),
        1 => (None, args[0].clone()),
        _ => (Some(args[0].clone()), args[1..].join(" ")),
    }
}

fn add(a: ContextAddArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    let (path_arg, text) = split_path_and_text(&a.args);
    if text.is_empty() {
        return Err(anyhow!(
            "Usage: rqmd context add [path] \"text\"\n\n\
             Examples:\n  \
             rqmd context add \"Context for current directory\"\n  \
             rqmd context add . \"Context for current directory\"\n  \
             rqmd context add qmd://notes/ \"Context for entire notes collection\"\n  \
             rqmd context add / \"Global context for all collections\""
        ));
    }

    // "/" → global context.
    if path_arg.as_deref() == Some("/") {
        let cfg = state.config_mut()?;
        cfg.set_global_context(Some(text.clone()))?;
        state.resync_config()?;
        println!("{}✓{} Set global context", p.green(), p.reset());
        println!("{}Context: {text}{}", p.dim(), p.reset());
        return Ok(());
    }

    let fs_path = path_arg
        .as_deref()
        .map(normalize_fs_path)
        .unwrap_or_else(|| pwd().to_string_lossy().to_string());

    // qmd://... virtual path.
    if is_virtual_path(&fs_path) {
        let vp = parse_virtual_path(&fs_path)
            .map_err(|_| anyhow!("{}Invalid virtual path: {fs_path}{}", p.yellow(), p.reset()))?;
        let cfg = state.config_mut()?;
        let ok = cfg.add_context(&vp.collection, &vp.path, text.clone())?;
        if !ok {
            return Err(anyhow!(
                "{}Collection not found: {}{}",
                p.yellow(),
                vp.collection,
                p.reset()
            ));
        }
        state.resync_config()?;
        let display = if vp.path.is_empty() {
            format!("qmd://{}/ (collection root)", vp.collection)
        } else {
            format!("qmd://{}/{}", vp.collection, vp.path)
        };
        println!("{}✓{} Added context for: {display}", p.green(), p.reset());
        println!("{}Context: {text}{}", p.dim(), p.reset());
        return Ok(());
    }

    // Filesystem path → detect owning collection.
    let detected = detect_collection_from_path(state, &fs_path)?.ok_or_else(|| {
        anyhow!(
            "{}Path is not in any indexed collection: {fs_path}{}\n{}Run 'rqmd status' to see indexed collections{}",
            p.yellow(),
            p.reset(),
            p.dim(),
            p.reset()
        )
    })?;

    let cfg = state.config_mut()?;
    let ok = cfg.add_context(
        &detected.collection_name,
        &detected.relative_path,
        text.clone(),
    )?;
    if !ok {
        return Err(anyhow!(
            "{}Collection not found: {}{}",
            p.yellow(),
            detected.collection_name,
            p.reset()
        ));
    }
    state.resync_config()?;
    let display = if detected.relative_path.is_empty() {
        format!("qmd://{}/", detected.collection_name)
    } else {
        format!(
            "qmd://{}/{}",
            detected.collection_name, detected.relative_path
        )
    };
    println!("{}✓{} Added context for: {display}", p.green(), p.reset());
    println!("{}Context: {text}{}", p.dim(), p.reset());
    Ok(())
}

fn list(state: &mut IndexState, p: &Palette) -> Result<()> {
    let cfg = state.config_mut()?;
    let mut entries: Vec<(String, String, String)> = cfg
        .list_all_contexts()
        .into_iter()
        .map(|e| {
            (
                e.collection.to_string(),
                e.path.to_string(),
                e.context.to_string(),
            )
        })
        .collect();
    if entries.is_empty() {
        println!(
            "{}No contexts configured. Use 'rqmd context add' to add one.{}",
            p.dim(),
            p.reset()
        );
        return Ok(());
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    println!("\n{}Configured Contexts{}\n", p.bold(), p.reset());
    let mut last = String::new();
    for (collection, path, context) in &entries {
        if collection != &last {
            println!("{}{collection}{}", p.cyan(), p.reset());
            last = collection.clone();
        }
        let display_path = if path.is_empty() {
            "  / (root)".to_string()
        } else {
            format!("  {path}")
        };
        println!("{display_path}");
        println!("    {}{context}{}", p.dim(), p.reset());
    }
    Ok(())
}

fn remove(a: ContextRmArgs, state: &mut IndexState, p: &Palette) -> Result<()> {
    if a.path == "/" {
        let cfg = state.config_mut()?;
        cfg.set_global_context(None)?;
        state.resync_config()?;
        println!("{}✓{} Removed global context", p.green(), p.reset());
        return Ok(());
    }

    if is_virtual_path(&a.path) {
        let vp = parse_virtual_path(&a.path).map_err(|_| {
            anyhow!(
                "{}Invalid virtual path: {}{}",
                p.yellow(),
                a.path,
                p.reset()
            )
        })?;
        let cfg = state.config_mut()?;
        let ok = cfg.remove_context(&vp.collection, &vp.path)?;
        if !ok {
            return Err(anyhow!(
                "{}No context found for: {}{}",
                p.yellow(),
                a.path,
                p.reset()
            ));
        }
        state.resync_config()?;
        println!(
            "{}✓{} Removed context for: {}",
            p.green(),
            p.reset(),
            a.path
        );
        return Ok(());
    }

    let fs_path = normalize_fs_path(&a.path);
    let detected = detect_collection_from_path(state, &fs_path)?.ok_or_else(|| {
        anyhow!(
            "{}Path is not in any indexed collection: {fs_path}{}",
            p.yellow(),
            p.reset()
        )
    })?;

    let cfg = state.config_mut()?;
    let ok = cfg.remove_context(&detected.collection_name, &detected.relative_path)?;
    if !ok {
        return Err(anyhow!(
            "{}No context found for: qmd://{}/{}{}",
            p.yellow(),
            detected.collection_name,
            detected.relative_path,
            p.reset()
        ));
    }
    state.resync_config()?;
    println!(
        "{}✓{} Removed context for: qmd://{}/{}",
        p.green(),
        p.reset(),
        detected.collection_name,
        detected.relative_path
    );
    Ok(())
}

// ----- helpers -----

fn normalize_fs_path(input: &str) -> String {
    if input == "." || input == "./" {
        return pwd().to_string_lossy().to_string();
    }
    if let Some(rest) = input.strip_prefix("~/") {
        return homedir().join(rest).to_string_lossy().to_string();
    }
    if input.starts_with('/') || is_virtual_path(input) {
        return input.to_string();
    }
    real_path(&pwd().join(input)).to_string_lossy().to_string()
}

struct DetectedCollection {
    collection_name: String,
    relative_path: String,
}

/// qmd's `detectCollectionFromPath` (qmd.ts:727–757). Longest-prefix match
/// against the YAML collection roots.
fn detect_collection_from_path(
    state: &mut IndexState,
    fs_path: &str,
) -> Result<Option<DetectedCollection>> {
    let real = real_path(Path::new(fs_path));
    let real_str = real.to_string_lossy().replace('\\', "/");

    let cfg = state.config_mut()?;
    let mut best: Option<(String, PathBuf)> = None;
    for c in cfg.list_collections() {
        let base = c.collection.path.replace('\\', "/");
        let base_with = if base.ends_with('/') {
            base.clone()
        } else {
            format!("{base}/")
        };
        if real_str == base || real_str.starts_with(&base_with) {
            let better = best
                .as_ref()
                .map(|(_, bp)| base.len() > bp.to_string_lossy().len())
                .unwrap_or(true);
            if better {
                best = Some((c.name.to_string(), PathBuf::from(&base)));
            }
        }
    }

    let Some((name, base)) = best else {
        return Ok(None);
    };
    let base_str = base.to_string_lossy().replace('\\', "/");
    let rel = if real_str == base_str {
        String::new()
    } else {
        real_str[base_str.len() + 1..].to_string()
    };
    Ok(Some(DetectedCollection {
        collection_name: name,
        relative_path: rel,
    }))
}
