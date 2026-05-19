//! Integration tests for `rmd_core::llm::pull` (offline parts only).
//!
//! Real-network tests for `pull_models` live in PR2 behind `#[ignore]`.

use std::fs;
use std::io::Write;

use tempfile::TempDir;

use rmd_core::llm::error::Error;
use rmd_core::llm::pull::{parse_hf_uri, validate_gguf_file, HfRef};

// =============================================================================
// parse_hf_uri
// =============================================================================

#[test]
fn parses_standard_hf_uri() {
    assert_eq!(
        parse_hf_uri("hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"),
        Some(HfRef {
            repo: "Qwen/Qwen3-Embedding-0.6B-GGUF".into(),
            file: "Qwen3-Embedding-0.6B-Q8_0.gguf".into(),
        })
    );
}

#[test]
fn parses_hf_uri_with_subdir_path() {
    assert_eq!(
        parse_hf_uri("hf:user/repo/subdir/path/model.gguf"),
        Some(HfRef {
            repo: "user/repo".into(),
            file: "subdir/path/model.gguf".into(),
        })
    );
}

#[test]
fn rejects_uris_without_hf_prefix() {
    assert_eq!(parse_hf_uri("user/repo/file.gguf"), None);
    assert_eq!(parse_hf_uri("/local/path/file.gguf"), None);
    assert_eq!(parse_hf_uri("C:\\local\\path\\file.gguf"), None);
}

#[test]
fn rejects_uris_with_too_few_segments() {
    // Need at least <org>/<repo>/<file>.
    assert_eq!(parse_hf_uri("hf:user/repo"), None);
    assert_eq!(parse_hf_uri("hf:user"), None);
    assert_eq!(parse_hf_uri("hf:"), None);
}

// =============================================================================
// validate_gguf_file
// =============================================================================

fn write_fixture(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.path().join(name);
    let mut f = fs::File::create(&p).unwrap();
    f.write_all(bytes).unwrap();
    f.flush().unwrap();
    p
}

#[test]
fn accepts_valid_gguf_magic() {
    let dir = TempDir::new().unwrap();
    // GGUF magic + some padding (real files have a full header but the
    // validator only inspects the first 4 bytes for the magic).
    let mut bytes = Vec::from(b"GGUF" as &[u8]);
    bytes.extend(vec![0u8; 64]);
    let path = write_fixture(&dir, "ok.gguf", &bytes);

    validate_gguf_file(&path, "hf:fake/ok/model.gguf").unwrap();
    assert!(path.exists(), "valid GGUF file must NOT be deleted");
}

#[test]
fn rejects_html_response_and_deletes_file() {
    let dir = TempDir::new().unwrap();
    let body = b"<!DOCTYPE html>\n<html><body>Login required</body></html>";
    let path = write_fixture(&dir, "html.gguf", body);

    let err = validate_gguf_file(&path, "hf:fake/proxy/model.gguf").unwrap_err();
    match err {
        Error::InvalidGguf {
            looks_like_html,
            path: ref p,
            ..
        } => {
            assert!(looks_like_html, "should detect HTML body");
            assert_eq!(p, &path);
        }
        other => panic!("expected InvalidGguf, got {other:?}"),
    }
    assert!(!path.exists(), "HTML-poisoned file must be deleted");
}

#[test]
fn rejects_random_bytes_and_deletes_file() {
    let dir = TempDir::new().unwrap();
    let body = b"\x7fELF\x02\x01\x01\x00not actually a model just some bytes here";
    let path = write_fixture(&dir, "elf.gguf", body);

    let err = validate_gguf_file(&path, "hf:fake/elf/model.gguf").unwrap_err();
    match err {
        Error::InvalidGguf {
            looks_like_html,
            path: ref p,
            ..
        } => {
            assert!(!looks_like_html, "ELF magic is not HTML");
            assert_eq!(p, &path);
        }
        other => panic!("expected InvalidGguf, got {other:?}"),
    }
    assert!(!path.exists(), "garbage file must be deleted");
}

#[test]
fn missing_file_is_a_noop_not_an_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nope.gguf");
    assert!(!path.exists());

    // Mirrors TS: `if (!existsSync(filePath)) return;`
    validate_gguf_file(&path, "hf:fake/missing/model.gguf").unwrap();
}

#[test]
fn rejects_short_file_that_isnt_gguf() {
    // Files shorter than the magic must still be rejected — the magic
    // simply isn't present.
    let dir = TempDir::new().unwrap();
    let path = write_fixture(&dir, "tiny.gguf", b"GG");

    let err = validate_gguf_file(&path, "hf:fake/tiny/model.gguf").unwrap_err();
    matches!(err, Error::InvalidGguf { .. });
    assert!(!path.exists());
}
