//! Real-network test for `pull_models`.
//!
//! Run with `cargo test -p rqmd-core -- --ignored integration_pull`. The
//! first call downloads ~600 MB; the second call must hit the cache.

use rqmd_core::llm::pull::pull_models;
use rqmd_core::llm::types::PullOptions;
use tempfile::TempDir;

const EMBED_URI: &str = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";

#[test]
#[ignore = "downloads ~600 MB from HuggingFace"]
fn pull_models_downloads_then_uses_cache() {
    let tmp = TempDir::new().unwrap();
    let opts_fresh = PullOptions {
        refresh: false,
        cache_dir: Some(tmp.path().to_path_buf()),
    };

    // First call: cold cache.
    let first = pull_models(&[EMBED_URI.into()], &opts_fresh).unwrap();
    assert_eq!(first.len(), 1);
    assert!(
        first[0].size_bytes > 100_000_000,
        "expected a real GGUF, got {}",
        first[0].size_bytes
    );
    assert!(
        !first[0].refreshed,
        "fresh download should not be marked as refreshed"
    );
    let first_path = first[0].path.clone();

    // Second call: warm cache.
    let second = pull_models(&[EMBED_URI.into()], &opts_fresh).unwrap();
    assert_eq!(second.len(), 1);
    assert_eq!(
        second[0].path, first_path,
        "cache hit must return the same path"
    );
    assert!(!second[0].refreshed);

    // refresh=true with an existing snapshot should mark refreshed=true.
    let opts_refresh = PullOptions {
        refresh: true,
        cache_dir: Some(tmp.path().to_path_buf()),
    };
    let third = pull_models(&[EMBED_URI.into()], &opts_refresh).unwrap();
    assert_eq!(third.len(), 1);
    assert!(
        third[0].refreshed,
        "refresh=true on a populated cache should report refreshed=true"
    );
}
