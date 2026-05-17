//! Port of the non-CLI cases from `tobi/qmd`'s `test/local-config.test.ts`.
//! Covers `find_local_config_path` and `local_db_path`.

use std::fs;

use rmd_core::{find_local_config_path, local_db_path};
use tempfile::TempDir;

#[test]
fn finds_local_config_from_nested_directory() {
    let root = TempDir::new().unwrap();
    let rmd_dir = root.path().join(".rmd");
    fs::create_dir_all(&rmd_dir).unwrap();
    let config_path = rmd_dir.join("index.yaml");
    fs::write(&config_path, "collections: {}\n").unwrap();

    let nested = root.path().join("wiki").join("shopify");
    fs::create_dir_all(&nested).unwrap();

    let found = find_local_config_path(&nested).expect("expected to find config");
    assert_eq!(found, config_path);
}

#[test]
fn prefers_yaml_over_yml_when_both_exist() {
    let root = TempDir::new().unwrap();
    let rmd_dir = root.path().join(".rmd");
    fs::create_dir_all(&rmd_dir).unwrap();
    let yaml = rmd_dir.join("index.yaml");
    let yml = rmd_dir.join("index.yml");
    fs::write(&yaml, "collections: {}\n").unwrap();
    fs::write(&yml, "collections: {}\n").unwrap();

    let found = find_local_config_path(root.path()).expect("expected to find config");
    assert_eq!(found, yaml);
}

#[test]
fn falls_back_to_yml_when_only_yml_exists() {
    let root = TempDir::new().unwrap();
    let rmd_dir = root.path().join(".rmd");
    fs::create_dir_all(&rmd_dir).unwrap();
    let yml = rmd_dir.join("index.yml");
    fs::write(&yml, "collections: {}\n").unwrap();

    let found = find_local_config_path(root.path()).expect("expected to find config");
    assert_eq!(found, yml);
}

#[test]
fn returns_none_when_no_local_config_present() {
    let root = TempDir::new().unwrap();
    let nested = root.path().join("deep").join("dir");
    fs::create_dir_all(&nested).unwrap();

    // Walks all the way up to the filesystem root without finding `.rmd`.
    // (Even if the user happens to have a `.rmd` somewhere up the tree on
    // their dev machine, the tempdir-rooted path will be inside the tempdir,
    // so any match must be inside `root`.) Restrict the assertion to paths
    // that are *not* inside `root` — those would indicate a real bug.
    let found = find_local_config_path(&nested);
    if let Some(p) = &found {
        assert!(
            !p.starts_with(root.path()),
            "found unexpected config inside tempdir: {p:?}"
        );
    }
}

#[test]
fn local_db_path_is_sibling_of_config() {
    let root = TempDir::new().unwrap();
    let rmd_dir = root.path().join(".rmd");
    fs::create_dir_all(&rmd_dir).unwrap();
    let config_path = rmd_dir.join("index.yaml");
    fs::write(&config_path, "collections: {}\n").unwrap();

    assert_eq!(local_db_path(&config_path), rmd_dir.join("index.sqlite"));
}
