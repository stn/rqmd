//! Port of `tobi/qmd`'s `test/collections-config.test.ts` — config path
//! resolution via env vars (`RQMD_CONFIG_DIR`, `XDG_CONFIG_HOME`, `HOME`,
//! `USERPROFILE`) and the custom-index-name helper.

mod common;

use std::path::PathBuf;

use rqmd_core::paths::rqmd_homedir;
use rqmd_core::Config;
use serial_test::serial;

use common::{EnvGuard, PATH_ENV_KEYS};

fn config_path(config: &Config) -> PathBuf {
    PathBuf::from(config.config_path().into_owned())
}

#[test]
#[serial(env)]
fn defaults_to_home_config_rqmd_when_no_env_vars() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.remove("RQMD_CONFIG_DIR");
    guard.remove("XDG_CONFIG_HOME");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        rqmd_homedir().join(".config").join("rqmd").join("index.yml")
    );
}

#[test]
#[serial(env)]
fn falls_back_to_userprofile_when_home_unset() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.remove("HOME");
    guard.remove("RQMD_CONFIG_DIR");
    guard.remove("XDG_CONFIG_HOME");
    guard.set("USERPROFILE", "/Users/windows-user");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/Users/windows-user")
            .join(".config")
            .join("rqmd")
            .join("index.yml")
    );
}

#[test]
#[serial(env)]
fn rqmd_config_dir_takes_priority() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.set("RQMD_CONFIG_DIR", "/custom/rqmd-config");
    guard.set("XDG_CONFIG_HOME", "/xdg/config");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/custom/rqmd-config").join("index.yml")
    );
}

#[test]
#[serial(env)]
fn xdg_config_home_used_when_rqmd_config_dir_unset() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.remove("RQMD_CONFIG_DIR");
    guard.set("XDG_CONFIG_HOME", "/xdg/config");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/xdg/config").join("rqmd").join("index.yml")
    );
}

#[test]
#[serial(env)]
fn xdg_config_home_appends_rqmd_subdir() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.remove("RQMD_CONFIG_DIR");
    guard.set("XDG_CONFIG_HOME", "/home/agent/.config");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/home/agent/.config")
            .join("rqmd")
            .join("index.yml")
    );
}

#[test]
#[serial(env)]
fn rqmd_config_dir_overrides_xdg() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.set("RQMD_CONFIG_DIR", "/override");
    guard.set("XDG_CONFIG_HOME", "/should-not-use");

    let config = Config::from_default_location().unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/override").join("index.yml")
    );
}

#[test]
#[serial(env)]
fn respects_custom_index_name() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.remove("RQMD_CONFIG_DIR");
    guard.set("XDG_CONFIG_HOME", "/xdg/config");

    let config = Config::from_default_location_with_index_name("myindex").unwrap();
    assert_eq!(
        config_path(&config),
        PathBuf::from("/xdg/config").join("rqmd").join("myindex.yml")
    );
}

#[test]
#[serial(env)]
fn from_default_location_with_index_name_helper_distinct_from_from_default_location() {
    let guard = EnvGuard::capture(PATH_ENV_KEYS);
    guard.set("RQMD_CONFIG_DIR", "/cfg");

    let default = Config::from_default_location().unwrap();
    let custom = Config::from_default_location_with_index_name("project-a").unwrap();

    assert_eq!(config_path(&default), PathBuf::from("/cfg/index.yml"));
    assert_eq!(config_path(&custom), PathBuf::from("/cfg/project-a.yml"));
    assert_ne!(config_path(&default), config_path(&custom));
}
