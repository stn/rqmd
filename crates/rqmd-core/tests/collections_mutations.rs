//! Mutator semantics — proves TS-spec parity for `add_collection`,
//! `remove_context`, `update_collection_settings`, `rename_collection`,
//! the `data()` / `into_data()` accessors, and `find_context_for_path`.

use rqmd_core::collections::Error;
use rqmd_core::{
    Collection, CollectionSettings, Config, ConfigData, IncludeByDefaultField, UpdateField,
};

fn config_with_one_collection() -> Config {
    let mut data = ConfigData::default();
    data.collections.insert(
        "docs".to_string(),
        Collection {
            path: "/d1".to_string(),
            pattern: "**/*.md".to_string(),
            ignore: Some(vec!["draft/**".to_string()]),
            context: None,
            update: Some("git pull".to_string()),
            include_by_default: Some(false),
        },
    );
    Config::inline(data)
}

#[test]
fn add_collection_preserves_existing_context() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("docs", "/p1", None).unwrap();
    config.add_context("docs", "/", "shared").unwrap();
    config.add_context("docs", "/2024", "older").unwrap();

    // Re-add with a new path; context must survive.
    config
        .add_collection("docs", "/p2", Some("**/*.txt"))
        .unwrap();

    let docs = config.get_collection("docs").unwrap().collection;
    assert_eq!(docs.path, "/p2");
    assert_eq!(docs.pattern, "**/*.txt");
    let ctx = docs.context.as_ref().expect("context preserved");
    assert_eq!(ctx.get("/"), Some(&"shared".to_string()));
    assert_eq!(ctx.get("/2024"), Some(&"older".to_string()));
}

#[test]
fn add_collection_drops_ignore_update_include_by_default() {
    let mut config = config_with_one_collection();
    // Sanity: the seed values are present.
    {
        let docs = config.get_collection("docs").unwrap().collection;
        assert!(docs.ignore.is_some());
        assert!(docs.update.is_some());
        assert_eq!(docs.include_by_default, Some(false));
    }

    // Re-add with the same name: ignore/update/include_by_default all reset.
    config.add_collection("docs", "/d2", None).unwrap();

    let docs = config.get_collection("docs").unwrap().collection;
    assert_eq!(docs.path, "/d2");
    assert_eq!(docs.pattern, "**/*.md"); // default
    assert!(docs.ignore.is_none());
    assert!(docs.update.is_none());
    assert!(docs.include_by_default.is_none());
}

#[test]
fn remove_context_drops_empty_field() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("docs", "/d", None).unwrap();
    config.add_context("docs", "/", "only-entry").unwrap();
    assert!(config.contexts("docs").is_some());

    let removed = config.remove_context("docs", "/").unwrap();
    assert!(removed);

    // The map should have been cleared to None entirely.
    let docs = config.get_collection("docs").unwrap().collection;
    assert!(
        docs.context.is_none(),
        "remove_context must reset to None when empty; got {:?}",
        docs.context
    );
}

#[test]
fn remove_context_keeps_field_when_other_entries_remain() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("docs", "/d", None).unwrap();
    config.add_context("docs", "/", "a").unwrap();
    config.add_context("docs", "/sub", "b").unwrap();

    let removed = config.remove_context("docs", "/").unwrap();
    assert!(removed);

    let ctx = config.contexts("docs").expect("still present");
    assert_eq!(ctx.len(), 1);
    assert_eq!(ctx.get("/sub"), Some(&"b".to_string()));
}

#[test]
fn remove_context_returns_false_for_missing_prefix_or_collection() {
    let mut config = Config::inline(ConfigData::default());
    assert!(!config.remove_context("missing", "/").unwrap());

    config.add_collection("docs", "/d", None).unwrap();
    assert!(!config.remove_context("docs", "/nope").unwrap());
}

#[test]
fn update_collection_settings_update_keep_clear_set() {
    let mut config = config_with_one_collection();

    // Keep: leaves the existing "git pull" alone.
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Keep,
                include_by_default: IncludeByDefaultField::Keep,
            },
        )
        .unwrap();
    assert_eq!(
        config.get_collection("docs").unwrap().collection.update,
        Some("git pull".to_string())
    );

    // Set: replaces.
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Set("git fetch".to_string()),
                include_by_default: IncludeByDefaultField::Keep,
            },
        )
        .unwrap();
    assert_eq!(
        config.get_collection("docs").unwrap().collection.update,
        Some("git fetch".to_string())
    );

    // Clear: removes.
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Clear,
                include_by_default: IncludeByDefaultField::Keep,
            },
        )
        .unwrap();
    assert_eq!(
        config.get_collection("docs").unwrap().collection.update,
        None
    );
}

#[test]
fn update_collection_settings_include_by_default_three_values() {
    let mut config = config_with_one_collection();
    // Seed has include_by_default = Some(false).

    // Keep
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Keep,
                include_by_default: IncludeByDefaultField::Keep,
            },
        )
        .unwrap();
    assert_eq!(
        config
            .get_collection("docs")
            .unwrap()
            .collection
            .include_by_default,
        Some(false)
    );

    // ResetToDefault → None
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Keep,
                include_by_default: IncludeByDefaultField::ResetToDefault,
            },
        )
        .unwrap();
    assert_eq!(
        config
            .get_collection("docs")
            .unwrap()
            .collection
            .include_by_default,
        None
    );
    assert!(
        config
            .get_collection("docs")
            .unwrap()
            .collection
            .is_included_by_default()
    );

    // SetFalse → Some(false)
    config
        .update_collection_settings(
            "docs",
            CollectionSettings {
                update: UpdateField::Keep,
                include_by_default: IncludeByDefaultField::SetFalse,
            },
        )
        .unwrap();
    assert_eq!(
        config
            .get_collection("docs")
            .unwrap()
            .collection
            .include_by_default,
        Some(false)
    );
}

#[test]
fn update_collection_settings_returns_false_for_missing_collection() {
    let mut config = Config::inline(ConfigData::default());
    let ok = config
        .update_collection_settings(
            "ghost",
            CollectionSettings {
                update: UpdateField::Clear,
                include_by_default: IncludeByDefaultField::Keep,
            },
        )
        .unwrap();
    assert!(!ok);
}

#[test]
fn rename_collection_returns_false_for_missing_source() {
    let mut config = Config::inline(ConfigData::default());
    let ok = config.rename_collection("ghost", "phantom").unwrap();
    assert!(!ok);
}

#[test]
fn rename_collection_returns_err_on_duplicate() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("a", "/a", None).unwrap();
    config.add_collection("b", "/b", None).unwrap();

    let err = config.rename_collection("a", "b").unwrap_err();
    assert!(
        matches!(err, Error::DuplicateCollection(ref name) if name == "b"),
        "unexpected error variant: {err:?}"
    );

    // Originals must be untouched.
    assert!(config.get_collection("a").is_some());
    assert!(config.get_collection("b").is_some());
}

#[test]
fn rename_collection_moves_entry() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("old", "/o", None).unwrap();
    assert!(config.rename_collection("old", "new").unwrap());
    assert!(config.get_collection("old").is_none());
    assert!(config.get_collection("new").is_some());
}

#[test]
fn data_accessor_exposes_full_config_data() {
    let mut config = Config::inline(ConfigData::default());
    config.set_global_context(Some("g".into())).unwrap();
    config.add_collection("a", "/a", None).unwrap();
    config.add_collection("b", "/b", None).unwrap();

    let data = config.data();
    assert_eq!(data.global_context.as_deref(), Some("g"));
    assert_eq!(data.collections.len(), config.list_collections().len());
    assert_eq!(config.global_context(), data.global_context.as_deref());
}

#[test]
fn into_data_consumes_self() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("a", "/a", None).unwrap();
    let data: ConfigData = config.into_data();
    assert_eq!(data.collections.len(), 1);
    assert!(data.collections.contains_key("a"));
}

#[test]
fn find_context_for_path_longest_prefix_wins() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("notes", "/n", None).unwrap();
    config.add_context("notes", "/", "root").unwrap();
    config.add_context("notes", "/2024", "year").unwrap();
    config
        .add_context("notes", "/2024/board", "board-2024")
        .unwrap();

    assert_eq!(
        config.find_context_for_path("notes", "/2024/board/jan.md"),
        Some("board-2024")
    );
    assert_eq!(
        config.find_context_for_path("notes", "/2024/random.md"),
        Some("year")
    );
    assert_eq!(
        config.find_context_for_path("notes", "/misc/x.md"),
        Some("root")
    );
}

#[test]
fn find_context_for_path_falls_back_to_global() {
    let mut config = Config::inline(ConfigData::default());
    config.add_collection("notes", "/n", None).unwrap();
    config.set_global_context(Some("global!".into())).unwrap();

    // No per-collection context at all → global.
    assert_eq!(
        config.find_context_for_path("notes", "/anything.md"),
        Some("global!")
    );

    // With contexts present but none matching → still global.
    config.add_context("notes", "/2024", "year").unwrap();
    assert_eq!(
        config.find_context_for_path("notes", "/2023/x.md"),
        Some("global!")
    );

    // Unknown collection → global.
    assert_eq!(
        config.find_context_for_path("ghost", "/x.md"),
        Some("global!")
    );
}
