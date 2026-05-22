//! Collection-filter resolution shared by `search`, `vsearch`, and `query`.
//!
//! Faithful port of qmd's `resolveCollectionFilter` / `filterByCollections`
//! (`src/cli/qmd.ts` lines 2272-2300). When `-c` is omitted the search is
//! scoped to the default collections (`includeByDefault !== false`) rather
//! than every collection; explicit names are validated up front.

use anyhow::{Result, bail};
use rqmd_core::collections::Config;

/// Port of `resolveCollectionFilter(raw, useDefaults)` (qmd.ts:2272).
///
/// * `raw` empty + `use_defaults` → the default collection names (YAML config).
/// * `raw` empty + `!use_defaults` → `[]`.
/// * `raw` non-empty → each name is validated against the config; an unknown
///   name fails with `Collection not found: <name>` (TS `console.error` +
///   `exit(1)`).
pub fn resolve_collection_filter(
    config: &Config,
    raw: &[String],
    use_defaults: bool,
) -> Result<Vec<String>> {
    if raw.is_empty() {
        if use_defaults {
            return Ok(config
                .default_collection_names()
                .into_iter()
                .map(str::to_string)
                .collect());
        }
        return Ok(Vec::new());
    }
    let mut validated = Vec::with_capacity(raw.len());
    for name in raw {
        if config.get_collection(name).is_none() {
            bail!("Collection not found: {name}");
        }
        validated.push(name.clone());
    }
    Ok(validated)
}

/// The collection to push into the DB-level filter, mirroring TS
/// `singleCollection = names.length === 1 ? names[0] : undefined`. With more
/// than one collection the DB query stays unfiltered and
/// [`filter_by_collections`] narrows the results afterwards.
pub fn single_collection(names: &[String]) -> Option<String> {
    match names {
        [one] => Some(one.clone()),
        _ => None,
    }
}

/// Port of `filterByCollections` (qmd.ts:2293). A `<= 1` collection list is a
/// no-op (the DB-level filter already handled it); otherwise keep only results
/// whose virtual path falls under one of the collections.
///
/// The prefix carries a trailing `/` (`qmd://<name>/`), so `foo` never matches
/// `foobar/...`; collection names are restricted to alphanumerics/`-`/`_` and
/// cannot contain `/`, so this is unambiguous (TS uses the same scheme).
pub fn filter_by_collections<T>(
    results: Vec<T>,
    names: &[String],
    file_of: impl Fn(&T) -> &str,
) -> Vec<T> {
    if names.len() <= 1 {
        return results;
    }
    let prefixes: Vec<String> = names.iter().map(|n| format!("qmd://{n}/")).collect();
    results
        .into_iter()
        .filter(|r| {
            let f = file_of(r);
            prefixes.iter().any(|p| f.starts_with(p.as_str()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rqmd_core::{CollectionSettings, Config, IncludeByDefaultField};
    use tempfile::TempDir;

    struct Rec {
        file: String,
    }

    fn rec(file: &str) -> Rec {
        Rec {
            file: file.to_string(),
        }
    }

    // ── single_collection ──

    #[test]
    fn single_collection_picks_only_when_exactly_one() {
        assert_eq!(single_collection(&[]), None);
        assert_eq!(single_collection(&["a".into()]), Some("a".into()));
        assert_eq!(single_collection(&["a".into(), "b".into()]), None);
    }

    // ── filter_by_collections ──

    #[test]
    fn filter_is_noop_for_zero_or_one_collection() {
        let r = vec![rec("qmd://docs/x.md"), rec("qmd://notes/y.md")];
        let out = filter_by_collections(r, &["docs".into()], |r| r.file.as_str());
        // <= 1 collection → DB already filtered, so nothing is dropped here.
        assert_eq!(out.len(), 2);

        // An empty list is also a no-op (`names.len() <= 1`, matching TS `length <= 1`).
        let r = vec![rec("qmd://docs/x.md"), rec("qmd://notes/y.md")];
        assert_eq!(filter_by_collections(r, &[], |r| r.file.as_str()).len(), 2);
    }

    #[test]
    fn filter_keeps_only_listed_collections() {
        let r = vec![
            rec("qmd://docs/x.md"),
            rec("qmd://notes/y.md"),
            rec("qmd://archive/z.md"),
        ];
        let out = filter_by_collections(r, &["docs".into(), "notes".into()], |r| r.file.as_str());
        let files: Vec<&str> = out.iter().map(|r| r.file.as_str()).collect();
        assert_eq!(files, vec!["qmd://docs/x.md", "qmd://notes/y.md"]);
    }

    #[test]
    fn filter_keeps_matching_collections_preserving_order() {
        // Mirrors qmd test "filters to matching collections when multiple specified":
        // `docs` appears twice and the original order is preserved.
        let r = vec![
            rec("qmd://docs/readme.md"),
            rec("qmd://notes/todo.md"),
            rec("qmd://journals/2024/jan.md"),
            rec("qmd://docs/api.md"),
        ];
        let out =
            filter_by_collections(r, &["docs".into(), "journals".into()], |r| r.file.as_str());
        let files: Vec<&str> = out.iter().map(|r| r.file.as_str()).collect();
        assert_eq!(
            files,
            vec![
                "qmd://docs/readme.md",
                "qmd://journals/2024/jan.md",
                "qmd://docs/api.md",
            ]
        );
    }

    #[test]
    fn filter_two_collections_non_adjacent() {
        // Mirrors qmd test "filters correctly with two collections": the two
        // selected collections are not adjacent in input order.
        let r = vec![
            rec("qmd://docs/readme.md"),
            rec("qmd://notes/todo.md"),
            rec("qmd://journals/2024/jan.md"),
            rec("qmd://docs/api.md"),
        ];
        let out =
            filter_by_collections(r, &["notes".into(), "journals".into()], |r| r.file.as_str());
        let files: Vec<&str> = out.iter().map(|r| r.file.as_str()).collect();
        assert_eq!(
            files,
            vec!["qmd://notes/todo.md", "qmd://journals/2024/jan.md"]
        );
    }

    #[test]
    fn filter_returns_empty_when_none_match() {
        // Mirrors qmd test "returns empty when no results match collections".
        let r = vec![
            rec("qmd://docs/readme.md"),
            rec("qmd://notes/todo.md"),
            rec("qmd://journals/2024/jan.md"),
            rec("qmd://docs/api.md"),
        ];
        let out =
            filter_by_collections(r, &["archive".into(), "trash".into()], |r| r.file.as_str());
        assert!(out.is_empty());
    }

    #[test]
    fn filter_prefix_does_not_collide_on_name_substring() {
        // `foo` must not match `foobar` thanks to the trailing slash.
        let r = vec![rec("qmd://foo/a.md"), rec("qmd://foobar/b.md")];
        let out = filter_by_collections(r, &["foo".into(), "baz".into()], |r| r.file.as_str());
        let files: Vec<&str> = out.iter().map(|r| r.file.as_str()).collect();
        assert_eq!(files, vec!["qmd://foo/a.md"]);
    }

    // ── resolve_collection_filter ──

    fn config_with_collections() -> (TempDir, Config) {
        let tmp = TempDir::new().unwrap();
        let mut config = Config::from_file(tmp.path().join("index.yml")).unwrap();
        config.add_collection("docs", "/docs", None).unwrap();
        config.add_collection("notes", "/notes", None).unwrap();
        config.add_collection("archive", "/archive", None).unwrap();
        // Exclude `archive` from default searches.
        config
            .update_collection_settings(
                "archive",
                CollectionSettings {
                    include_by_default: IncludeByDefaultField::SetFalse,
                    ..Default::default()
                },
            )
            .unwrap();
        (tmp, config)
    }

    #[test]
    fn resolve_uses_defaults_when_empty_and_skips_excluded() {
        let (_tmp, config) = config_with_collections();
        let names = resolve_collection_filter(&config, &[], true).unwrap();
        assert_eq!(names, vec!["docs".to_string(), "notes".to_string()]);
    }

    #[test]
    fn resolve_empty_without_defaults_is_empty() {
        let (_tmp, config) = config_with_collections();
        assert!(
            resolve_collection_filter(&config, &[], false)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn resolve_validates_explicit_name() {
        let (_tmp, config) = config_with_collections();
        // An excluded collection is still a valid explicit choice.
        let names = resolve_collection_filter(&config, &["archive".into()], true).unwrap();
        assert_eq!(names, vec!["archive".to_string()]);
    }

    #[test]
    fn resolve_errors_on_unknown_name() {
        let (_tmp, config) = config_with_collections();
        let err = resolve_collection_filter(&config, &["nope".into()], true).unwrap_err();
        assert_eq!(err.to_string(), "Collection not found: nope");
    }
}
