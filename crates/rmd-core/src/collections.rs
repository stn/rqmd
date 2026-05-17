//! Collections configuration management.
//!
//! Port of `tobi/qmd`'s `src/collections.ts` into Rust. Manages the YAML
//! configuration that defines which directories are indexed, their per-path
//! contexts, and the model bindings used for embedding / reranking / generation.
//!
//! Unlike the TypeScript original — which relied on mutable module-level
//! state — the Rust API exposes an owned [`Config`] struct that the caller
//! holds onto and passes around explicitly.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::paths;

// ============================================================================
// Data model
// ============================================================================

/// The complete configuration document.
///
/// Mirrors `CollectionConfig` in `collections.ts`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,

    /// `editor_uri_template` is accepted as an alias on deserialize; on save
    /// the canonical `editor_uri` key is always written.
    #[serde(
        default,
        alias = "editor_uri_template",
        skip_serializing_if = "Option::is_none"
    )]
    pub editor_uri: Option<String>,

    #[serde(default, deserialize_with = "deserialize_null_as_empty_map")]
    pub collections: IndexMap<String, Collection>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelsConfig>,
}

/// A single collection — a filesystem root + glob + optional context map.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Collection {
    pub path: String,
    pub pattern: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextMap>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,

    /// `None` and `Some(true)` are equivalent: the collection is included
    /// by default. `Some(true)` is normalized away on save to match the
    /// TS behaviour of `delete collection.includeByDefault`.
    #[serde(
        rename = "includeByDefault",
        alias = "include_by_default",
        default,
        skip_serializing_if = "is_none_or_true"
    )]
    pub include_by_default: Option<bool>,
}

impl Collection {
    /// Returns the effective `include_by_default` value (defaults to true).
    pub fn is_included_by_default(&self) -> bool {
        !matches!(self.include_by_default, Some(false))
    }
}

/// Model bindings — optional GGUF model names for the LLM stack.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<String>,
}

/// Per-path contexts inside a collection. The key is a path prefix
/// (e.g. `/`, `/2024`); the value is a free-form context string.
pub type ContextMap = IndexMap<String, String>;

fn is_none_or_true(v: &Option<bool>) -> bool {
    matches!(v, None | Some(true))
}

/// Coerce an explicit YAML `null` for the `collections` key into an empty map.
///
/// `#[serde(default)]` alone handles a missing key but not `collections: null`.
fn deserialize_null_as_empty_map<'de, D>(
    d: D,
) -> std::result::Result<IndexMap<String, Collection>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<IndexMap<String, Collection>>::deserialize(d).map(|o| o.unwrap_or_default())
}

// ============================================================================
// Borrowed views
// ============================================================================

/// A [`Collection`] paired with its name, returned from `get_collection`,
/// `list_collections`, etc.
#[derive(Debug, Clone, Copy)]
pub struct NamedCollectionRef<'a> {
    pub name: &'a str,
    pub collection: &'a Collection,
}

/// One entry from `list_all_contexts`. Global context uses `collection = "*"`
/// and `path = "/"` to match the TS layout.
#[derive(Debug, Clone, Copy)]
pub struct ContextEntry<'a> {
    pub collection: &'a str,
    pub path: &'a str,
    pub context: &'a str,
}

// ============================================================================
// Settings tri-states
// ============================================================================

/// Settings argument to [`Config::update_collection_settings`]. Each field
/// distinguishes "leave untouched" / "clear" / "set" so callers can express
/// the full TS tri-state semantics.
#[derive(Debug, Clone, Default)]
pub struct CollectionSettings {
    pub update: UpdateField,
    pub include_by_default: IncludeByDefaultField,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum UpdateField {
    /// Don't touch (TS `undefined`).
    #[default]
    Keep,
    /// Remove the field (TS `null`).
    Clear,
    /// Set the field (TS `string`).
    Set(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum IncludeByDefaultField {
    /// Don't touch (TS `undefined`).
    #[default]
    Keep,
    /// Remove the field — falls back to the default `true` (TS `=== true`).
    ResetToDefault,
    /// Persist `false` (TS `=== false`).
    SetFalse,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse YAML at {path}: {source}", path = path.display())]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_norway::Error,
    },

    #[error("failed to serialize YAML: {0}")]
    YamlSerialize(#[source] serde_norway::Error),

    #[error("collection '{0}' already exists")]
    DuplicateCollection(String),

    #[error("invalid collection name: '{0}'")]
    InvalidCollectionName(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// Config
// ============================================================================

/// Configuration source — either a file on disk or an in-memory document.
#[derive(Debug, Clone)]
enum Source {
    /// File-backed. `path = None` means "derive from `index_name` + env vars
    /// at every I/O call" (so env mutations between calls are picked up).
    File { path: Option<PathBuf> },
    /// In-memory only. `save()` is a no-op.
    Inline,
}

/// The owned configuration. Holds the parsed [`ConfigData`] plus enough
/// metadata to know where to reload from / save to.
#[derive(Debug, Clone)]
pub struct Config {
    inner: ConfigData,
    source: Source,
    index_name: String,
}

impl Config {
    // ── Constructors ──

    /// Load a configuration from an explicit file path. Returns an empty
    /// config if the file does not exist (matches TS `loadConfig`).
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();
        let inner = read_yaml_if_exists(&path)?;
        Ok(Self {
            inner,
            source: Source::File { path: Some(path) },
            index_name: "index".into(),
        })
    }

    /// Load from the default config location (`<config_dir>/index.yml`).
    pub fn from_default_location() -> Result<Self> {
        Self::from_default_location_with_index_name("index")
    }

    /// Load from the default config directory with a custom index name.
    /// The name is sanitized (path-like inputs are flattened to a single
    /// filename component, including on Windows).
    pub fn from_default_location_with_index_name(name: impl Into<String>) -> Result<Self> {
        let name = sanitize_index_name(&name.into());
        let path = paths::config_file_path(&name);
        let inner = read_yaml_if_exists(&path)?;
        Ok(Self {
            inner,
            source: Source::File { path: None },
            index_name: name,
        })
    }

    /// Alias for [`Self::from_default_location_with_index_name`].
    pub fn from_index_name(name: impl Into<String>) -> Result<Self> {
        Self::from_default_location_with_index_name(name)
    }

    /// Wrap an in-memory [`ConfigData`]. `save()` becomes a no-op.
    pub fn inline(data: ConfigData) -> Self {
        Self {
            inner: data,
            source: Source::Inline,
            index_name: "index".into(),
        }
    }

    /// Convenience for `inline(ConfigData::default())`.
    pub fn empty() -> Self {
        Self::inline(ConfigData::default())
    }

    // ── I/O ──

    /// Re-read the file (no-op for inline configs).
    pub fn reload(&mut self) -> Result<()> {
        let Some(path) = self.effective_path() else {
            return Ok(());
        };
        self.inner = read_yaml_if_exists(&path)?;
        Ok(())
    }

    /// Write the YAML back out. Creates parent directories as needed.
    /// No-op when the source is `Source::Inline`.
    pub fn save(&self) -> Result<()> {
        let Some(path) = self.effective_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).map_err(|e| Error::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let yaml = serde_norway::to_string(&self.inner).map_err(Error::YamlSerialize)?;
        std::fs::write(&path, yaml).map_err(|e| Error::Io { path, source: e })?;
        Ok(())
    }

    /// Display string for the active config location. Returns `"<inline>"`
    /// for in-memory configs.
    pub fn config_path(&self) -> Cow<'_, str> {
        match &self.source {
            Source::Inline => Cow::Borrowed("<inline>"),
            Source::File { path: Some(p) } => Cow::Owned(p.to_string_lossy().into_owned()),
            Source::File { path: None } => Cow::Owned(
                paths::config_file_path(&self.index_name)
                    .to_string_lossy()
                    .into_owned(),
            ),
        }
    }

    /// `true` if the config exists. Inline configs always return `true`.
    pub fn config_exists(&self) -> bool {
        match &self.source {
            Source::Inline => true,
            _ => self.effective_path().is_some_and(|p| p.exists()),
        }
    }

    // ── Raw data accessor ──

    /// Borrow the full [`ConfigData`]. Useful for downstream code (e.g. the
    /// future `store` module) that needs to consume the whole document.
    pub fn data(&self) -> &ConfigData {
        &self.inner
    }

    /// Take ownership of the underlying [`ConfigData`], consuming the Config.
    pub fn into_data(self) -> ConfigData {
        self.inner
    }

    // ── Collections ──

    pub fn get_collection(&self, name: &str) -> Option<NamedCollectionRef<'_>> {
        self.inner
            .collections
            .get_key_value(name)
            .map(|(k, v)| NamedCollectionRef {
                name: k.as_str(),
                collection: v,
            })
    }

    pub fn list_collections(&self) -> Vec<NamedCollectionRef<'_>> {
        self.inner
            .collections
            .iter()
            .map(|(k, v)| NamedCollectionRef {
                name: k.as_str(),
                collection: v,
            })
            .collect()
    }

    pub fn default_collections(&self) -> Vec<NamedCollectionRef<'_>> {
        self.inner
            .collections
            .iter()
            .filter(|(_, c)| c.is_included_by_default())
            .map(|(k, v)| NamedCollectionRef {
                name: k.as_str(),
                collection: v,
            })
            .collect()
    }

    pub fn default_collection_names(&self) -> Vec<&str> {
        self.inner
            .collections
            .iter()
            .filter(|(_, c)| c.is_included_by_default())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Add or replace a collection.
    ///
    /// **This is destructive.** If `name` already exists, only the `context`
    /// map is carried over to the new entry — `ignore`, `update`, and
    /// `include_by_default` are reset to defaults. This matches TS
    /// `addCollection` in `collections.ts`.
    pub fn add_collection(
        &mut self,
        name: &str,
        path: impl Into<String>,
        pattern: Option<&str>,
    ) -> Result<()> {
        let context = self
            .inner
            .collections
            .get(name)
            .and_then(|c| c.context.clone());
        let collection = Collection {
            path: path.into(),
            pattern: pattern.unwrap_or("**/*.md").to_string(),
            ignore: None,
            context,
            update: None,
            include_by_default: None,
        };
        self.inner.collections.insert(name.to_string(), collection);
        self.save()
    }

    /// Remove a collection. Returns `false` if it didn't exist.
    pub fn remove_collection(&mut self, name: &str) -> Result<bool> {
        if self.inner.collections.shift_remove(name).is_none() {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }

    /// Rename `old` to `new`. Returns `false` if `old` did not exist, and
    /// [`Error::DuplicateCollection`] if `new` already exists.
    pub fn rename_collection(&mut self, old: &str, new: &str) -> Result<bool> {
        if !self.inner.collections.contains_key(old) {
            return Ok(false);
        }
        if self.inner.collections.contains_key(new) {
            return Err(Error::DuplicateCollection(new.to_string()));
        }
        let collection = self.inner.collections.shift_remove(old).expect("checked");
        self.inner.collections.insert(new.to_string(), collection);
        self.save()?;
        Ok(true)
    }

    /// Apply [`CollectionSettings`] to a named collection.
    pub fn update_collection_settings(
        &mut self,
        name: &str,
        s: CollectionSettings,
    ) -> Result<bool> {
        {
            let Some(collection) = self.inner.collections.get_mut(name) else {
                return Ok(false);
            };
            match s.update {
                UpdateField::Keep => {}
                UpdateField::Clear => collection.update = None,
                UpdateField::Set(v) => collection.update = Some(v),
            }
            match s.include_by_default {
                IncludeByDefaultField::Keep => {}
                IncludeByDefaultField::ResetToDefault => collection.include_by_default = None,
                IncludeByDefaultField::SetFalse => collection.include_by_default = Some(false),
            }
        }
        self.save()?;
        Ok(true)
    }

    // ── Context ──

    pub fn global_context(&self) -> Option<&str> {
        self.inner.global_context.as_deref()
    }

    pub fn set_global_context(&mut self, ctx: Option<String>) -> Result<()> {
        self.inner.global_context = ctx;
        self.save()
    }

    pub fn contexts(&self, collection: &str) -> Option<&ContextMap> {
        self.inner
            .collections
            .get(collection)
            .and_then(|c| c.context.as_ref())
    }

    pub fn add_context(
        &mut self,
        collection: &str,
        prefix: &str,
        text: impl Into<String>,
    ) -> Result<bool> {
        {
            let Some(c) = self.inner.collections.get_mut(collection) else {
                return Ok(false);
            };
            c.context
                .get_or_insert_with(IndexMap::new)
                .insert(prefix.to_string(), text.into());
        }
        self.save()?;
        Ok(true)
    }

    /// Remove a context entry. When the last entry is removed, the
    /// `context` field is cleared entirely so the YAML output omits it
    /// (matches TS `removeContext`).
    pub fn remove_context(&mut self, collection: &str, prefix: &str) -> Result<bool> {
        {
            let Some(c) = self.inner.collections.get_mut(collection) else {
                return Ok(false);
            };
            let Some(map) = c.context.as_mut() else {
                return Ok(false);
            };
            if map.shift_remove(prefix).is_none() {
                return Ok(false);
            }
            if map.is_empty() {
                c.context = None;
            }
        }
        self.save()?;
        Ok(true)
    }

    pub fn list_all_contexts(&self) -> Vec<ContextEntry<'_>> {
        let mut out = Vec::new();
        if let Some(g) = self.inner.global_context.as_deref() {
            out.push(ContextEntry {
                collection: "*",
                path: "/",
                context: g,
            });
        }
        for (name, collection) in &self.inner.collections {
            if let Some(ctx) = collection.context.as_ref() {
                for (path, context) in ctx {
                    out.push(ContextEntry {
                        collection: name.as_str(),
                        path: path.as_str(),
                        context: context.as_str(),
                    });
                }
            }
        }
        out
    }

    /// Best-match context for a path within a collection. Picks the longest
    /// path-prefix match; falls back to `global_context` if nothing matches.
    pub fn find_context_for_path(&self, collection_name: &str, file_path: &str) -> Option<&str> {
        let collection = self.inner.collections.get(collection_name);
        let Some(contexts) = collection.and_then(|c| c.context.as_ref()) else {
            return self.inner.global_context.as_deref();
        };
        let normalized_path = normalize_leading_slash(file_path);
        let mut best: Option<(usize, &str)> = None;
        for (prefix, context) in contexts {
            let normalized_prefix = normalize_leading_slash(prefix);
            if normalized_path.starts_with(&normalized_prefix) {
                let len = normalized_prefix.len();
                if best.is_none_or(|(best_len, _)| len > best_len) {
                    best = Some((len, context.as_str()));
                }
            }
        }
        best.map(|(_, ctx)| ctx)
            .or(self.inner.global_context.as_deref())
    }

    // ── Internals ──

    fn effective_path(&self) -> Option<PathBuf> {
        match &self.source {
            Source::Inline => None,
            Source::File { path: Some(p) } => Some(p.clone()),
            Source::File { path: None } => Some(paths::config_file_path(&self.index_name)),
        }
    }
}

fn read_yaml_if_exists(path: &Path) -> Result<ConfigData> {
    if !path.exists() {
        return Ok(ConfigData::default());
    }
    let content = std::fs::read_to_string(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_norway::from_str(&content).map_err(|e| Error::Yaml {
        path: path.to_path_buf(),
        source: e,
    })
}

fn normalize_leading_slash(s: &str) -> String {
    if s.starts_with('/') {
        s.to_string()
    } else {
        format!("/{s}")
    }
}

// ============================================================================
// Free functions (TS exports that don't belong on Config)
// ============================================================================

/// Walk up from `start` looking for a `.rmd/index.yaml` (preferred) or
/// `.rmd/index.yml`. Returns the path to the first match, or `None`.
pub fn find_local_config_path(start: &Path) -> Option<PathBuf> {
    let start = std::path::absolute(start).unwrap_or_else(|_| start.to_path_buf());
    for dir in start.ancestors() {
        let rmd_dir = dir.join(".rmd");
        let yaml = rmd_dir.join("index.yaml");
        if yaml.is_file() {
            return Some(yaml);
        }
        let yml = rmd_dir.join("index.yml");
        if yml.is_file() {
            return Some(yml);
        }
    }
    None
}

/// Sibling SQLite path for a local YAML config (`<dir>/index.sqlite`).
pub fn local_db_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or(Path::new(""))
        .join("index.sqlite")
}

/// `true` if `name` only contains ASCII alphanumerics, `-`, or `_`.
pub fn is_valid_collection_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Sanitize an index-name input that may contain path separators.
///
/// NOTE: `rmd` sanitizes both `/` and `\\` (and `:`); the original qmd TS only
/// stripped `/`, which produces broken filenames on Windows.
fn sanitize_index_name(name: &str) -> String {
    if !name.contains('/') && !name.contains('\\') {
        return name.to_string();
    }
    let abs = std::env::current_dir().unwrap_or_default().join(name);
    let s = abs
        .to_string_lossy()
        .replace(['/', '\\'], "_")
        .replace(':', "_");
    s.trim_start_matches('_').to_string()
}

// ============================================================================
// Unit tests (pure logic)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_index_name_simple_passes_through() {
        assert_eq!(sanitize_index_name("index"), "index");
        assert_eq!(sanitize_index_name("myindex"), "myindex");
        assert_eq!(sanitize_index_name("a-b_c"), "a-b_c");
    }

    #[test]
    fn sanitize_index_name_replaces_slashes_and_backslashes_and_colon() {
        // Without exercising current_dir (which varies), feed a path that
        // already contains both kinds of separators and a drive colon.
        // Because the input contains `\`, the function takes the resolve
        // branch — current_dir().join("C:\\foo\\bar") on Windows returns
        // "C:\\foo\\bar"; on Unix it returns "<cwd>/C:\\foo\\bar".
        let out = sanitize_index_name("C:\\foo\\bar");
        // Whichever OS: must not contain any of the originals.
        assert!(!out.contains('/'), "got: {out}");
        assert!(!out.contains('\\'), "got: {out}");
        assert!(!out.contains(':'), "got: {out}");
        assert!(!out.starts_with('_'), "got: {out}");
    }

    #[test]
    fn is_valid_collection_name_accepts_alnum_dash_underscore() {
        assert!(is_valid_collection_name("docs"));
        assert!(is_valid_collection_name("my-notes"));
        assert!(is_valid_collection_name("my_notes"));
        assert!(is_valid_collection_name("notes2024"));
        assert!(is_valid_collection_name("ABC_def-123"));
    }

    #[test]
    fn is_valid_collection_name_rejects_spaces_dots_slashes() {
        assert!(!is_valid_collection_name(""));
        assert!(!is_valid_collection_name("my notes"));
        assert!(!is_valid_collection_name("my.notes"));
        assert!(!is_valid_collection_name("foo/bar"));
        assert!(!is_valid_collection_name("foo\\bar"));
        assert!(!is_valid_collection_name("日本語"));
    }

    #[test]
    fn find_context_for_path_normalizes_leading_slash() {
        // Construct an in-memory config with two prefixes on the same
        // collection — one rooted, one without leading slash. Both should
        // be normalized; the longer (more specific) wins.
        let mut data = ConfigData::default();
        let mut ctx = ContextMap::new();
        ctx.insert("/".to_string(), "root".to_string());
        ctx.insert("2024".to_string(), "year".to_string()); // no leading slash
        data.collections.insert(
            "notes".to_string(),
            Collection {
                path: "/n".to_string(),
                pattern: "**/*.md".to_string(),
                ignore: None,
                context: Some(ctx),
                update: None,
                include_by_default: None,
            },
        );
        let config = Config::inline(data);

        // Both "/2024/foo" and "2024/foo" should match the "2024" prefix
        // (more specific than "/").
        assert_eq!(
            config.find_context_for_path("notes", "/2024/foo.md"),
            Some("year")
        );
        assert_eq!(
            config.find_context_for_path("notes", "2024/foo.md"),
            Some("year")
        );
        // Path outside "2024" should still hit the "/" prefix.
        assert_eq!(
            config.find_context_for_path("notes", "/other.md"),
            Some("root")
        );
    }
}
