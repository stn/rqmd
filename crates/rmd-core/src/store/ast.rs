//! AST-aware chunking support via native tree-sitter grammars.
//!
//! Port of `tobi/qmd`'s `src/ast.ts`. The TS version drives `web-tree-sitter`
//! (WASM) and is async; this port uses the native [`tree_sitter`] crate plus
//! per-language grammar crates and is therefore fully synchronous — no
//! `Parser::init`, no WASM load step.
//!
//! Provides:
//! - Language detection from filename extension.
//! - AST break point extraction at function / class / import boundaries,
//!   producing [`BreakPoint`]s that merge cleanly with the regex-based break
//!   points from [`super::chunking`].
//! - A status probe used by `rmd-cli status` to report grammar availability.
//!
//! All public functions degrade gracefully: unsupported language, query
//! compile failure, and parse failure each return an empty `Vec`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor, QueryError};

use super::chunking::{BreakKind, BreakPoint};

// ============================================================================
// Language detection
// ============================================================================

/// Source languages with tree-sitter grammars available in this port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Typescript,
    Tsx,
    Javascript,
    Python,
    Go,
    Rust,
}

impl SupportedLanguage {
    /// Display name used for status output (lower-case identifier).
    pub fn as_str(self) -> &'static str {
        match self {
            SupportedLanguage::Typescript => "typescript",
            SupportedLanguage::Tsx => "tsx",
            SupportedLanguage::Javascript => "javascript",
            SupportedLanguage::Python => "python",
            SupportedLanguage::Go => "go",
            SupportedLanguage::Rust => "rust",
        }
    }

    /// All variants — convenience for iteration in [`get_ast_status`].
    pub const ALL: [SupportedLanguage; 6] = [
        SupportedLanguage::Typescript,
        SupportedLanguage::Tsx,
        SupportedLanguage::Javascript,
        SupportedLanguage::Python,
        SupportedLanguage::Go,
        SupportedLanguage::Rust,
    ];
}

/// Detect language from a path's extension.
///
/// Mirrors the TS `EXTENSION_MAP` (`ast.ts:36–48`). Returns `None` for
/// markdown (`.md`) and any other unsupported extension. Case-insensitive on
/// the extension only.
pub fn detect_language(filepath: &str) -> Option<SupportedLanguage> {
    let ext = Path::new(filepath)
        .extension()?
        .to_str()?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "ts" | "mts" | "cts" => SupportedLanguage::Typescript,
        "tsx" | "jsx" => SupportedLanguage::Tsx,
        "js" | "mjs" | "cjs" => SupportedLanguage::Javascript,
        "py" => SupportedLanguage::Python,
        "go" => SupportedLanguage::Go,
        "rs" => SupportedLanguage::Rust,
        _ => return None,
    })
}

// ============================================================================
// Per-language queries
// ============================================================================

// Tree-sitter S-expression queries. Each capture name maps to a [`BreakKind`]
// via [`ast_kind_for`]. Identical to the TS version (`ast.ts:94–151`) except
// `interface_declaration` is omitted from `javascript` (the JS grammar has
// no such node).

const TYPESCRIPT_QUERY: &str = r#"
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
"#;

const TSX_QUERY: &str = TYPESCRIPT_QUERY;

const JAVASCRIPT_QUERY: &str = r#"
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
"#;

const PYTHON_QUERY: &str = r#"
    (class_definition) @class
    (function_definition) @func
    (decorated_definition) @decorated
    (import_statement) @import
    (import_from_statement) @import
"#;

const GO_QUERY: &str = r#"
    (type_declaration) @type
    (function_declaration) @func
    (method_declaration) @method
    (import_declaration) @import
"#;

const RUST_QUERY: &str = r#"
    (struct_item) @struct
    (impl_item) @impl
    (function_item) @func
    (trait_item) @trait
    (enum_item) @enum
    (use_declaration) @import
    (type_item) @type
    (mod_item) @mod
"#;

fn query_source(lang: SupportedLanguage) -> &'static str {
    match lang {
        SupportedLanguage::Typescript => TYPESCRIPT_QUERY,
        SupportedLanguage::Tsx => TSX_QUERY,
        SupportedLanguage::Javascript => JAVASCRIPT_QUERY,
        SupportedLanguage::Python => PYTHON_QUERY,
        SupportedLanguage::Go => GO_QUERY,
        SupportedLanguage::Rust => RUST_QUERY,
    }
}

/// Map a tree-sitter capture name to a [`BreakKind`].
///
/// TS uses `SCORE_MAP[name] ?? 20` (`ast.ts:158–172, 306`); we preserve the
/// fallback via [`BreakKind::AstUnknown`] so future query edits introducing
/// unknown capture names aren't silently promoted to function priority.
fn ast_kind_for(name: &str) -> BreakKind {
    match name {
        "class" | "iface" | "struct" | "trait" | "impl" | "mod" => BreakKind::AstClass,
        "export" | "func" | "method" | "decorated" => BreakKind::AstFunc,
        "type" | "enum" => BreakKind::AstType,
        "import" => BreakKind::AstImport,
        _ => BreakKind::AstUnknown,
    }
}

// ============================================================================
// Grammar / query caches
// ============================================================================

// Native grammars never fail to load; we still cache the `Language` to avoid
// repeated `LanguageFn` → `Language` conversions.

fn language_for(lang: SupportedLanguage) -> &'static Language {
    match lang {
        SupportedLanguage::Typescript => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        }
        SupportedLanguage::Tsx => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_typescript::LANGUAGE_TSX.into())
        }
        SupportedLanguage::Javascript => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_javascript::LANGUAGE.into())
        }
        SupportedLanguage::Python => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_python::LANGUAGE.into())
        }
        SupportedLanguage::Go => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_go::LANGUAGE.into())
        }
        SupportedLanguage::Rust => {
            static L: OnceLock<Language> = OnceLock::new();
            L.get_or_init(|| tree_sitter_rust::LANGUAGE.into())
        }
    }
}

/// Compile and cache the query for a language. `QueryError` is cached too —
/// once a query fails, we don't retry, matching the TS `failedLanguages` set
/// (`ast.ts:184`).
fn query_for(lang: SupportedLanguage) -> Result<&'static Query, &'static QueryError> {
    fn slot(lang: SupportedLanguage) -> &'static OnceLock<Result<Query, QueryError>> {
        match lang {
            SupportedLanguage::Typescript => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
            SupportedLanguage::Tsx => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
            SupportedLanguage::Javascript => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
            SupportedLanguage::Python => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
            SupportedLanguage::Go => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
            SupportedLanguage::Rust => {
                static Q: OnceLock<Result<Query, QueryError>> = OnceLock::new();
                &Q
            }
        }
    }

    let cell = slot(lang);
    let entry = cell.get_or_init(|| {
        let language = language_for(lang);
        let result = Query::new(language, query_source(lang));
        if let Err(err) = &result {
            eprintln!(
                "[rmd] tree-sitter query failed for {}: {err}; falling back to regex chunking",
                lang.as_str()
            );
        }
        result
    });
    entry.as_ref()
}

// ============================================================================
// AST break point extraction
// ============================================================================

/// Parse `content` and return break points at AST node boundaries.
///
/// Returns an empty `Vec` for unsupported languages, parse failures, or
/// query compile failures. Never panics. Mirrors `getASTBreakPoints`
/// (`ast.ts:274–323`).
pub fn get_ast_break_points(content: &str, filepath: &str) -> Vec<BreakPoint> {
    let Some(lang) = detect_language(filepath) else {
        return Vec::new();
    };
    let Ok(query) = query_for(lang) else {
        return Vec::new();
    };
    let language = language_for(lang);

    let mut parser = Parser::new();
    if parser.set_language(language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), content.as_bytes());
    let capture_names = query.capture_names();

    // pos → BreakPoint (highest score wins) — TS uses Map<number, BreakPoint>.
    let mut seen: BTreeMap<usize, BreakPoint> = BTreeMap::new();
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let pos = cap.node.start_byte();
            // graceful: don't panic if capture_names is shorter than expected.
            let name = capture_names.get(cap.index as usize).copied().unwrap_or("");
            let kind = ast_kind_for(name);
            let score = kind.score();
            seen.entry(pos)
                .and_modify(|bp| {
                    if score > bp.score {
                        bp.score = score;
                        bp.kind = kind;
                    }
                })
                .or_insert(BreakPoint { pos, score, kind });
        }
    }

    // BTreeMap iteration is sorted by key (pos), so the result is already
    // sorted — matching the invariant `find_best_cutoff` relies on.
    seen.into_values().collect()
}

// ============================================================================
// Status probe
// ============================================================================

/// Per-language availability entry returned by [`get_ast_status`].
#[derive(Debug, Clone)]
pub struct LangStatus {
    pub language: SupportedLanguage,
    pub available: bool,
    pub error: Option<String>,
}

/// Aggregated grammar availability. Mirrors `getASTStatus` (`ast.ts:333–375`).
#[derive(Debug, Clone)]
pub struct AstStatus {
    pub available: bool,
    pub languages: Vec<LangStatus>,
}

/// Probe each grammar by compiling its query. With native bindings the only
/// failure mode is `Query::new` (e.g. a grammar-version mismatch renaming a
/// node). Cheap to call repeatedly thanks to the `query_for` cache.
pub fn get_ast_status() -> AstStatus {
    let mut languages = Vec::with_capacity(SupportedLanguage::ALL.len());
    for lang in SupportedLanguage::ALL {
        match query_for(lang) {
            Ok(_) => languages.push(LangStatus {
                language: lang,
                available: true,
                error: None,
            }),
            Err(err) => languages.push(LangStatus {
                language: lang,
                available: false,
                error: Some(err.to_string()),
            }),
        }
    }
    AstStatus {
        available: languages.iter().any(|l| l.available),
        languages,
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::chunking::{merge_break_points, scan_break_points};

    // ---- detect_language ----

    #[test]
    fn detect_language_typescript() {
        assert_eq!(
            detect_language("foo.ts"),
            Some(SupportedLanguage::Typescript)
        );
        assert_eq!(
            detect_language("foo.mts"),
            Some(SupportedLanguage::Typescript)
        );
        assert_eq!(
            detect_language("foo.cts"),
            Some(SupportedLanguage::Typescript)
        );
    }

    #[test]
    fn detect_language_tsx_jsx() {
        assert_eq!(detect_language("foo.tsx"), Some(SupportedLanguage::Tsx));
        assert_eq!(detect_language("foo.jsx"), Some(SupportedLanguage::Tsx));
    }

    #[test]
    fn detect_language_uppercase_extension() {
        // Path::extension preserves case; our match lower-cases first.
        assert_eq!(detect_language("FOO.RS"), Some(SupportedLanguage::Rust));
    }

    #[test]
    fn detect_language_returns_none_for_markdown_and_unknown() {
        assert_eq!(detect_language("doc.md"), None);
        assert_eq!(detect_language("notes.txt"), None);
        assert_eq!(detect_language("no-extension"), None);
    }

    // ---- get_ast_break_points ----

    #[test]
    fn break_points_for_typescript_class_and_function() {
        let src = "export class Foo {\n  bar() { return 1 }\n}\nfunction baz() { return 2 }\n";
        let bps = get_ast_break_points(src, "x.ts");
        assert!(!bps.is_empty(), "expected at least one break point");
        // Scores follow the documented hierarchy: class >= func.
        let max_class = bps
            .iter()
            .filter(|b| b.kind == BreakKind::AstClass)
            .map(|b| b.score)
            .max();
        let max_func = bps
            .iter()
            .filter(|b| b.kind == BreakKind::AstFunc)
            .map(|b| b.score)
            .max();
        assert_eq!(max_class, Some(100));
        assert_eq!(max_func, Some(90));
    }

    #[test]
    fn break_points_for_python_function() {
        let src = "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n";
        let bps = get_ast_break_points(src, "x.py");
        assert!(bps.iter().any(|b| b.kind == BreakKind::AstFunc));
        assert!(bps.iter().any(|b| b.kind == BreakKind::AstClass));
    }

    #[test]
    fn break_points_for_rust_struct_and_use() {
        let src = "use std::io;\nstruct S;\nfn f() {}\n";
        let bps = get_ast_break_points(src, "x.rs");
        assert!(bps.iter().any(|b| b.kind == BreakKind::AstClass)); // struct
        assert!(bps.iter().any(|b| b.kind == BreakKind::AstFunc));
        assert!(bps.iter().any(|b| b.kind == BreakKind::AstImport));
    }

    #[test]
    fn break_points_for_unparseable_source_are_empty_or_partial() {
        // Tree-sitter is error-tolerant; it returns a tree with ERROR nodes
        // rather than failing. The contract is "never panic, never throw",
        // which we exercise here.
        let _ = get_ast_break_points("!!!@@@###", "x.ts");
        let _ = get_ast_break_points("", "x.ts");
    }

    #[test]
    fn break_points_skipped_for_markdown_and_text() {
        // Critical: chunk_document must not pay an AST cost for .md / .txt.
        assert!(get_ast_break_points("# hi\n", "doc.md").is_empty());
        assert!(get_ast_break_points("hello", "notes.txt").is_empty());
    }

    // ---- merge invariant ----

    #[test]
    fn merge_with_regex_yields_sorted_break_points() {
        let src = "use std::io;\nstruct S;\nfn f() {}\n";
        let regex_bps = scan_break_points(src);
        let ast_bps = get_ast_break_points(src, "x.rs");
        let merged = merge_break_points(&regex_bps, &ast_bps);
        // find_best_cutoff relies on ascending pos order.
        for w in merged.windows(2) {
            assert!(w[0].pos <= w[1].pos, "merged break points not sorted");
        }
    }

    // ---- status ----

    #[test]
    fn status_reports_all_six_grammars_available() {
        let s = get_ast_status();
        assert!(s.available);
        assert_eq!(s.languages.len(), 6);
        for lang in s.languages {
            assert!(
                lang.available,
                "{} grammar should compile (error: {:?})",
                lang.language.as_str(),
                lang.error
            );
        }
    }
}
