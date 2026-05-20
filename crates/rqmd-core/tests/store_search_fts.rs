//! End-to-end test for `search_fts` against a real in-memory store.

use rqmd_core::store::documents::{hash_content, insert_content, insert_document};
use rqmd_core::store::path::now_rfc3339;
use rqmd_core::store::search::{search_fts, SearchSource};
use rqmd_core::Store;
use tempfile::NamedTempFile;

fn insert(store: &Store, collection: &str, path: &str, title: &str, body: &str) {
    let now = now_rfc3339();
    let hash = hash_content(body);
    store
        .with_connection(|c| insert_content(c, &hash, body, &now))
        .unwrap();
    store
        .with_connection(|c| insert_document(c, collection, path, title, &hash, &now, &now))
        .unwrap();
}

fn deactivate(store: &Store, collection: &str, path: &str) {
    store
        .with_connection(|c| {
            c.execute(
                "UPDATE documents SET active = 0 WHERE collection = ? AND path = ?",
                rqmd_core::db::rusqlite::params![collection, path],
            )
            .map(|_| ())
        })
        .unwrap();
}

/// BM25 IDF needs corpus depth — a 2-doc corpus has near-zero IDF. These
/// non-matching docs make term-frequency differentiation meaningful.
fn add_noise(store: &Store, collection: &str, count: usize) {
    for i in 0..count {
        insert(
            store,
            collection,
            &format!("noise{i}.md"),
            &format!("Unrelated Topic {i}"),
            &format!("This document discusses gardening and cooking {i}"),
        );
    }
}

#[test]
fn search_fts_returns_matching_documents() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(&store, "docs", "a.md", "Alpha", "the quick brown fox");
    insert(&store, "docs", "b.md", "Beta", "the lazy dog barked");
    insert(&store, "docs", "c.md", "Gamma", "completely unrelated text");

    let hits = store
        .with_connection(|c| search_fts(c, "fox", Some(10), None))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.display_path, "docs/a.md");
    assert_eq!(hits[0].source, SearchSource::Fts);
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_fts_filters_by_collection() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(&store, "docs", "a.md", "Alpha", "the quick brown fox");
    insert(
        &store,
        "notes",
        "x.md",
        "Note",
        "the quick brown fox lives here too",
    );

    let hits = store
        .with_connection(|c| search_fts(c, "fox", Some(10), Some("docs")))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.collection_name, "docs");
}

#[test]
fn search_fts_handles_cjk_query() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(
        &store,
        "docs",
        "ja.md",
        "日本語",
        "これは日本語のテストです",
    );
    insert(
        &store,
        "docs",
        "en.md",
        "English",
        "this is an English test",
    );

    let hits = store
        .with_connection(|c| search_fts(c, "日本語", Some(10), None))
        .unwrap();
    assert!(!hits.is_empty(), "expected at least one CJK match");
    assert!(hits.iter().any(|h| h.doc.display_path == "docs/ja.md"));
}

/// Regression: CJK queries containing punctuation (、。「」 etc.) must not
/// produce a malformed FTS5 MATCH. After unifying on the TS-equivalent
/// *content filter* (`sanitize_fts5_term`), punctuation is stripped (matching
/// TS `sanitizeFTS5Phrase`) rather than quoted, so the query still parses and
/// the surrounding CJK characters still match.
#[test]
fn search_fts_handles_cjk_query_with_punctuation() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    // Body has the contiguous run 日本語テスト (the CJK character tokenizer
    // turns it into adjacent single-char tokens).
    insert(&store, "docs", "ja.md", "日本語", "これは日本語テストです");
    insert(
        &store,
        "docs",
        "en.md",
        "English",
        "this is an English test",
    );

    // The 、 must be sanitised away; the remaining 日本語テスト forms a phrase
    // that matches the contiguous run in the document.
    let hits = store
        .with_connection(|c| search_fts(c, "日本語、テスト", Some(10), None))
        .unwrap();
    assert!(
        hits.iter().any(|h| h.doc.display_path == "docs/ja.md"),
        "expected CJK match despite punctuation in query"
    );
}

/// Regression: a quoted phrase containing FTS5-special characters must not
/// produce a malformed MATCH. The removed quoting helper would map `c++` to
/// `"c++"` and then re-wrap the whole phrase, yielding `""c++" code"` (a
/// syntax error); the content filter strips `+` so the phrase parses cleanly.
#[test]
fn search_fts_handles_quoted_phrase_with_special_chars() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(&store, "docs", "a.md", "Doc", "learning c code is fun");

    // Quoted phrase with `+` — must sanitise to a valid query without error.
    let hits = store
        .with_connection(|c| search_fts(c, "\"c++ code\"", Some(10), None))
        .unwrap();
    assert!(hits.iter().any(|h| h.doc.display_path == "docs/a.md"));
}

#[test]
fn search_fts_supports_negation() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();

    insert(
        &store,
        "docs",
        "good.md",
        "G",
        "the quick brown fox is good",
    );
    insert(&store, "docs", "bad.md", "B", "the quick brown fox is bad");

    let hits = store
        .with_connection(|c| search_fts(c, "fox -bad", Some(10), None))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.display_path, "docs/good.md");
}

// --- ported from store.test.ts `describe("FTS Search")` ---

fn search(store: &Store, query: &str, limit: usize) -> Vec<rqmd_core::store::search::SearchResult> {
    store
        .with_connection(|c| search_fts(c, query, Some(limit), None))
        .unwrap()
}

#[test]
fn search_fts_returns_empty_for_no_match() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    insert(
        &store,
        "docs",
        "a.md",
        "Doc",
        "the quick brown fox jumps over the lazy dog",
    );
    assert!(search(&store, "nonexistent-term-xyz", 10).is_empty());
}

#[test]
fn search_fts_ranks_title_matches_higher() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    // "fox" in body only.
    insert(
        &store,
        "docs",
        "body.md",
        "Some Other Title",
        "The fox is here in the body",
    );
    // "fox" in title (4x BM25 weight) and body.
    insert(
        &store,
        "docs",
        "title.md",
        "Fox Title",
        "Different content without the animal fox",
    );

    let hits = search(&store, "fox", 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc.display_path, "docs/title.md");
}

#[test]
fn search_fts_title_boost_outweighs_body_frequency() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    add_noise(&store, "docs", 8);
    // "quantum" several times in a longer body but NOT the title.
    insert(
        &store,
        "docs",
        "body-only.md",
        "General Science Notes",
        "This research paper discusses quantum mechanics and the quantum model of computation. The quantum approach beats classical methods.",
    );
    // "quantum" in the title, short body without it.
    insert(
        &store,
        "docs",
        "title-match.md",
        "Quantum Computing Overview",
        "An introduction to the fundamentals of this emerging computing paradigm.",
    );

    let hits = search(&store, "quantum", 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc.display_path, "docs/title-match.md");
}

#[test]
fn search_fts_respects_limit() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    for i in 0..10 {
        insert(
            &store,
            "docs",
            &format!("doc{i}.md"),
            "Doc",
            "common keyword appears here",
        );
    }
    assert_eq!(search(&store, "common keyword", 3).len(), 3);
}

#[test]
fn search_fts_keeps_english_behavior_with_cjk_indexed() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    insert(
        &store,
        "docs",
        "english.md",
        "Vector Search Notes",
        "The quick brown fox explains vector search and BM25 ranking.",
    );
    insert(
        &store,
        "docs",
        "zh.md",
        "中文检索说明",
        "这里介绍向量数据库和关键词检索。",
    );

    let hits = search(&store, "quick fox", 10);
    let paths: Vec<_> = hits.iter().map(|h| h.doc.display_path.as_str()).collect();
    assert!(paths.contains(&"docs/english.md"));
    assert!(!paths.contains(&"docs/zh.md"));
}

#[test]
fn search_fts_handles_special_characters_without_error() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    insert(
        &store,
        "docs",
        "a.md",
        "Doc",
        "Function with params: foo(bar, baz)",
    );
    // Must not error on FTS5-special characters.
    let _ = store.with_connection(|c| search_fts(c, "foo(bar)", Some(10), None));
}

#[test]
fn search_fts_stronger_match_scores_higher_in_unit_range() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    add_noise(&store, "docs", 8);
    insert(
        &store,
        "docs",
        "strong.md",
        "Alpha Guide",
        "This is the definitive alpha reference with alpha details and more alpha info",
    );
    insert(
        &store,
        "docs",
        "weak.md",
        "General Notes",
        "Some notes that mention alpha in passing among other topics and keywords",
    );

    let hits = search(&store, "alpha", 10);
    assert_eq!(hits.len(), 2);
    let strong = hits
        .iter()
        .find(|h| h.doc.display_path.contains("strong"))
        .unwrap();
    let weak = hits
        .iter()
        .find(|h| h.doc.display_path.contains("weak"))
        .unwrap();
    assert!(strong.score > weak.score);
    for h in &hits {
        assert!(h.score > 0.0 && h.score < 1.0);
    }
}

#[test]
fn search_fts_min_score_filter_keeps_strong_drops_weak() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    add_noise(&store, "docs", 8);
    insert(
        &store,
        "docs",
        "strong.md",
        "Kubernetes Deployment",
        "Kubernetes deployment strategies for kubernetes clusters using kubernetes operators",
    );
    insert(
        &store,
        "docs",
        "weak.md",
        "Random Notes",
        "Various topics including a brief kubernetes mention among many other unrelated things",
    );

    let hits = search(&store, "kubernetes", 10);
    assert_eq!(hits.len(), 2);
    let strong = hits
        .iter()
        .find(|h| h.doc.display_path.contains("strong"))
        .unwrap()
        .score;
    let weak = hits
        .iter()
        .find(|h| h.doc.display_path.contains("weak"))
        .unwrap()
        .score;
    let threshold = (strong + weak) / 2.0;
    let kept: Vec<_> = hits.iter().filter(|h| h.score >= threshold).collect();
    assert_eq!(kept.len(), 1);
    assert!(kept[0].doc.display_path.contains("strong"));
}

#[test]
fn search_fts_ignores_inactive_documents() {
    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    insert(&store, "docs", "active.md", "Active", "findme content");
    insert(&store, "docs", "inactive.md", "Inactive", "findme content");
    deactivate(&store, "docs", "inactive.md");

    let hits = search(&store, "findme", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc.display_path, "docs/active.md");
    assert_eq!(hits[0].doc.filepath, "qmd://docs/active.md");
}

#[test]
fn search_fts_strong_signal_detection() {
    use rqmd_core::store::{STRONG_SIGNAL_MIN_GAP, STRONG_SIGNAL_MIN_SCORE};

    let tmp = NamedTempFile::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    // 50 noise docs give IDF ≈ log(50/2) — enough for scores above 0.85.
    add_noise(&store, "docs", 50);
    // Dominant: keyword in filepath + title + body.
    insert(
        &store,
        "docs",
        "zephyr/zephyr-guide.md",
        "Zephyr Configuration Guide",
        "Complete zephyr configuration guide. Zephyr setup instructions for zephyr deployment.",
    );
    // Weak: keyword once in a long body.
    insert(
        &store,
        "docs",
        "notes/misc.md",
        "General Notes",
        "Various topics covering many areas of technology and design. One of them might relate to \
         zephyr but mostly about other things entirely. Additional content about databases, \
         networking, security, performance, monitoring, deployment, testing, and documentation.",
    );

    let hits = search(&store, "zephyr", 10);
    assert_eq!(hits.len(), 2);
    let top = hits[0].score;
    let second = hits[1].score;
    assert!(
        top >= STRONG_SIGNAL_MIN_SCORE,
        "top {top} < {STRONG_SIGNAL_MIN_SCORE}"
    );
    let gap = top - second;
    assert!(
        gap >= STRONG_SIGNAL_MIN_GAP,
        "gap {gap} < {STRONG_SIGNAL_MIN_GAP}"
    );
}
