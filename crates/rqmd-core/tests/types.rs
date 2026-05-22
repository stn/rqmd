//! Serde roundtrip tests for the public data types.
//!
//! These guard against silent breaks of the `#[serde(rename = "type")]`
//! / `#[serde(rename_all = "lowercase")]` attributes that map between
//! TS-style JSON (`{"type": "lex"}`) and Rust enum/field names. PR2
//! callers cross the JSON boundary in two places (MCP server replies
//! and LLM cache values), so a wrong rename here would corrupt user
//! data without ever failing to compile.

use rqmd_core::llm::types::{
    EmbeddingResult, QueryType, Queryable, RerankDocumentResult, RerankResult,
};

#[test]
fn queryable_serializes_with_type_field_lowercase() {
    let q = Queryable {
        type_: QueryType::Lex,
        text: "rust async runtime".into(),
    };
    let json = serde_json::to_value(&q).unwrap();
    assert_eq!(json["type"], "lex");
    assert_eq!(json["text"], "rust async runtime");
    // No leaked `type_` field name.
    assert!(json.get("type_").is_none());
}

#[test]
fn queryable_deserializes_from_ts_shape() {
    for raw in &[
        r#"{"type": "lex", "text": "a"}"#,
        r#"{"type": "vec", "text": "b"}"#,
        r#"{"type": "hyde", "text": "c"}"#,
    ] {
        let q: Queryable = serde_json::from_str(raw).expect("must deserialize TS shape");
        assert_eq!(q.text.len(), 1);
        match q.type_ {
            QueryType::Lex | QueryType::Vec | QueryType::Hyde => {}
        }
    }
}

#[test]
fn queryable_roundtrip_preserves_value() {
    for type_ in [QueryType::Lex, QueryType::Vec, QueryType::Hyde] {
        let original = Queryable {
            type_,
            text: format!("text-{}", type_.as_str()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: Queryable = serde_json::from_str(&json).unwrap();
        assert_eq!(back.type_, original.type_);
        assert_eq!(back.text, original.text);
    }
}

#[test]
fn query_type_serializes_as_lowercase_string() {
    assert_eq!(serde_json::to_string(&QueryType::Lex).unwrap(), "\"lex\"");
    assert_eq!(serde_json::to_string(&QueryType::Vec).unwrap(), "\"vec\"");
    assert_eq!(serde_json::to_string(&QueryType::Hyde).unwrap(), "\"hyde\"");
}

#[test]
fn embedding_result_roundtrip() {
    let original = EmbeddingResult {
        embedding: vec![0.1, -0.2, 0.3],
        model: "hf:example/model.gguf".into(),
    };
    let json = serde_json::to_string(&original).unwrap();
    let back: EmbeddingResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model, original.model);
    assert_eq!(back.embedding, original.embedding);
}

#[test]
fn rerank_result_roundtrip_with_documents() {
    let original = RerankResult {
        results: vec![
            RerankDocumentResult {
                file: "a.md".into(),
                score: 0.9985,
                index: 0,
            },
            RerankDocumentResult {
                file: "b.md".into(),
                score: 0.0007,
                index: 1,
            },
        ],
        model: "hf:example/rerank.gguf".into(),
    };
    let json = serde_json::to_string(&original).unwrap();
    let back: RerankResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model, original.model);
    assert_eq!(back.results.len(), 2);
    assert_eq!(back.results[0].file, "a.md");
    assert!((back.results[0].score - 0.9985).abs() < 1e-6);
}
