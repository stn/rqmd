//! Shared test helpers — keep MockLlm here so every `tests/*.rs` file can
//! `mod common; use common::*;` without rebuilding the fake LLM in each.

pub mod mock_llm;
