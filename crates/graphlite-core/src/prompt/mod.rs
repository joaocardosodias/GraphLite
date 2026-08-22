//! Token counting, prompt budget management, and markdown context formatting for LLM ingestion.

pub mod json;
pub mod markdown;
pub mod pruner;
pub mod token_counter;

pub use json::{
    format_subgraph_triples, to_json_payload, JsonEntity, JsonRelation, JsonSubgraphPayload,
};

#[cfg(feature = "serde")]
pub use json::format_subgraph_json;

pub use markdown::{
    format_connected_subgraph_markdown, format_pruned_subgraph_markdown, MarkdownFormatConfig,
    MarkdownStyle,
};
pub use pruner::{prune_subgraph_by_budget, PrunedSubgraph};
pub use token_counter::{
    count_tokens, HeuristicTokenCounter, TiktokenCounter, TokenCounter, TokenizerEncoding,
};
