//! Token counting, prompt budget management, and markdown context formatting for LLM ingestion.

pub mod token_counter;

pub use token_counter::{
    count_tokens, HeuristicTokenCounter, TiktokenCounter, TokenCounter, TokenizerEncoding,
};
