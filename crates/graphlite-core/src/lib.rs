//! # GraphLite Core
//!
//! An embedded, single-file Graph and Vector database engine written in pure Rust.
//! Designed for local-first GraphRAG, AI memory systems, and low-latency knowledge graphs.
//!
//! ## Architecture Overview
//!
//! GraphLite combines high-performance Compressed Sparse Row (CSR) graph topology
//! with SIMD-accelerated Int8 Scalarly Quantized (SQ8) vector search in a crash-resilient
//! single file format (`.graph`).
//!
//! ```text
//! [Query Vector] ──► [SIMD Vector Index] ──► [Top-K Seeds]
//!                                                  │
//!                                                  ▼
//! [LLM Markdown] ◄── [Token Budget Pruner] ◄── [CSR Multi-Hop BFS]
//! ```
//!
//! ## Quick Start Example
//!
//! ```rust
//! use graphlite_core::{GraphLiteConfig, GraphLiteEngine, QueryOptions};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Initialize an in-memory GraphLite instance with 4-dimensional vectors
//!     let config = GraphLiteConfig::new().with_dim(4).with_max_tokens(500);
//!     let db = GraphLiteEngine::in_memory(config)?;
//!
//!     // 2. Ingest entities with optional embeddings
//!     let v_titan = [1.0, 0.0, 0.0, 0.0];
//!     let v_ana = [0.95, 0.05, 0.0, 0.0];
//!
//!     let id_titan = db.upsert_node("Project Titan", "Project", "Generative AI Core", Some(&v_titan))?;
//!     let id_ana = db.upsert_node("Ana Silva", "Person", "Tech Lead", Some(&v_ana))?;
//!
//!     // 3. Connect entities with weighted relationships
//!     db.add_edge(id_ana, id_titan, "LEADS", 0.95, true)?;
//!
//!     // 4. Retrieve token-budgeted prompt context for LLMs
//!     let query_vector = [0.98, 0.02, 0.0, 0.0];
//!     let result = db.retrieve_context(&query_vector, None)?;
//!
//!     assert!(!result.markdown.is_empty());
//!     assert!(result.token_count <= 500);
//!     Ok(())
//! }
//! ```

#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

pub mod cache;
pub mod engine;
pub mod error;
pub mod graph;
pub mod id;
pub mod interner;
pub mod prompt;
pub mod record;
pub mod storage;
pub mod vector;

pub use cache::{CacheStats, EmbeddingCache, QueryCache, QueryCacheKey};
pub use engine::{
    GraphLiteConfig, GraphLiteEngine, QueryOptions, QueryResult, ResolutionConfig, ResolutionResult,
};
pub use error::{GraphLiteError, Result};
pub use graph::{
    bfs_adjacency, bfs_csr, compute_hybrid_scores, compute_hybrid_scores_with_rrf,
    compute_rrf_fused_ranks, extract_subgraph_adjacency, extract_subgraph_csr,
    reciprocal_rank_fusion, AdjacencyGraph, Bm25Index, Bm25Params, ConnectedSubgraph, CsrGraph,
    HybridScoreConfig, ScoredEntity, TraversalConfig, TraversalDirection, TraversedNode,
};
pub use id::{EdgeId, NodeId, StringId};
pub use interner::StringInterner;
pub use prompt::{
    count_tokens, format_connected_subgraph_markdown, format_pruned_subgraph_markdown,
    format_subgraph_triples, prune_subgraph_by_budget, prune_subgraph_by_budget_mmr,
    to_json_payload, HeuristicTokenCounter, JsonEntity, JsonRelation, JsonSubgraphPayload,
    MarkdownFormatConfig, MarkdownStyle, PrunedSubgraph, TiktokenCounter, TokenCounter,
    TokenizerEncoding,
};

#[cfg(feature = "serde")]
pub use prompt::format_subgraph_json;

pub use record::{
    EdgeRecord, NodeRecord, EDGE_RECORD_SIZE, FLAG_ACTIVE, FLAG_DIRECTED, NODE_RECORD_SIZE,
    NO_VECTOR_OFFSET,
};
pub use storage::{
    compute_file_checksum, crc32, crc32_update, deserialize_csr_block, deserialize_node_block,
    deserialize_quantized_vector_block, deserialize_string_table, serialize_csr_block,
    serialize_database, serialize_node_block, serialize_quantized_vector_block,
    serialize_string_table, verify_file_integrity, write_database_atomic, write_database_direct,
    GraphHeader, MmapGraphReader, ZeroCopyCsrBlock, ZeroCopyNodeBlock, ZeroCopyStringTable,
    ZeroCopyVectorBlock, FLAG_COMPRESSED, FLAG_QUANTIZED_SQ8, GRAPH_MAGIC, GRAPH_VERSION,
    HEADER_SIZE,
};
pub use vector::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance, norm, norm_squared,
    normalize_in_place, normalized, simd_cosine_similarity, simd_dot_product,
    simd_euclidean_distance, simd_norm_squared, LocalEmbedder, LocalReranker, Metric, Quantization,
    QuantizedVector, RerankResult, TopKQueue, VectorStore,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
