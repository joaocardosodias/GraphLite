//! # GraphLite Core
//!
//! An embedded, single-file Graph and Vector database engine written in pure Rust.
//! Designed for local-first GraphRAG, AI memory, and low-latency knowledge graphs.

pub mod error;
pub mod graph;
pub mod id;
pub mod interner;
pub mod prompt;
pub mod record;
pub mod storage;
pub mod vector;

pub use error::{GraphLiteError, Result};
pub use graph::{
    bfs_adjacency, bfs_csr, compute_hybrid_scores, extract_subgraph_adjacency,
    extract_subgraph_csr, AdjacencyGraph, ConnectedSubgraph, CsrGraph,
    HybridScoreConfig, ScoredEntity, TraversalConfig, TraversalDirection, TraversedNode,
};
pub use id::{EdgeId, NodeId, StringId};
pub use interner::StringInterner;
pub use prompt::{
    count_tokens, format_connected_subgraph_markdown, format_pruned_subgraph_markdown,
    format_subgraph_triples, prune_subgraph_by_budget, to_json_payload, HeuristicTokenCounter,
    JsonEntity, JsonRelation, JsonSubgraphPayload, MarkdownFormatConfig, MarkdownStyle,
    PrunedSubgraph, TiktokenCounter, TokenCounter, TokenizerEncoding,
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
    serialize_node_block, serialize_quantized_vector_block, serialize_string_table,
    verify_file_integrity, GraphHeader, ZeroCopyCsrBlock, ZeroCopyNodeBlock,
    ZeroCopyStringTable, ZeroCopyVectorBlock, FLAG_COMPRESSED, FLAG_QUANTIZED_SQ8,
    GRAPH_MAGIC, GRAPH_VERSION, HEADER_SIZE,
};
pub use vector::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance, norm,
    norm_squared, normalize_in_place, normalized, simd_cosine_similarity,
    simd_dot_product, simd_euclidean_distance, simd_norm_squared, Metric,
    Quantization, QuantizedVector, TopKQueue, VectorStore,
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
