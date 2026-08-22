//! # GraphLite Core
//!
//! An embedded, single-file Graph and Vector database engine written in pure Rust.
//! Designed for local-first GraphRAG, AI memory, and low-latency knowledge graphs.

pub mod error;
pub mod id;
pub mod interner;
pub mod record;
pub mod vector;

pub use error::{GraphLiteError, Result};
pub use id::{EdgeId, NodeId, StringId};
pub use interner::StringInterner;
pub use record::{EdgeRecord, NodeRecord, FLAG_ACTIVE, FLAG_DIRECTED, NO_VECTOR_OFFSET};
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
