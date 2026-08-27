//! Vector math, similarity metrics, SIMD acceleration, and quantization.

pub mod device;
pub mod distance;
pub mod embedding;
pub mod quantization;
pub mod reranker;
pub mod simd;
pub mod store;
pub mod topk;

pub use device::{CudaStatus, DeviceType};
pub use distance::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance, norm, norm_squared,
    normalize_in_place, normalized, Metric,
};

pub use embedding::{EmbeddingModelType, LocalEmbedder};
pub use reranker::{LocalReranker, RerankResult, RerankerModelType};

pub use quantization::{Quantization, QuantizedVector};

pub use simd::{
    simd_cosine_similarity, simd_dot_product, simd_euclidean_distance, simd_norm_squared,
};

pub use store::VectorStore;
pub use topk::TopKQueue;
