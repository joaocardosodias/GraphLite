//! Vector math, similarity metrics, SIMD acceleration, and quantization.

pub mod distance;
pub mod simd;

pub use distance::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance, norm,
    norm_squared, normalize_in_place, normalized, Metric,
};

pub use simd::{
    simd_cosine_similarity, simd_dot_product, simd_euclidean_distance, simd_norm_squared,
};
