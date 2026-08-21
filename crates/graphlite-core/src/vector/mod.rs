//! Vector math, similarity metrics, SIMD acceleration, and quantization.

pub mod distance;

pub use distance::{
    cosine_similarity, dot_product, euclidean_distance, manhattan_distance, norm,
    norm_squared, normalize_in_place, normalized, Metric,
};
