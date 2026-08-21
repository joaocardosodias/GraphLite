use crate::error::{GraphLiteError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Distance metric used for vector similarity calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum Metric {
    /// Cosine similarity (measures angle between vectors, range [-1.0, 1.0]).
    #[default]
    Cosine = 0,
    /// Dot product / Inner product (useful for normalized embeddings).
    DotProduct = 1,
    /// Euclidean distance (L2 distance in vector space).
    Euclidean = 2,
}

impl Metric {
    /// Computes the similarity/distance score between two vectors according to this metric.
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        match self {
            Metric::Cosine => cosine_similarity(a, b),
            Metric::DotProduct => dot_product(a, b),
            Metric::Euclidean => euclidean_distance(a, b),
        }
    }
}

/// Computes the Dot Product (Inner Product) of two floating-point vectors.
///
/// $$\text{dot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot b_i$$
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphLiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    Ok(dot)
}

/// Computes the L2 Norm (Euclidean magnitude) of a vector.
///
/// $$\|a\| = \sqrt{\sum_{i=0}^{n-1} a_i^2}$$
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    let sum_sq: f32 = a.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

/// Computes the Cosine Similarity between two vectors.
///
/// Returns a value in the range `[-1.0, 1.0]`, where `1.0` means identical direction.
/// If either vector has near-zero magnitude, returns `0.0` to avoid division by zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphLiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = (norm_a_sq * norm_b_sq).sqrt();
    if denominator <= f32::EPSILON {
        return Ok(0.0);
    }

    // Clamp to valid range [-1.0, 1.0] to account for floating point inaccuracies
    let sim = (dot / denominator).clamp(-1.0, 1.0);
    Ok(sim)
}

/// Computes the Euclidean Distance (L2 distance) between two vectors.
///
/// $$\text{dist}(a, b) = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}$$
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphLiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    Ok(sum_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot_product(&v1, &v2).unwrap(), 32.0);

        // Orthogonal vectors
        let v3 = [1.0, 0.0];
        let v4 = [0.0, 1.0];
        assert_eq!(dot_product(&v3, &v4).unwrap(), 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors -> 1.0
        let v1 = [1.0, 2.0, 3.0];
        let v2 = [1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v1, &v2).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);

        // Opposite vectors -> -1.0
        let v_pos = [1.0, 0.0, 0.0];
        let v_neg = [-1.0, 0.0, 0.0];
        let sim_neg = cosine_similarity(&v_pos, &v_neg).unwrap();
        assert!((sim_neg - (-1.0)).abs() < 1e-6);

        // Orthogonal vectors -> 0.0
        let v_x = [1.0, 0.0];
        let v_y = [0.0, 1.0];
        assert_eq!(cosine_similarity(&v_x, &v_y).unwrap(), 0.0);

        // Zero vector handling (safe no NaN)
        let zero = [0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&v1, &zero).unwrap(), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        // Classic 3-4-5 right triangle
        let p1 = [0.0, 0.0];
        let p2 = [3.0, 4.0];
        let dist = euclidean_distance(&p1, &p2).unwrap();
        assert!((dist - 5.0).abs() < 1e-6);

        // Identical points -> 0.0
        assert_eq!(euclidean_distance(&p1, &p1).unwrap(), 0.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let v1 = [1.0, 2.0];
        let v2 = [1.0, 2.0, 3.0];

        assert!(matches!(
            dot_product(&v1, &v2),
            Err(GraphLiteError::VectorDimensionMismatch { expected: 2, found: 3 })
        ));

        assert!(matches!(
            cosine_similarity(&v1, &v2),
            Err(GraphLiteError::VectorDimensionMismatch { expected: 2, found: 3 })
        ));

        assert!(matches!(
            euclidean_distance(&v1, &v2),
            Err(GraphLiteError::VectorDimensionMismatch { expected: 2, found: 3 })
        ));
    }

    #[test]
    fn test_metric_enum_dispatch() {
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0];

        assert_eq!(Metric::Cosine.calculate(&v1, &v2).unwrap(), 0.0);
        assert_eq!(Metric::DotProduct.calculate(&v1, &v2).unwrap(), 0.0);
        let euc = Metric::Euclidean.calculate(&v1, &v2).unwrap();
        assert!((euc - 2.0f32.sqrt()).abs() < 1e-6);
    }
}
