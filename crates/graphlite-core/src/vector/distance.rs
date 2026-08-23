use crate::error::{GraphLiteError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported distance metrics for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum Metric {
    /// Cosine Similarity (measures the angle between vectors, range: [-1.0, 1.0]).
    #[default]
    Cosine = 0,
    /// Dot Product (inner product, unnormalized).
    DotProduct = 1,
    /// Euclidean Distance (L2 distance, straight-line distance in Euclidean space).
    Euclidean = 2,
    /// Manhattan Distance (L1 distance, sum of absolute coordinate differences).
    Manhattan = 3,
}

impl Metric {
    /// Computes the similarity or distance score between two vectors using this metric.
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        match self {
            Metric::Cosine => cosine_similarity(a, b),
            Metric::DotProduct => dot_product(a, b),
            Metric::Euclidean => euclidean_distance(a, b),
            Metric::Manhattan => manhattan_distance(a, b),
        }
    }
}

/// Computes the dot product (inner product) between two vectors.
///
/// $$\text{dot}(a, b) = \sum_{i=0}^{n-1} a_i \cdot b_i$$
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    validate_dimensions(a, b)?;

    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok(sum)
}

/// Computes the squared L2 norm (magnitude squared) of a vector.
///
/// $$\|a\|^2 = \sum_{i=0}^{n-1} a_i^2$$
#[inline]
pub fn norm_squared(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Computes the L2 norm (Euclidean length) of a vector.
///
/// $$\|a\| = \sqrt{\sum_{i=0}^{n-1} a_i^2}$$
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    norm_squared(a).sqrt()
}

/// Computes the Cosine Similarity between two vectors.
///
/// $$\text{cos}(\theta) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$
///
/// Returns a value between -1.0 (opposite) and 1.0 (identical direction).
/// If either vector has zero magnitude, returns 0.0 safely.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    validate_dimensions(a, b)?;

    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = (norm_a_sq * norm_b_sq).sqrt();
    if denominator < 1e-12 {
        Ok(0.0)
    } else {
        // Clamp to [-1.0, 1.0] to guard against floating-point rounding errors
        Ok((dot / denominator).clamp(-1.0, 1.0))
    }
}

/// Computes the Euclidean Distance (L2 distance) between two vectors.
///
/// $$d(a, b) = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}$$
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    validate_dimensions(a, b)?;

    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    Ok(sum_sq.sqrt())
}

/// Computes the Manhattan Distance (L1 distance) between two vectors.
///
/// $$d(a, b) = \sum_{i=0}^{n-1} |a_i - b_i|$$
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    validate_dimensions(a, b)?;

    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum)
}

/// Normalizes a vector in-place so that its L2 norm becomes 1.0.
pub fn normalize_in_place(a: &mut [f32]) {
    let mag = norm(a);
    if mag > 1e-12 {
        let inv = 1.0 / mag;
        for x in a.iter_mut() {
            *x *= inv;
        }
    }
}

/// Returns a new normalized vector with L2 norm equal to 1.0.
pub fn normalized(a: &[f32]) -> Vec<f32> {
    let mut cloned = a.to_vec();
    normalize_in_place(&mut cloned);
    cloned
}

/// Helper function to ensure vector slices have identical dimensions.
#[inline]
fn validate_dimensions(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        Err(GraphLiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let dot = dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [-1.0, 0.0, 0.0];

        // Identical vectors -> 1.0
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
        // Orthogonal vectors (90 degrees) -> 0.0
        assert!(cosine_similarity(&a, &c).unwrap().abs() < 1e-6);
        // Opposite vectors (180 degrees) -> -1.0
        assert!((cosine_similarity(&a, &d).unwrap() - (-1.0)).abs() < 1e-6);

        // Zero vector safety
        let zero = [0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &zero).unwrap(), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        // sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert_eq!(euclidean_distance(&a, &b).unwrap(), 5.0);

        // Distance to self is 0.0
        assert_eq!(euclidean_distance(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        // |4-1| + |6-2| = 3 + 4 = 7.0
        assert_eq!(manhattan_distance(&a, &b).unwrap(), 7.0);
    }

    #[test]
    fn test_normalization() {
        let a = [3.0, 4.0];
        let norm_a = normalized(&a);
        assert!((norm(&norm_a) - 1.0).abs() < 1e-6);
        assert_eq!(norm_a, vec![0.6, 0.8]);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];

        let err = dot_product(&a, &b).unwrap_err();
        match err {
            GraphLiteError::VectorDimensionMismatch { expected, found } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 2);
            }
            _ => panic!("Expected VectorDimensionMismatch"),
        }
    }

    #[test]
    fn test_metric_enum_compute() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];

        assert_eq!(
            Metric::DotProduct.compute(&a, &b).unwrap(),
            dot_product(&a, &b).unwrap()
        );
        assert_eq!(
            Metric::Euclidean.compute(&a, &b).unwrap(),
            euclidean_distance(&a, &b).unwrap()
        );
    }
}
