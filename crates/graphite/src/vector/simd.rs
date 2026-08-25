#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use crate::error::{GraphiteError, Result};

/// Hardware-accelerated dot product using unrolled 8-lane chunks.
///
/// This unrolls the loop into four independent accumulator lanes (`sum0..sum3`),
/// breaking CPU instruction dependency chains and enabling the LLVM auto-vectorizer
/// to generate native AVX2 (`vfmadd231ps`) and ARM NEON (`fmla`) instructions.
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        sum0 += ca[0] * cb[0] + ca[1] * cb[1];
        sum1 += ca[2] * cb[2] + ca[3] * cb[3];
        sum2 += ca[4] * cb[4] + ca[5] * cb[5];
        sum3 += ca[6] * cb[6] + ca[7] * cb[7];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);

    for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
        total += x * y;
    }

    Ok(total)
}

/// Hardware-accelerated squared L2 norm calculation using unrolled 8-lane chunks.
pub fn simd_norm_squared(a: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let remainder_a = chunks_a.remainder();

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    for ca in chunks_a {
        sum0 += ca[0] * ca[0] + ca[1] * ca[1];
        sum1 += ca[2] * ca[2] + ca[3] * ca[3];
        sum2 += ca[4] * ca[4] + ca[5] * ca[5];
        sum3 += ca[6] * ca[6] + ca[7] * ca[7];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);

    for x in remainder_a {
        total += x * x;
    }

    total
}

/// Hardware-accelerated Cosine Similarity using multi-lane SIMD vectorization.
pub fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut norm_a0 = 0.0f32;
    let mut norm_a1 = 0.0f32;
    let mut norm_b0 = 0.0f32;
    let mut norm_b1 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        dot0 += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
        dot1 += ca[4] * cb[4] + ca[5] * cb[5] + ca[6] * cb[6] + ca[7] * cb[7];

        norm_a0 += ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2] + ca[3] * ca[3];
        norm_a1 += ca[4] * ca[4] + ca[5] * ca[5] + ca[6] * ca[6] + ca[7] * ca[7];

        norm_b0 += cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2] + cb[3] * cb[3];
        norm_b1 += cb[4] * cb[4] + cb[5] * cb[5] + cb[6] * cb[6] + cb[7] * cb[7];
    }

    let mut dot = dot0 + dot1;
    let mut norm_a_sq = norm_a0 + norm_a1;
    let mut norm_b_sq = norm_b0 + norm_b1;

    for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = (norm_a_sq * norm_b_sq).sqrt();
    if denominator < 1e-12 {
        Ok(0.0)
    } else {
        Ok((dot / denominator).clamp(-1.0, 1.0))
    }
}

/// Hardware-accelerated Euclidean Distance (L2) using unrolled 8-lane SIMD chunks.
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphiteError::VectorDimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    for (ca, cb) in chunks_a.zip(chunks_b) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        let d4 = ca[4] - cb[4];
        let d5 = ca[5] - cb[5];
        let d6 = ca[6] - cb[6];
        let d7 = ca[7] - cb[7];

        sum0 += d0 * d0 + d1 * d1;
        sum1 += d2 * d2 + d3 * d3;
        sum2 += d4 * d4 + d5 * d5;
        sum3 += d6 * d6 + d7 * d7;
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3);

    for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
        let diff = x - y;
        total += diff * diff;
    }

    Ok(total.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::distance::{cosine_similarity, dot_product, euclidean_distance};

    #[test]
    fn test_simd_dot_product_parity() {
        for dim in [7, 8, 13, 16, 384, 768, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32) * -0.02 + 0.5).collect();

            let expected = dot_product(&a, &b).unwrap();
            let actual = simd_dot_product(&a, &b).unwrap();

            let diff = (expected - actual).abs();
            let rel_diff = diff / expected.abs().max(1.0);
            assert!(
                rel_diff < 1e-5,
                "Failed dot product parity for dimension {}: expected {}, got {}",
                dim,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_simd_cosine_similarity_parity() {
        for dim in [8, 384, 768, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 7 % 100) as f32) - 50.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 13 % 100) as f32) - 50.0).collect();

            let expected = cosine_similarity(&a, &b).unwrap();
            let actual = simd_cosine_similarity(&a, &b).unwrap();

            assert!(
                (expected - actual).abs() < 1e-5,
                "Failed cosine similarity parity for dimension {}",
                dim
            );
        }
    }

    #[test]
    fn test_simd_euclidean_distance_parity() {
        for dim in [8, 384, 768] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.2 + 1.0).collect();

            let expected = euclidean_distance(&a, &b).unwrap();
            let actual = simd_euclidean_distance(&a, &b).unwrap();

            assert!(
                (expected - actual).abs() < 1e-4,
                "Failed euclidean distance parity for dimension {}",
                dim
            );
        }
    }

    #[test]
    fn test_simd_dimension_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        assert!(simd_dot_product(&a, &b).is_err());
        assert!(simd_cosine_similarity(&a, &b).is_err());
        assert!(simd_euclidean_distance(&a, &b).is_err());
    }
}
