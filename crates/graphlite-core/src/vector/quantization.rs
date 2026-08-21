use crate::error::{GraphLiteError, Result};
use crate::vector::distance::norm;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Quantization mode for storing and querying vectors in GraphLite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum Quantization {
    /// Full precision 32-bit floating point (4 bytes per dimension).
    #[default]
    None = 0,
    /// 8-bit Scalar Quantization (1 byte per dimension, 4x memory compression).
    ScalarInt8 = 1,
}

/// An 8-bit Scalarly Quantized Vector (SQ8).
///
/// Compresses 32-bit floating-point numbers (`f32`) into 8-bit signed integers (`i8`),
/// reducing memory and disk usage by 75% while retaining >99% retrieval accuracy.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizedVector {
    /// Quantized 8-bit values in the range [-127, 127].
    pub data: Vec<i8>,
    /// Scaling factor used to map integer values back to the original float range.
    pub scale: f32,
    /// Original L2 norm (magnitude) of the unquantized float vector.
    pub norm: f32,
}

impl QuantizedVector {
    /// Quantizes a 32-bit float vector slice into an 8-bit `QuantizedVector`.
    pub fn quantize(v: &[f32]) -> Self {
        let original_norm = norm(v);
        if v.is_empty() {
            return Self {
                data: Vec::new(),
                scale: 0.0,
                norm: 0.0,
            };
        }

        // Find the maximum absolute value for symmetric quantization scaling
        let mut max_abs = 0.0f32;
        for &val in v {
            let abs = val.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        if max_abs < 1e-12 {
            return Self {
                data: vec![0; v.len()],
                scale: 0.0,
                norm: 0.0,
            };
        }

        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;

        let mut data = Vec::with_capacity(v.len());
        for &val in v {
            let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
            data.push(q);
        }

        Self {
            data,
            scale,
            norm: original_norm,
        }
    }

    /// Dequantizes the 8-bit integer vector back into an approximate `Vec<f32>`.
    pub fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&q| (q as f32) * self.scale).collect()
    }

    /// Asymmetric Dot Product: computes the dot product between this quantized vector and a raw `f32` query vector.
    ///
    /// This is the most common retrieval scenario in production: query stays in `f32` for maximal precision,
    /// while millions of stored vectors remain compressed in `i8`.
    pub fn dot_product_asymmetric(&self, query_f32: &[f32]) -> Result<f32> {
        if self.data.len() != query_f32.len() {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.data.len(),
                found: query_f32.len(),
            });
        }

        if self.scale == 0.0 {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;
        let chunks_q = self.data.chunks_exact(8);
        let chunks_f = query_f32.chunks_exact(8);
        let rem_q = chunks_q.remainder();
        let rem_f = chunks_f.remainder();

        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;

        for (cq, cf) in chunks_q.zip(chunks_f) {
            s0 += (cq[0] as f32) * cf[0] + (cq[1] as f32) * cf[1];
            s1 += (cq[2] as f32) * cf[2] + (cq[3] as f32) * cf[3];
            s2 += (cq[4] as f32) * cf[4] + (cq[5] as f32) * cf[5];
            s3 += (cq[6] as f32) * cf[6] + (cq[7] as f32) * cf[7];
        }

        sum += (s0 + s1) + (s2 + s3);

        for (q, f) in rem_q.iter().zip(rem_f.iter()) {
            sum += (*q as f32) * f;
        }

        Ok(sum * self.scale)
    }

    /// Symmetric Dot Product: computes the dot product directly between two quantized `i8` vectors.
    pub fn dot_product_symmetric(&self, other: &Self) -> Result<f32> {
        if self.data.len() != other.data.len() {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.data.len(),
                found: other.data.len(),
            });
        }

        let mut int_sum: i32 = 0;
        let chunks_a = self.data.chunks_exact(8);
        let chunks_b = other.data.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        let mut s0: i32 = 0;
        let mut s1: i32 = 0;

        for (ca, cb) in chunks_a.zip(chunks_b) {
            s0 += (ca[0] as i32) * (cb[0] as i32)
                + (ca[1] as i32) * (cb[1] as i32)
                + (ca[2] as i32) * (cb[2] as i32)
                + (ca[3] as i32) * (cb[3] as i32);
            s1 += (ca[4] as i32) * (cb[4] as i32)
                + (ca[5] as i32) * (cb[5] as i32)
                + (ca[6] as i32) * (cb[6] as i32)
                + (ca[7] as i32) * (cb[7] as i32);
        }

        int_sum += s0 + s1;

        for (a, b) in rem_a.iter().zip(rem_b.iter()) {
            int_sum += (*a as i32) * (*b as i32);
        }

        Ok((int_sum as f32) * (self.scale * other.scale))
    }

    /// Asymmetric Cosine Similarity: query in `f32` vs stored `QuantizedVector`.
    pub fn cosine_similarity_asymmetric(&self, query_f32: &[f32], query_norm: f32) -> Result<f32> {
        let dot = self.dot_product_asymmetric(query_f32)?;
        let denominator = self.norm * query_norm;

        if denominator < 1e-12 {
            Ok(0.0)
        } else {
            Ok((dot / denominator).clamp(-1.0, 1.0))
        }
    }

    /// Returns the vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Returns the byte footprint of this quantized vector.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::distance::{cosine_similarity, dot_product};

    #[test]
    fn test_quantize_dequantize_precision() {
        let original: Vec<f32> = (0..384).map(|i| ((i as f32) * 0.05 - 9.0).sin()).collect();
        let q = QuantizedVector::quantize(&original);

        assert_eq!(q.dim(), 384);
        let reconstructed = q.dequantize();

        // Check that maximum reconstruction error is small (due to 127 quantization bins)
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.02);
        }
    }

    #[test]
    fn test_asymmetric_dot_product_accuracy() {
        let v1: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let v2: Vec<f32> = (0..384).map(|i| (i as f32) * -0.02 + 0.5).collect();

        let q1 = QuantizedVector::quantize(&v1);
        let exact_dot = dot_product(&v1, &v2).unwrap();
        let approx_dot = q1.dot_product_asymmetric(&v2).unwrap();

        let rel_error = (exact_dot - approx_dot).abs() / exact_dot.abs().max(1.0);
        // Error should be less than 0.5%
        assert!(rel_error < 0.005, "Relative error: {}", rel_error);
    }

    #[test]
    fn test_asymmetric_cosine_similarity_accuracy() {
        let v1: Vec<f32> = (0..768).map(|i| ((i * 3 % 100) as f32) - 50.0).collect();
        let v2: Vec<f32> = (0..768).map(|i| ((i * 7 % 100) as f32) - 40.0).collect();

        let q1 = QuantizedVector::quantize(&v1);
        let exact_cos = cosine_similarity(&v1, &v2).unwrap();
        let approx_cos = q1.cosine_similarity_asymmetric(&v2, norm(&v2)).unwrap();

        assert!((exact_cos - approx_cos).abs() < 0.01);
    }

    #[test]
    fn test_symmetric_dot_product() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![2.0, 0.5, 1.0, 2.0];

        let q1 = QuantizedVector::quantize(&v1);
        let q2 = QuantizedVector::quantize(&v2);

        let exact = dot_product(&v1, &v2).unwrap();
        let sym = q1.dot_product_symmetric(&q2).unwrap();

        assert!((exact - sym).abs() < 0.1);
    }

    #[test]
    fn test_zero_vector_quantization() {
        let zero = vec![0.0f32; 128];
        let q = QuantizedVector::quantize(&zero);
        assert_eq!(q.scale, 0.0);
        assert_eq!(q.norm, 0.0);
        assert_eq!(q.dequantize(), zero);
    }
}
