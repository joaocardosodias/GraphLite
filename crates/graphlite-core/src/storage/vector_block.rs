use crate::error::{GraphLiteError, Result};
use crate::vector::quantization::QuantizedVector;

/// Serializes an array of `QuantizedVector`s into a contiguous binary block.
///
/// Binary Layout:
/// - `4 bytes` : `count` (u32, number of vectors)
/// - `2 bytes` : `dim` (u16, dimension per vector)
/// - `2 bytes` : `_reserved` (padding for 8-byte alignment)
/// - Vector payloads contiguously:
///   - Per vector: `scale: f32 (4B)` + `norm: f32 (4B) [módulo do vetor]` + `data: [i8; dim] (dim B)`
pub fn serialize_quantized_vector_block(vectors: &[QuantizedVector], dim: usize) -> Vec<u8> {
    let count = vectors.len() as u32;
    let vector_stride = 8 + dim;
    let total_bytes = 8 + (vectors.len() * vector_stride);
    let mut buffer = Vec::with_capacity(total_bytes);

    buffer.extend_from_slice(&count.to_le_bytes());
    buffer.extend_from_slice(&(dim as u16).to_le_bytes());
    buffer.extend_from_slice(&[0u8; 2]); // alignment padding

    for v in vectors {
        buffer.extend_from_slice(&v.scale.to_le_bytes());
        buffer.extend_from_slice(&v.norm.to_le_bytes()); // módulo do vetor

        // Cast &[i8] to &[u8] for serialization
        let data_u8: &[u8] =
            unsafe { std::slice::from_raw_parts(v.data.as_ptr() as *const u8, v.data.len()) };
        buffer.extend_from_slice(data_u8);
    }

    buffer
}

/// A zero-copy reader over a memory-mapped binary quantized vector block.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyVectorBlock<'a> {
    count: usize,
    dim: usize,
    vector_stride: usize,
    data_slice: &'a [u8],
}

impl<'a> ZeroCopyVectorBlock<'a> {
    /// Creates a `ZeroCopyVectorBlock` from a raw byte slice.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self {
                count: 0,
                dim: 0,
                vector_stride: 0,
                data_slice: &[],
            });
        }

        if bytes.len() < 8 {
            return Err(GraphLiteError::CorruptedFormat(
                "Vector block too short for header".to_string(),
            ));
        }

        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dim = u16::from_le_bytes(bytes[4..6].try_into().unwrap()) as usize;
        let vector_stride = 8 + dim;
        let expected_payload = count * vector_stride;

        if bytes.len() < 8 + expected_payload {
            return Err(GraphLiteError::CorruptedFormat(format!(
                "Vector block payload truncated: expected {} bytes, got {}",
                8 + expected_payload,
                bytes.len()
            )));
        }

        Ok(Self {
            count,
            dim,
            vector_stride,
            data_slice: &bytes[8..8 + expected_payload],
        })
    }

    /// Number of quantized vectors stored in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Dimensionality of the vectors.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns `true` if the vector block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Retrieves a `QuantizedVector` by direct index in $O(1)$ time.
    pub fn get(&self, index: usize) -> Option<QuantizedVector> {
        if index >= self.count {
            return None;
        }

        let start = index * self.vector_stride;
        let end = start + self.vector_stride;
        let slice = &self.data_slice[start..end];

        let scale = f32::from_le_bytes(slice[0..4].try_into().unwrap());
        let norm = f32::from_le_bytes(slice[4..8].try_into().unwrap()); // módulo do vetor

        let data_raw = &slice[8..8 + self.dim];
        let mut data = Vec::with_capacity(self.dim);
        for &b in data_raw {
            data.push(b as i8);
        }

        Some(QuantizedVector { data, scale, norm })
    }
}

/// Deserializes a binary vector block slice into an owned `Vec<QuantizedVector>`.
pub fn deserialize_quantized_vector_block(bytes: &[u8]) -> Result<(Vec<QuantizedVector>, usize)> {
    let viewer = ZeroCopyVectorBlock::from_bytes(bytes)?;
    let mut vectors = Vec::with_capacity(viewer.len());

    for i in 0..viewer.len() {
        if let Some(v) = viewer.get(i) {
            vectors.push(v);
        }
    }

    Ok((vectors, viewer.dim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_vector_block_roundtrip() {
        let v0 = QuantizedVector {
            data: vec![10, -20, 30, -40],
            scale: 0.05,
            norm: 1.0, // módulo do vetor
        };
        let v1 = QuantizedVector {
            data: vec![-50, 60, -70, 80],
            scale: 0.08,
            norm: 2.5,
        };

        let vectors = vec![v0.clone(), v1.clone()];
        let serialized = serialize_quantized_vector_block(&vectors, 4);

        let viewer = ZeroCopyVectorBlock::from_bytes(&serialized).unwrap();
        assert_eq!(viewer.len(), 2);
        assert_eq!(viewer.dim(), 4);

        let r0 = viewer.get(0).unwrap();
        assert_eq!(r0, v0);

        let r1 = viewer.get(1).unwrap();
        assert_eq!(r1, v1);

        assert!(viewer.get(2).is_none());
    }
}
