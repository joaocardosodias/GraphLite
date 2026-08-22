use std::collections::HashMap;
use rayon::prelude::*;

use crate::error::{GraphLiteError, Result};
use crate::id::NodeId;
use crate::vector::distance::{norm, Metric};
use crate::vector::quantization::{Quantization, QuantizedVector};
use crate::vector::simd::{simd_cosine_similarity, simd_dot_product, simd_euclidean_distance};

/// An in-memory vector storage engine with parallel SIMD scanning via `rayon`.
///
/// Supports both full-precision Float32 vectors and 8-bit Quantized (SQ8) vectors,
/// storing all data in cache-contiguous buffers for maximum throughput.
#[derive(Debug, Clone)]
pub struct VectorStore {
    /// Embedding dimensionality (e.g. 384, 768, 1536).
    dim: usize,
    /// Distance metric used for similarity calculation.
    metric: Metric,
    /// Quantization mode (Float32 vs ScalarInt8).
    quantization: Quantization,
    /// Parallel array of NodeIds matching internal vector slot indices.
    node_ids: Vec<NodeId>,
    /// Fast mapping from `NodeId` to its internal slot index.
    node_to_idx: HashMap<NodeId, usize>,
    /// Contiguous flat buffer for Float32 vectors (`[N * dim]`).
    flat_f32: Vec<f32>,
    /// Vector of 8-bit quantized records (used when `quantization == Quantization::ScalarInt8`).
    quantized: Vec<QuantizedVector>,
}

impl VectorStore {
    /// Creates a new `VectorStore` with the specified configuration.
    pub fn new(dim: usize, metric: Metric, quantization: Quantization) -> Self {
        Self {
            dim,
            metric,
            quantization,
            node_ids: Vec::new(),
            node_to_idx: HashMap::new(),
            flat_f32: Vec::new(),
            quantized: Vec::new(),
        }
    }

    /// Creates a new `VectorStore` pre-allocated with capacity for $N$ vectors.
    pub fn with_capacity(
        dim: usize,
        metric: Metric,
        quantization: Quantization,
        capacity: usize,
    ) -> Self {
        Self {
            dim,
            metric,
            quantization,
            node_ids: Vec::with_capacity(capacity),
            node_to_idx: HashMap::with_capacity(capacity),
            flat_f32: if quantization == Quantization::None {
                Vec::with_capacity(capacity * dim)
            } else {
                Vec::new()
            },
            quantized: if quantization == Quantization::ScalarInt8 {
                Vec::with_capacity(capacity)
            } else {
                Vec::new()
            },
        }
    }

    /// Inserts or updates an embedding vector for a given `NodeId`.
    pub fn insert(&mut self, node_id: NodeId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.dim,
                found: vector.len(),
            });
        }

        // If node already exists, update in-place
        if let Some(&idx) = self.node_to_idx.get(&node_id) {
            match self.quantization {
                Quantization::None => {
                    let start = idx * self.dim;
                    let end = start + self.dim;
                    self.flat_f32[start..end].copy_from_slice(vector);
                }
                Quantization::ScalarInt8 => {
                    self.quantized[idx] = QuantizedVector::quantize(vector);
                }
            }
            return Ok(());
        }

        // New node insertion
        let idx = self.node_ids.len();
        self.node_ids.push(node_id);
        self.node_to_idx.insert(node_id, idx);

        match self.quantization {
            Quantization::None => {
                self.flat_f32.extend_from_slice(vector);
            }
            Quantization::ScalarInt8 => {
                self.quantized.push(QuantizedVector::quantize(vector));
            }
        }

        Ok(())
    }

    /// Inserts or updates a pre-quantized vector for a given `NodeId`.
    pub fn insert_quantized(&mut self, node_id: NodeId, qv: QuantizedVector) -> Result<()> {
        if qv.data.len() != self.dim {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.dim,
                found: qv.data.len(),
            });
        }

        if let Some(&idx) = self.node_to_idx.get(&node_id) {
            match self.quantization {
                Quantization::ScalarInt8 => {
                    self.quantized[idx] = qv;
                }
                Quantization::None => {
                    let deq = qv.dequantize();
                    let start = idx * self.dim;
                    let end = start + self.dim;
                    self.flat_f32[start..end].copy_from_slice(&deq);
                }
            }
            return Ok(());
        }

        let idx = self.node_ids.len();
        self.node_ids.push(node_id);
        self.node_to_idx.insert(node_id, idx);

        match self.quantization {
            Quantization::ScalarInt8 => {
                self.quantized.push(qv);
            }
            Quantization::None => {
                let deq = qv.dequantize();
                self.flat_f32.extend_from_slice(&deq);
            }
        }

        Ok(())
    }

    /// Retrieves the float embedding vector for a given `NodeId`.
    pub fn get(&self, node_id: NodeId) -> Option<Vec<f32>> {
        let &idx = self.node_to_idx.get(&node_id)?;
        match self.quantization {
            Quantization::None => {
                let start = idx * self.dim;
                let end = start + self.dim;
                Some(self.flat_f32[start..end].to_vec())
            }
            Quantization::ScalarInt8 => Some(self.quantized[idx].dequantize()),
        }
    }

    /// Retrieves the quantized embedding vector for a given `NodeId`.
    pub fn get_quantized(&self, node_id: NodeId) -> Option<QuantizedVector> {
        let &idx = self.node_to_idx.get(&node_id)?;
        match self.quantization {
            Quantization::ScalarInt8 => Some(self.quantized[idx].clone()),
            Quantization::None => {
                let start = idx * self.dim;
                let end = start + self.dim;
                Some(QuantizedVector::quantize(&self.flat_f32[start..end]))
            }
        }
    }

    /// Executes a parallel linear vector scan, returning the Top-K closest nodes.
    ///
    /// Returns a list of `(NodeId, score)` sorted in descending order of relevance.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(NodeId, f32)>> {
        if query.len() != self.dim {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.dim,
                found: query.len(),
            });
        }

        if self.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let query_norm = norm(query);
        let num_items = self.node_ids.len();

        // Calculate scores in parallel across CPU cores using Rayon
        let mut scores: Vec<(NodeId, f32)> = match self.quantization {
            Quantization::None => {
                (0..num_items)
                    .into_par_iter()
                    .map(|idx| {
                        let start = idx * self.dim;
                        let end = start + self.dim;
                        let stored_slice = &self.flat_f32[start..end];
                        let node_id = self.node_ids[idx];

                        let score = match self.metric {
                            Metric::Cosine => {
                                simd_cosine_similarity(query, stored_slice).unwrap_or(0.0)
                            }
                            Metric::DotProduct => {
                                simd_dot_product(query, stored_slice).unwrap_or(0.0)
                            }
                            Metric::Euclidean => {
                                // For Euclidean, lower distance is better, so negate score for descending sort
                                -simd_euclidean_distance(query, stored_slice).unwrap_or(f32::MAX)
                            }
                            Metric::Manhattan => {
                                let dist: f32 = query
                                    .iter()
                                    .zip(stored_slice.iter())
                                    .map(|(a, b)| (a - b).abs())
                                    .sum();
                                -dist
                            }
                        };

                        (node_id, score)
                    })
                    .collect()
            }
            Quantization::ScalarInt8 => {
                (0..num_items)
                    .into_par_iter()
                    .map(|idx| {
                        let q = &self.quantized[idx];
                        let node_id = self.node_ids[idx];

                        let score = match self.metric {
                            Metric::Cosine => {
                                q.cosine_similarity_asymmetric(query, query_norm).unwrap_or(0.0)
                            }
                            Metric::DotProduct => {
                                q.dot_product_asymmetric(query).unwrap_or(0.0)
                            }
                            Metric::Euclidean | Metric::Manhattan => {
                                // Dequantize for distance fallback
                                let deq = q.dequantize();
                                -simd_euclidean_distance(query, &deq).unwrap_or(f32::MAX)
                            }
                        };

                        (node_id, score)
                    })
                    .collect()
            }
        };

        // Partial sort to select Top-K elements with highest scores
        let k = top_k.min(scores.len());
        scores.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        scores.truncate(k);
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // If metric is Euclidean or Manhattan, restore positive distance values
        if self.metric == Metric::Euclidean || self.metric == Metric::Manhattan {
            for (_, score) in scores.iter_mut() {
                *score = -*score;
            }
        }

        Ok(scores)
    }

    /// Returns the number of vectors stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    /// Returns `true` if the vector store contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    /// Returns the vector dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the configured metric.
    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Returns the configured quantization mode.
    #[inline]
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Calculates the approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let ids_mem = self.node_ids.len() * std::mem::size_of::<NodeId>();
        let map_mem = self.node_to_idx.len() * (std::mem::size_of::<NodeId>() + std::mem::size_of::<usize>());
        let payload_mem = match self.quantization {
            Quantization::None => self.flat_f32.len() * std::mem::size_of::<f32>(),
            Quantization::ScalarInt8 => self.quantized.iter().map(|q| q.byte_size()).sum(),
        };

        ids_mem + map_mem + payload_mem
    }

    /// Clears all vectors from the store.
    pub fn clear(&mut self) {
        self.node_ids.clear();
        self.node_to_idx.clear();
        self.flat_f32.clear();
        self.quantized.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_vector_store_insert_get_search() {
        let mut store = VectorStore::new(4, Metric::Cosine, Quantization::None);

        let v1 = [1.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0];
        let v3 = [0.9, 0.1, 0.0, 0.0];

        store.insert(NodeId::new(1), &v1).unwrap();
        store.insert(NodeId::new(2), &v2).unwrap();
        store.insert(NodeId::new(3), &v3).unwrap();

        assert_eq!(store.len(), 3);
        assert_eq!(store.get(NodeId::new(1)), Some(v1.to_vec()));
        assert_eq!(store.get(NodeId::new(99)), None);

        // Query close to v1 and v3
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = store.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, NodeId::new(1)); // Exact match (score ~1.0)
        assert_eq!(results[1].0, NodeId::new(3)); // Close match (score ~0.99)
    }

    #[test]
    fn test_quantized_vector_store_search() {
        let mut store = VectorStore::new(384, Metric::Cosine, Quantization::ScalarInt8);

        let v1: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let v2: Vec<f32> = (0..384).map(|i| -(i as f32) * 0.01).collect();

        store.insert(NodeId::new(10), &v1).unwrap();
        store.insert(NodeId::new(20), &v2).unwrap();

        let query: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let results = store.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(10));
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_update_existing_node() {
        let mut store = VectorStore::new(2, Metric::DotProduct, Quantization::None);

        store.insert(NodeId::new(1), &[1.0, 2.0]).unwrap();
        assert_eq!(store.get(NodeId::new(1)), Some(vec![1.0, 2.0]));

        // Update with new vector
        store.insert(NodeId::new(1), &[3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(NodeId::new(1)), Some(vec![3.0, 4.0]));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = VectorStore::new(3, Metric::Cosine, Quantization::None);
        let bad_vec = [1.0, 2.0];
        assert!(store.insert(NodeId::new(1), &bad_vec).is_err());
        assert!(store.search(&bad_vec, 1).is_err());
    }

    #[test]
    fn test_euclidean_ranking() {
        let mut store = VectorStore::new(2, Metric::Euclidean, Quantization::None);

        store.insert(NodeId::new(1), &[0.0, 0.0]).unwrap();
        store.insert(NodeId::new(2), &[0.0, 3.0]).unwrap();
        store.insert(NodeId::new(3), &[0.0, 10.0]).unwrap();

        let query = [0.0, 0.0];
        let results = store.search(&query, 3).unwrap();

        // Lowest distance first
        assert_eq!(results[0].0, NodeId::new(1));
        assert_eq!(results[0].1, 0.0);
        assert_eq!(results[1].0, NodeId::new(2));
        assert_eq!(results[1].1, 3.0);
        assert_eq!(results[2].0, NodeId::new(3));
        assert_eq!(results[2].1, 10.0);
    }
}
