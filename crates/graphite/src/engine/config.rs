use crate::error::{GraphiteError, Result};
use crate::graph::hybrid_score::HybridScoreConfig;
use crate::graph::traversal::{TraversalConfig, TraversalDirection};
use crate::vector::distance::Metric;
use crate::vector::quantization::Quantization;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Comprehensive configuration and builder for the Graphite database engine.
///
/// # Example
///
/// ```rust
/// use graphite::engine::GraphiteConfig;
/// use graphite::vector::Metric;
/// use graphite::vector::Quantization;
///
/// let config = GraphiteConfig::new()
///     .with_dim(384)
///     .with_metric(Metric::Cosine)
///     .with_quantization(Quantization::ScalarInt8)
///     .with_threshold(0.70);
///
/// assert_eq!(config.vector_dim, 384);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphiteConfig {
    /// Dimensionality of the embedding vectors (e.g. 384, 512, 1536).
    pub vector_dim: usize,
    /// Distance/similarity metric used for vector search.
    pub metric: Metric,
    /// Vector quantization mode (full-precision Float32 vs 8-bit SQ8).
    pub quantization: Quantization,
    /// Configuration for hybrid vector + graph relevance scoring.
    pub hybrid_config: HybridScoreConfig,
    /// Configuration for multi-hop graph BFS traversal.
    pub traversal_config: TraversalConfig,
    /// Whether to automatically persist mutations to disk on commit/save.
    pub auto_flush: bool,
    /// Whether in-memory LRU query context caching is enabled.
    pub enable_cache: bool,
    /// Maximum number of cached query context entries in the LRU cache.
    pub cache_capacity: usize,
    /// Whether to write directly to destination file without temporary staging files (.tmp).
    pub direct_write: bool,
    /// MMR (Maximal Marginal Relevance) diversity parameter $\lambda \in [0.0, 1.0]$.
    /// - 1.0: 100% Relevance, no diversity penalty.
    /// - 0.75: Balanced relevance with duplicate suppression (default).
    pub mmr_lambda: f32,
    /// Configured local Embedding Model identifier (0: all-MiniLM-L6-v2, 1: bge-small, etc.).
    pub embedding_model_id: u8,
    /// Configured local Reranker Model identifier (0: none, 1: bge-reranker-base, etc.).
    pub reranker_model_id: u8,
    /// Target hardware acceleration device (Auto, Cpu, Cuda).
    pub device: crate::vector::DeviceType,
}

impl Default for GraphiteConfig {
    fn default() -> Self {
        Self {
            vector_dim: 384,
            metric: Metric::Cosine,
            quantization: Quantization::ScalarInt8,
            hybrid_config: HybridScoreConfig {
                alpha: 0.6,
                depth_decay: 0.85,
                min_score_threshold: 0.05,
                relative_drop_off: None,
                use_rrf: true,
                rrf_k: 60,
            },
            traversal_config: TraversalConfig {
                max_depth: 2,
                min_edge_weight: 0.5,
                max_nodes: 100,
                direction: TraversalDirection::Outgoing,
            },
            auto_flush: true,
            enable_cache: true,
            cache_capacity: 1000,
            direct_write: false,
            mmr_lambda: 0.75,
            embedding_model_id: 0, // all-MiniLM-L6-v2
            reranker_model_id: 1,  // bge-reranker-base
            device: crate::vector::DeviceType::Auto,
        }
    }
}

impl GraphiteConfig {
    /// Creates a new `GraphiteConfig` with standard recommended defaults.
    ///
    /// ```rust
    /// use graphite::engine::GraphiteConfig;
    ///
    /// let config = GraphiteConfig::new();
    /// assert_eq!(config.vector_dim, 384);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the hardware execution device (Auto, Cpu, Cuda).
    pub fn with_device(mut self, device: crate::vector::DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Sets the configured embedding and reranker model identifiers.
    pub fn with_models(mut self, embedding_model_id: u8, reranker_model_id: u8) -> Self {
        self.embedding_model_id = embedding_model_id;
        self.reranker_model_id = reranker_model_id;
        self
    }

    /// Sets the embedding model identifier.
    pub fn with_embedding_model_id(mut self, id: u8) -> Self {
        self.embedding_model_id = id;
        self
    }

    /// Sets the reranker model identifier.
    pub fn with_reranker_model_id(mut self, id: u8) -> Self {
        self.reranker_model_id = id;
        self
    }

    /// Sets the vector embedding dimension.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.vector_dim = dim;
        self
    }

    /// Sets the vector distance metric.
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the quantization mode (e.g. `Quantization::ScalarInt8`).
    pub fn with_quantization(mut self, quant: Quantization) -> Self {
        self.quantization = quant;
        self
    }

    /// Sets the alpha balancing factor ($0.0 \le \alpha \le 1.0$) for hybrid scoring:
    /// $\alpha \cdot \text{Vector} + (1 - \alpha) \cdot \text{Graph}$.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.hybrid_config.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Sets the exponential depth decay factor ($\gamma$) per hop during BFS.
    pub fn with_depth_decay(mut self, decay: f32) -> Self {
        self.hybrid_config.depth_decay = decay.clamp(0.0, 1.0);
        self
    }

    /// Sets the maximum number of hops (depth) explored during graph traversal.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.traversal_config.max_depth = max_depth;
        self
    }

    /// Sets the minimum hybrid score threshold for entities to be included during query retrieval.
    pub fn with_min_score_threshold(mut self, threshold: f32) -> Self {
        self.hybrid_config.min_score_threshold = threshold;
        self
    }

    /// Sets the minimum relevance score threshold for entities to be included during query retrieval (alias).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.hybrid_config.min_score_threshold = threshold;
        self
    }

    /// Sets the minimum edge weight threshold for traversing graph connections.
    pub fn with_min_edge_weight(mut self, min_weight: f32) -> Self {
        self.traversal_config.min_edge_weight = min_weight;
        self
    }

    /// Sets whether changes should automatically be flushed to disk on mutation.
    pub fn with_auto_flush(mut self, auto_flush: bool) -> Self {
        self.auto_flush = auto_flush;
        self
    }

    /// Sets whether the query context LRU cache is enabled.
    pub fn with_cache(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }

    /// Sets the maximum number of query context items stored in the LRU cache.
    pub fn with_cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity;
        self
    }

    /// Sets whether to write directly to destination file without temporary staging files (.tmp).
    pub fn with_direct_write(mut self, direct_write: bool) -> Self {
        self.direct_write = direct_write;
        self
    }

    /// Sets the MMR (Maximal Marginal Relevance) diversity parameter $\lambda \in [0.0, 1.0]$.
    pub fn with_mmr_lambda(mut self, lambda: f32) -> Self {
        self.mmr_lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Sets whether to use Reciprocal Rank Fusion (RRF) for dense + sparse rank fusion.
    pub fn with_rrf(mut self, use_rrf: bool) -> Self {
        self.hybrid_config.use_rrf = use_rrf;
        self
    }

    /// Validates the configuration parameters for internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.vector_dim == 0 {
            return Err(GraphiteError::CorruptedFormat(
                "Vector dimension must be greater than zero".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.hybrid_config.alpha) {
            return Err(GraphiteError::CorruptedFormat(
                "Hybrid score alpha must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_pattern() {
        let config = GraphiteConfig::new()
            .with_dim(1536)
            .with_metric(Metric::DotProduct)
            .with_quantization(Quantization::None)
            .with_threshold(0.75)
            .with_alpha(0.7)
            .with_max_depth(3)
            .with_auto_flush(false);

        assert_eq!(config.vector_dim, 1536);
        assert_eq!(config.metric, Metric::DotProduct);
        assert_eq!(config.quantization, Quantization::None);
        assert_eq!(config.hybrid_config.min_score_threshold, 0.75);
        assert_eq!(config.hybrid_config.alpha, 0.7);
        assert_eq!(config.traversal_config.max_depth, 3);
        assert!(!config.auto_flush);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_errors() {
        let invalid_dim = GraphiteConfig::new().with_dim(0);
        assert!(invalid_dim.validate().is_err());
    }
}
