//! In-memory local Cross-Encoder Reranker using embedded ONNX models (FastEmbed).

#[cfg(feature = "fastembed")]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
#[cfg(feature = "fastembed")]
use parking_lot::Mutex;

#[cfg(feature = "fastembed")]
use crate::error::{GraphiteError, Result};

/// Sigmoid activation function to normalize raw Cross-Encoder logits into a [0.0, 1.0] probability.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A scored rerank candidate with its original index and normalized cross-encoder relevance score [0.0, 1.0].
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// Zero-based index into the original input candidate array.
    pub index: usize,
    /// Cross-encoder normalized relevance score between 0.0 and 1.0.
    pub score: f32,
    /// Raw unnormalized logit from the neural network.
    pub raw_score: f32,
}

/// Supported local Cross-Encoder reranker model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerModelType {
    /// Reranking disabled (pure vector + graph + BM25 RRF fusion)
    None,
    /// `BAAI/bge-reranker-base` (~1.11 GB) - Default / High accuracy multilingual
    BGERerankerBase,
    /// `BAAI/bge-reranker-v2-m3` (~2.2 GB) - Multilingual SOTA / 8192 context length
    BGERerankerV2M3,
    /// `jinaai/jina-reranker-v1-turbo-en` (~130 MB) - Ultralight CPU reranker
    JinaRerankerV1TurboEn,
    /// `jinaai/jina-reranker-v2-base-multilingual` (~1.1 GB) - Multilingual SOTA / Portuguese
    JinaRerankerV2BaseMultilingual,
    /// Custom external reranker
    Custom,
}

impl RerankerModelType {
    /// Returns the unique numeric ID stored in `.graph` binary header (0..255).
    pub fn id(&self) -> u8 {
        match self {
            Self::None => crate::storage::header::RERANKER_MODEL_NONE,
            Self::BGERerankerBase => crate::storage::header::RERANKER_MODEL_BGE_RERANKER_BASE,
            Self::BGERerankerV2M3 => crate::storage::header::RERANKER_MODEL_BGE_RERANKER_LARGE,
            Self::JinaRerankerV1TurboEn => {
                crate::storage::header::RERANKER_MODEL_JINA_RERANKER_V1_TINY_EN
            }
            Self::JinaRerankerV2BaseMultilingual => {
                crate::storage::header::RERANKER_MODEL_JINA_RERANKER_V2_BASE_MULTILINGUAL
            }
            Self::Custom => crate::storage::header::RERANKER_MODEL_CUSTOM,
        }
    }

    /// Resolves a reranker model type from its header ID.
    pub fn from_id(id: u8) -> Self {
        match id {
            crate::storage::header::RERANKER_MODEL_NONE => Self::None,
            crate::storage::header::RERANKER_MODEL_BGE_RERANKER_BASE => Self::BGERerankerBase,
            crate::storage::header::RERANKER_MODEL_BGE_RERANKER_LARGE => Self::BGERerankerV2M3,
            crate::storage::header::RERANKER_MODEL_JINA_RERANKER_V1_TINY_EN => {
                Self::JinaRerankerV1TurboEn
            }
            crate::storage::header::RERANKER_MODEL_JINA_RERANKER_V2_BASE_MULTILINGUAL => {
                Self::JinaRerankerV2BaseMultilingual
            }
            _ => Self::None,
        }
    }

    /// Resolves a reranker model type from a CLI string identifier.
    pub fn from_str_name(name: &str) -> Option<Self> {
        match name.to_lowercase().replace('_', "-").as_str() {
            "none" | "disabled" | "off" => Some(Self::None),
            "bge-reranker-base" | "bge-base" | "default" => Some(Self::BGERerankerBase),
            "bge-reranker-v2-m3" | "bge-m3" | "bge-large" => Some(Self::BGERerankerV2M3),
            "jina-reranker-v1-turbo-en" | "jina-tiny" | "jina-turbo" => {
                Some(Self::JinaRerankerV1TurboEn)
            }
            "jina-reranker-v2-base-multilingual" | "jina-v2" | "jina-multilingual" => {
                Some(Self::JinaRerankerV2BaseMultilingual)
            }
            _ => None,
        }
    }

    /// Returns the user-friendly CLI identifier string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BGERerankerBase => "bge-reranker-base",
            Self::BGERerankerV2M3 => "bge-reranker-v2-m3",
            Self::JinaRerankerV1TurboEn => "jina-reranker-v1-turbo-en",
            Self::JinaRerankerV2BaseMultilingual => "jina-reranker-v2-base-multilingual",
            Self::Custom => "custom",
        }
    }

    /// Returns a human-readable display string for interactive menus.
    pub fn display_label(&self) -> &'static str {
        match self {
            Self::BGERerankerBase => {
                "bge-reranker-base               1.1 GB   (Recommended, Multilingual)"
            }
            Self::JinaRerankerV2BaseMultilingual => {
                "jina-reranker-v2-multilingual   1.1 GB   (SOTA Multilingual / Portuguese)"
            }
            Self::BGERerankerV2M3 => {
                "bge-reranker-v2-m3              2.2 GB   (SOTA Multilingual / 8k Context)"
            }
            Self::JinaRerankerV1TurboEn => {
                "jina-reranker-v1-turbo          130 MB   (Ultralight / Low CPU)"
            }
            Self::None => {
                "None                                     (Fast Hybrid Retrieval, < 10ms)"
            }
            Self::Custom => "Custom",
        }
    }

    /// Returns `true` if this reranker model's ONNX weights are already downloaded and cached locally.
    pub fn is_cached(&self) -> bool {
        let pattern = match self {
            Self::None | Self::Custom => return true,
            Self::BGERerankerBase => "bge-reranker-base",
            Self::BGERerankerV2M3 => "bge-reranker-v2-m3",
            Self::JinaRerankerV1TurboEn => "jina-reranker-v1-turbo",
            Self::JinaRerankerV2BaseMultilingual => "jina-reranker-v2-base-multilingual",
        };

        crate::vector::embedding::is_model_cached(pattern)
    }
}

/// Local in-memory Cross-Encoder Reranker for deep semantic re-ranking.
pub struct LocalReranker {
    #[cfg(feature = "fastembed")]
    model: Mutex<TextRerank>,
    model_type: RerankerModelType,
    device: crate::vector::DeviceType,
}

impl LocalReranker {
    /// Initializes a local reranker from a `RerankerModelType` using automatic device detection.
    #[cfg(feature = "fastembed")]
    pub fn from_model_type(model_type: RerankerModelType) -> Result<Option<Self>> {
        Self::from_model_type_and_device(model_type, crate::vector::DeviceType::Auto)
    }

    /// Initializes a local reranker from a `RerankerModelType` on a specified execution device (CPU or CUDA).
    #[cfg(feature = "fastembed")]
    pub fn from_model_type_and_device(
        model_type: RerankerModelType,
        device: crate::vector::DeviceType,
    ) -> Result<Option<Self>> {
        let fastembed_model = match model_type {
            RerankerModelType::None | RerankerModelType::Custom => return Ok(None),
            RerankerModelType::BGERerankerBase => RerankerModel::BGERerankerBase,
            RerankerModelType::BGERerankerV2M3 => RerankerModel::BGERerankerV2M3,
            RerankerModelType::JinaRerankerV1TurboEn => RerankerModel::JINARerankerV1TurboEn,
            RerankerModelType::JinaRerankerV2BaseMultilingual => {
                RerankerModel::JINARerankerV2BaseMultiligual
            }
        };

        let resolved_device = device.resolve();
        let cached = model_type.is_cached();
        let mut options = RerankInitOptions::default();
        options.model_name = fastembed_model;
        options.show_download_progress = !cached;
        let cache_dir = crate::vector::embedding::default_model_cache_dir();
        options.cache_dir = cache_dir.clone();

        #[cfg(feature = "cuda")]
        if resolved_device.is_cuda() {
            let cuda_ep = ort::execution_providers::CUDA::default()
                .with_device_id(0)
                .with_arena_extend_strategy(
                    ort::execution_providers::ArenaExtendStrategy::SameAsRequested,
                )
                .with_cuda_graph(true)
                .build();
            options.execution_providers = vec![
                cuda_ep,
                ort::execution_providers::CPU::default().build(),
            ];
        }

        #[cfg(not(feature = "cuda"))]
        if resolved_device.is_cuda() {
            // Graphite was compiled without static CUDA feature; defaults gracefully to high-performance CPU SIMD
        }

        // Clean any stale .lock files from interrupted downloads recursively
        crate::vector::embedding::clean_stale_lock_files(&cache_dir);

        let model = match TextRerank::try_new(options.clone()) {
            Ok(m) => m,
            Err(orig_err) => {
                // If direct download failed, automatically retry with mirror
                if std::env::var("HF_ENDPOINT").is_err() {
                    std::env::set_var("HF_ENDPOINT", "https://hf-mirror.com");
                    crate::vector::embedding::clean_stale_lock_files(&cache_dir);
                    TextRerank::try_new(options).map_err(|e| {
                        GraphiteError::Io(std::io::Error::other(format!(
                            "{}. Mirror retry failed: {}",
                            orig_err, e
                        )))
                    })?
                } else {
                    return Err(GraphiteError::Io(std::io::Error::other(
                        orig_err.to_string(),
                    )));
                }
            }
        };

        Ok(Some(Self {
            model: Mutex::new(model),
            model_type,
            device: resolved_device,
        }))
    }

    /// Initializes a new local reranker using `bge-reranker-base`.
    #[cfg(feature = "fastembed")]
    pub fn new_bge_base() -> Result<Self> {
        Self::from_model_type(RerankerModelType::BGERerankerBase).and_then(|opt| {
            opt.ok_or_else(|| {
                GraphiteError::Io(std::io::Error::other("Failed to create bge-reranker-base"))
            })
        })
    }

    /// Returns the model type configured for this reranker.
    pub fn model_type(&self) -> RerankerModelType {
        self.model_type
    }

    /// Returns the execution device used by this reranker.
    pub fn device(&self) -> crate::vector::DeviceType {
        self.device
    }

    /// Reranks a slice of candidate documents against a query string.
    ///
    /// Returns a list of `RerankResult` sorted in descending order of normalized relevance score.
    #[cfg(feature = "fastembed")]
    pub fn rerank<S: AsRef<str>>(&self, query: &str, documents: &[S]) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let doc_strs: Vec<&str> = documents.iter().map(|s| s.as_ref()).collect();
        let batch_size = if self.device.is_cuda() { 128 } else { 32 };
        let mut guard = self.model.lock();

        let results = guard
            .rerank(query, doc_strs, true, Some(batch_size))
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        let mapped = results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: calibrate_reranker_score(r.score),
                raw_score: r.score,
            })
            .collect();

        Ok(mapped)
    }
}

/// Calibrates raw Cross-Encoder model logits into an intuitive [0.0, 1.0] relevance probability.
#[inline]
pub fn calibrate_reranker_score(raw_logit: f32) -> f32 {
    // Cross-encoder models (BGE / Jina) output logits centered around -3.0 for general domain text.
    // We calibrate with shift offset -2.8 and temperature 1.6 to map confident matches to 0.75 - 0.98.
    let shifted = (raw_logit + 2.8) / 1.6;
    sigmoid(shifted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_normalization() {
        // Zero logit maps to 0.5 probability
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
        // Positive logit maps to > 0.5
        assert!(sigmoid(2.0) > 0.85);
        // Negative logit maps to < 0.5
        assert!(sigmoid(-0.49) > 0.35 && sigmoid(-0.49) < 0.40);
        assert!(sigmoid(-7.59) < 0.001);
    }

    #[test]
    fn test_calibrated_reranker_score() {
        // Confident match (-1.2 logit) maps to ~0.73
        let score = calibrate_reranker_score(-1.2);
        assert!(score > 0.70 && score < 0.76);

        // Strong match (+1.0 logit) maps to ~0.91
        let strong = calibrate_reranker_score(1.0);
        assert!(strong > 0.88 && strong < 0.95);

        // Irrelevant match (-8.0 logit) maps to ~0.03
        let irrelevant = calibrate_reranker_score(-8.0);
        assert!(irrelevant < 0.05);
    }
}
