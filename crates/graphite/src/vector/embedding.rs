//! In-memory local text embedding engine using embedded ONNX models (FastEmbed).

use std::collections::HashMap;

#[cfg(feature = "fastembed")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(feature = "fastembed")]
use parking_lot::Mutex;

#[cfg(feature = "fastembed")]
use crate::error::{GraphiteError, Result};

/// Supported local embedding model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModelType {
    /// `all-MiniLM-L6-v2` (384 dimensions, ~90 MB) - Default / Fast / English
    AllMiniLML6V2,
    /// `BGE-Small-ENV1.5` (384 dimensions, ~130 MB) - High accuracy / English
    BGESmallENV15,
    /// `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, ~470 MB) - 50+ languages (Portuguese)
    MultilingualMiniLML12V2,
    /// `multilingual-e5-base` (768 dimensions, ~1.1 GB) - 100+ languages / Legal & Technical
    MultilingualE5Base,
    /// `bge-m3` (1024 dimensions, ~2.2 GB) - Multilingual SOTA / 8192 context length
    BGEM3,
    /// `nomic-embed-text-v1.5` (768 dimensions, ~550 MB) - Code / Text
    NomicEmbedTextV15,
    /// Custom external dimension (no local FastEmbed model instance)
    Custom(usize),
}

impl EmbeddingModelType {
    /// Returns the embedding dimensionality for this model type.
    pub fn dimension(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 384,
            Self::BGESmallENV15 => 384,
            Self::MultilingualMiniLML12V2 => 384,
            Self::MultilingualE5Base => 768,
            Self::BGEM3 => 1024,
            Self::NomicEmbedTextV15 => 768,
            Self::Custom(dim) => *dim,
        }
    }

    /// Returns the unique numeric ID stored in `.graph` binary header (0..255).
    pub fn id(&self) -> u8 {
        match self {
            Self::AllMiniLML6V2 => crate::storage::header::EMBEDDING_MODEL_ALL_MINILM_L6_V2,
            Self::BGESmallENV15 => crate::storage::header::EMBEDDING_MODEL_BGE_SMALL_EN_V15,
            Self::MultilingualMiniLML12V2 => {
                crate::storage::header::EMBEDDING_MODEL_MULTILINGUAL_MINILM_L12_V2
            }
            Self::MultilingualE5Base => {
                crate::storage::header::EMBEDDING_MODEL_MULTILINGUAL_E5_BASE
            }
            Self::BGEM3 => crate::storage::header::EMBEDDING_MODEL_BGE_M3,
            Self::NomicEmbedTextV15 => crate::storage::header::EMBEDDING_MODEL_NOMIC_EMBED_TEXT_V15,
            Self::Custom(_) => crate::storage::header::EMBEDDING_MODEL_CUSTOM,
        }
    }

    /// Resolves an embedding model type from its header ID and vector dimension.
    pub fn from_id(id: u8, dim: usize) -> Self {
        match id {
            crate::storage::header::EMBEDDING_MODEL_ALL_MINILM_L6_V2 => Self::AllMiniLML6V2,
            crate::storage::header::EMBEDDING_MODEL_BGE_SMALL_EN_V15 => Self::BGESmallENV15,
            crate::storage::header::EMBEDDING_MODEL_MULTILINGUAL_MINILM_L12_V2 => {
                Self::MultilingualMiniLML12V2
            }
            crate::storage::header::EMBEDDING_MODEL_MULTILINGUAL_E5_BASE => {
                Self::MultilingualE5Base
            }
            crate::storage::header::EMBEDDING_MODEL_BGE_M3 => Self::BGEM3,
            crate::storage::header::EMBEDDING_MODEL_NOMIC_EMBED_TEXT_V15 => Self::NomicEmbedTextV15,
            _ => Self::Custom(dim),
        }
    }

    /// Resolves an embedding model type from a CLI string identifier.
    pub fn from_str_name(name: &str) -> Option<Self> {
        match name.to_lowercase().replace('_', "-").as_str() {
            "all-minilm-l6-v2" | "minilm" | "minilm-l6" | "default" => Some(Self::AllMiniLML6V2),
            "bge-small-en-v1.5" | "bge-small" => Some(Self::BGESmallENV15),
            "paraphrase-multilingual-minilm-l12-v2"
            | "multilingual-minilm-l12-v2"
            | "multilingual-minilm"
            | "minilm-l12" => Some(Self::MultilingualMiniLML12V2),
            "multilingual-e5-base" | "e5-base" | "e5" => Some(Self::MultilingualE5Base),
            "bge-m3" | "m3" => Some(Self::BGEM3),
            "nomic-embed-text-v1.5" | "nomic" => Some(Self::NomicEmbedTextV15),
            _ => None,
        }
    }

    /// Returns the user-friendly CLI identifier string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2 => "all-minilm-l6-v2",
            Self::BGESmallENV15 => "bge-small-en-v1.5",
            Self::MultilingualMiniLML12V2 => "paraphrase-multilingual-minilm-l12-v2",
            Self::MultilingualE5Base => "multilingual-e5-base",
            Self::BGEM3 => "bge-m3",
            Self::NomicEmbedTextV15 => "nomic-embed-text-v1.5",
            Self::Custom(_) => "custom",
        }
    }

    /// Returns a human-readable display string for interactive menus.
    pub fn display_label(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2 => "all-MiniLM-L6-v2                 384d  ·   90 MB   (Default, Fast)",
            Self::MultilingualMiniLML12V2 => {
                "paraphrase-multilingual-minilm   384d  ·  470 MB   (Multilingual / Portuguese)"
            }
            Self::MultilingualE5Base => {
                "multilingual-e5-base             768d  ·  1.1 GB   (Multilingual / Legal & Tech)"
            }
            Self::BGEM3 => {
                "bge-m3                          1024d  ·  2.2 GB   (SOTA Multilingual / 8k Context)"
            }
            Self::BGESmallENV15 => {
                "bge-small-en-v1.5                384d  ·  130 MB   (High Accuracy / English)"
            }
            Self::NomicEmbedTextV15 => {
                "nomic-embed-text-v1.5            768d  ·  550 MB   (Code & Text)"
            }
            Self::Custom(_) => "Custom                                     (Manual dimension)",
        }
    }

    /// Returns `true` if this model's ONNX weights are already downloaded and cached locally.
    pub fn is_cached(&self) -> bool {
        let pattern = match self {
            Self::AllMiniLML6V2 => "all-MiniLM-L6-v2",
            Self::BGESmallENV15 => "bge-small-en",
            Self::MultilingualMiniLML12V2 => "paraphrase-multilingual-MiniLM-L12",
            Self::MultilingualE5Base => "multilingual-e5-base",
            Self::BGEM3 => "bge-m3",
            Self::NomicEmbedTextV15 => "nomic-embed-text",
            Self::Custom(_) => return true,
        };

        is_model_cached(pattern)
    }
}

/// Returns the global standard user cache directory for storing ONNX model files (~/.cache/graphite/models).
pub fn default_model_cache_dir() -> std::path::PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home)
            .join(".cache")
            .join("graphite")
            .join("models")
    } else if let Ok(userprofile) = std::env::var("USERPROFILE") {
        std::path::PathBuf::from(userprofile)
            .join(".cache")
            .join("graphite")
            .join("models")
    } else {
        std::env::temp_dir().join("graphite_models")
    }
}

/// Checks if an ONNX model matching the given pattern is already downloaded in the cache.
pub fn is_model_cached(pattern: &str) -> bool {
    let cache_dir = default_model_cache_dir();
    if !cache_dir.exists() {
        return false;
    }

    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.to_lowercase().contains(&pattern.to_lowercase()) {
                let snapshots = entry.path().join("snapshots");
                if snapshots.exists() {
                    if let Ok(snap_entries) = std::fs::read_dir(snapshots) {
                        for snap in snap_entries.flatten() {
                            let model_file = snap.path().join("model.onnx");
                            if model_file.exists() {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

/// Recursively removes stale `.lock` files from model cache directories.
pub fn clean_stale_lock_files(dir: &std::path::Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                clean_stale_lock_files(&path);
            } else if path.extension().is_some_and(|ext| ext == "lock") {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// Default in-memory query cache capacity.
const DEFAULT_QUERY_CACHE_CAPACITY: usize = 512;

/// Local in-memory embedding model runner for computing vector embeddings in pure Rust.
pub struct LocalEmbedder {
    #[cfg(feature = "fastembed")]
    model: Mutex<TextEmbedding>,
    dim: usize,
    model_type: EmbeddingModelType,
    device: crate::vector::DeviceType,
    query_cache: Mutex<HashMap<u64, Vec<f32>>>,
}

impl LocalEmbedder {
    /// Initializes an embedder from an `EmbeddingModelType` using automatic device detection.
    #[cfg(feature = "fastembed")]
    pub fn from_model_type(model_type: EmbeddingModelType) -> Result<Self> {
        Self::from_model_type_and_device(model_type, crate::vector::DeviceType::Auto)
    }

    /// Initializes an embedder from an `EmbeddingModelType` on a specified execution device (CPU or CUDA).
    #[cfg(feature = "fastembed")]
    pub fn from_model_type_and_device(
        model_type: EmbeddingModelType,
        device: crate::vector::DeviceType,
    ) -> Result<Self> {
        let resolved_device = device.resolve();
        let (fastembed_model, dim) = match model_type {
            EmbeddingModelType::AllMiniLML6V2 => (EmbeddingModel::AllMiniLML6V2, 384),
            EmbeddingModelType::BGESmallENV15 => (EmbeddingModel::BGESmallENV15, 384),
            EmbeddingModelType::MultilingualMiniLML12V2 => {
                (EmbeddingModel::ParaphraseMLMiniLML12V2, 384)
            }
            EmbeddingModelType::MultilingualE5Base => (EmbeddingModel::MultilingualE5Base, 768),
            EmbeddingModelType::BGEM3 => (EmbeddingModel::BGEM3, 1024),
            EmbeddingModelType::NomicEmbedTextV15 => (EmbeddingModel::NomicEmbedTextV15, 768),
            EmbeddingModelType::Custom(d) => (EmbeddingModel::AllMiniLML6V2, d),
        };

        let cached = model_type.is_cached();
        let mut options = InitOptions::default();
        options.model_name = fastembed_model;
        options.show_download_progress = !cached;
        let cache_dir = default_model_cache_dir();
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
        clean_stale_lock_files(&cache_dir);

        let model = match TextEmbedding::try_new(options.clone()) {
            Ok(m) => m,
            Err(orig_err) => {
                // If direct download failed, automatically retry with mirror
                if std::env::var("HF_ENDPOINT").is_err() {
                    std::env::set_var("HF_ENDPOINT", "https://hf-mirror.com");
                    clean_stale_lock_files(&cache_dir);
                    TextEmbedding::try_new(options).map_err(|e| {
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

        Ok(Self {
            model: Mutex::new(model),
            dim,
            model_type,
            device: resolved_device,
            query_cache: Mutex::new(HashMap::with_capacity(DEFAULT_QUERY_CACHE_CAPACITY)),
        })
    }

    /// Initializes a new local embedder using `all-MiniLM-L6-v2` (384 dimensions).
    #[cfg(feature = "fastembed")]
    pub fn new_minilm() -> Result<Self> {
        Self::from_model_type(EmbeddingModelType::AllMiniLML6V2)
    }

    /// Initializes a new local embedder using `BGE-Small-ENV1.5` (384 dimensions).
    #[cfg(feature = "fastembed")]
    pub fn new_bge_small() -> Result<Self> {
        Self::from_model_type(EmbeddingModelType::BGESmallENV15)
    }

    /// Returns the execution device used by this embedder.
    pub fn device(&self) -> crate::vector::DeviceType {
        self.device
    }

    /// Embeds a single text prompt into a normalized `Vec<f32>` with 0ms in-memory query cache.
    #[cfg(feature = "fastembed")]
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }

        // 1. Check in-memory query vector cache (instant 0.00ms hit)
        let hash_key = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            trimmed.hash(&mut hasher);
            hasher.finish()
        };

        {
            let cache = self.query_cache.lock();
            if let Some(cached_vec) = cache.get(&hash_key) {
                return Ok(cached_vec.clone());
            }
        }

        // 2. Perform ONNX tensor inference
        let mut guard = self.model.lock();
        let embeddings = guard
            .embed(vec![trimmed], None)
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        let vector = embeddings.into_iter().next().ok_or_else(|| {
            GraphiteError::Io(std::io::Error::other(
                "Embedding model produced empty output vector",
            ))
        })?;

        // 3. Cache the normalized query vector for instant retrieval
        let mut cache = self.query_cache.lock();
        if cache.len() >= DEFAULT_QUERY_CACHE_CAPACITY {
            cache.clear();
        }
        cache.insert(hash_key, vector.clone());

        Ok(vector)
    }

    /// Batch embeds a slice of text strings in parallel chunks.
    #[cfg(feature = "fastembed")]
    pub fn embed_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let batch_size = if self.device.is_cuda() { 256 } else { 64 };
        let mut guard = self.model.lock();

        let embeddings = guard
            .embed(str_refs, Some(batch_size))
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        Ok(embeddings)
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the model type configured for this embedder.
    pub fn model_type(&self) -> EmbeddingModelType {
        self.model_type
    }
}
