//! In-memory local text embedding engine using embedded ONNX models (FastEmbed).

#[cfg(feature = "fastembed")]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(feature = "fastembed")]
use parking_lot::Mutex;

#[cfg(feature = "fastembed")]
use crate::error::{GraphiteError, Result};

/// Local in-memory embedding model runner for computing vector embeddings in pure Rust.
pub struct LocalEmbedder {
    #[cfg(feature = "fastembed")]
    model: Mutex<TextEmbedding>,
    dim: usize,
}

/// Returns the global standard user cache directory for storing ONNX model files (~/.cache/graphite/models).
#[cfg(feature = "fastembed")]
fn default_model_cache_dir() -> std::path::PathBuf {
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

impl LocalEmbedder {
    /// Initializes a new local embedder using `all-MiniLM-L6-v2` (384 dimensions).
    #[cfg(feature = "fastembed")]
    pub fn new_minilm() -> Result<Self> {
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::AllMiniLML6V2;
        options.show_download_progress = false;
        options.cache_dir = default_model_cache_dir();

        let model = TextEmbedding::try_new(options)
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        Ok(Self {
            model: Mutex::new(model),
            dim: 384,
        })
    }

    /// Initializes a new local embedder using `BGE-Small-ENV1.5` (384 dimensions).
    #[cfg(feature = "fastembed")]
    pub fn new_bge_small() -> Result<Self> {
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::BGESmallENV15;
        options.show_download_progress = false;
        options.cache_dir = default_model_cache_dir();

        let model = TextEmbedding::try_new(options)
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        Ok(Self {
            model: Mutex::new(model),
            dim: 384,
        })
    }

    /// Embeds a single text prompt into a normalized `Vec<f32>`.
    #[cfg(feature = "fastembed")]
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut guard = self.model.lock();
        let embeddings = guard
            .embed(vec![text], None)
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))?;

        embeddings.into_iter().next().ok_or_else(|| {
            GraphiteError::Io(std::io::Error::other(
                "Empty embedding output from ONNX runtime",
            ))
        })
    }

    /// Embeds a batch of texts into normalized `Vec<Vec<f32>>`.
    #[cfg(feature = "fastembed")]
    pub fn embed_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        let texts_vec: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let mut guard = self.model.lock();
        guard
            .embed(texts_vec, None)
            .map_err(|e| GraphiteError::Io(std::io::Error::other(e.to_string())))
    }

    /// Returns the vector embedding dimensionality (e.g. 384).
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}
