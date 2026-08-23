//! In-memory local Cross-Encoder Reranker using embedded ONNX models (FastEmbed).

#[cfg(feature = "fastembed")]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
#[cfg(feature = "fastembed")]
use parking_lot::Mutex;

#[cfg(feature = "fastembed")]
use crate::error::{GraphLiteError, Result};

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

/// Local in-memory Cross-Encoder Reranker for deep semantic re-ranking.
pub struct LocalReranker {
    #[cfg(feature = "fastembed")]
    model: Mutex<TextRerank>,
}

impl LocalReranker {
    /// Initializes a new local reranker using `bge-reranker-base`.
    #[cfg(feature = "fastembed")]
    pub fn new_bge_base() -> Result<Self> {
        let mut options = RerankInitOptions::default();
        options.model_name = RerankerModel::BGERerankerBase;
        options.show_download_progress = false;

        let model = TextRerank::try_new(options)
            .map_err(|e| GraphLiteError::Io(std::io::Error::other(e.to_string())))?;

        Ok(Self {
            model: Mutex::new(model),
        })
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
        let mut guard = self.model.lock();

        let results = guard
            .rerank(query, doc_strs, true, None)
            .map_err(|e| GraphLiteError::Io(std::io::Error::other(e.to_string())))?;

        let mapped = results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: sigmoid(r.score),
                raw_score: r.score,
            })
            .collect();

        Ok(mapped)
    }
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
}
