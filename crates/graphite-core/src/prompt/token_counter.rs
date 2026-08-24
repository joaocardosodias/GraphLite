use std::fmt::Debug;

#[cfg(feature = "tiktoken")]
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, r50k_base, CoreBPE};

/// Common trait for counting tokens in text strings for LLM context budget management.
pub trait TokenCounter: Send + Sync + Debug {
    /// Returns the exact or estimated number of tokens for the given text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// A fast, zero-allocation heuristic token estimator based on character and word boundaries.
///
/// Uses an average ratio of 4 characters per token with word boundary adjustments.
/// Runs in sub-microsecond time with zero dependencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    #[inline]
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        // Rule of thumb: ~4 characters per token for Latin alphabets,
        // with a baseline of 1 token per whitespace word.
        let chars = text.chars().count();
        let words = text.split_whitespace().count();

        // Blend character-based and word-based estimates: max(words, ceil(chars / 4.0))
        let char_based = chars.div_ceil(4);
        words.max(char_based)
    }
}

/// Tokenizer encoding flavors supported by `TiktokenCounter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenizerEncoding {
    /// Standard encoding used by GPT-4, GPT-3.5-Turbo, and text-embedding-ada-002 (cl100k_base).
    #[default]
    Cl100kBase,
    /// Advanced encoding used by GPT-4o and newer frontier models (o200k_base).
    O200kBase,
    /// Code and legacy model encodings (p50k_base).
    P50kBase,
    /// Legacy davinci encodings (r50k_base).
    R50kBase,
}

/// High-precision BPE Token Counter powered by `tiktoken-rs`.
#[derive(Clone)]
pub struct TiktokenCounter {
    encoding: TokenizerEncoding,
    #[cfg(feature = "tiktoken")]
    bpe: std::sync::Arc<CoreBPE>,
}

impl Debug for TiktokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("encoding", &self.encoding)
            .finish()
    }
}

#[cfg(feature = "tiktoken")]
static CL100K_BPE: std::sync::OnceLock<std::sync::Arc<CoreBPE>> = std::sync::OnceLock::new();
#[cfg(feature = "tiktoken")]
static O200K_BPE: std::sync::OnceLock<std::sync::Arc<CoreBPE>> = std::sync::OnceLock::new();
#[cfg(feature = "tiktoken")]
static P50K_BPE: std::sync::OnceLock<std::sync::Arc<CoreBPE>> = std::sync::OnceLock::new();
#[cfg(feature = "tiktoken")]
static R50K_BPE: std::sync::OnceLock<std::sync::Arc<CoreBPE>> = std::sync::OnceLock::new();

impl TiktokenCounter {
    /// Creates a new `TiktokenCounter` for the specified encoding.
    pub fn new(encoding: TokenizerEncoding) -> Self {
        #[cfg(feature = "tiktoken")]
        {
            let bpe = match encoding {
                TokenizerEncoding::Cl100kBase => CL100K_BPE
                    .get_or_init(|| {
                        std::sync::Arc::new(cl100k_base().expect("failed to load cl100k_base"))
                    })
                    .clone(),
                TokenizerEncoding::O200kBase => O200K_BPE
                    .get_or_init(|| {
                        std::sync::Arc::new(o200k_base().expect("failed to load o200k_base"))
                    })
                    .clone(),
                TokenizerEncoding::P50kBase => P50K_BPE
                    .get_or_init(|| {
                        std::sync::Arc::new(p50k_base().expect("failed to load p50k_base"))
                    })
                    .clone(),
                TokenizerEncoding::R50kBase => R50K_BPE
                    .get_or_init(|| {
                        std::sync::Arc::new(r50k_base().expect("failed to load r50k_base"))
                    })
                    .clone(),
            };

            Self { encoding, bpe }
        }

        #[cfg(not(feature = "tiktoken"))]
        {
            Self { encoding }
        }
    }

    /// Creates a counter configured with the standard `cl100k_base` encoding.
    pub fn cl100k() -> Self {
        Self::new(TokenizerEncoding::Cl100kBase)
    }

    /// Creates a counter configured with the frontier `o200k_base` encoding.
    pub fn o200k() -> Self {
        Self::new(TokenizerEncoding::O200kBase)
    }
}

impl Default for TiktokenCounter {
    fn default() -> Self {
        Self::cl100k()
    }
}

impl TokenCounter for TiktokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        #[cfg(feature = "tiktoken")]
        {
            self.bpe.encode_with_special_tokens(text).len()
        }

        #[cfg(not(feature = "tiktoken"))]
        {
            HeuristicTokenCounter.count_tokens(text)
        }
    }
}

/// Global convenience function to count tokens in text using the default cl100k tokenizer.
pub fn count_tokens(text: &str) -> usize {
    TiktokenCounter::default().count_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_token_counter() {
        let counter = HeuristicTokenCounter;

        assert_eq!(counter.count_tokens(""), 0);
        assert_eq!(counter.count_tokens("hello"), 2); // 5 chars -> 2 tokens
        assert_eq!(counter.count_tokens("hello world"), 3); // 2 words, 11 chars -> 3 tokens

        let long_text = "The quick brown fox jumps over the lazy dog";
        let tokens = counter.count_tokens(long_text);
        assert!((9..=12).contains(&tokens));
    }

    #[test]
    fn test_tiktoken_counter_precision() {
        let counter = TiktokenCounter::cl100k();

        assert_eq!(counter.count_tokens(""), 0);

        let phrase = "Ana Silva é a líder do Projeto Titan.";
        let token_count = counter.count_tokens(phrase);
        assert!(token_count > 0 && token_count < 20);

        // Code snippet test
        let code = "fn main() { println!(\"Hello, Graphite!\"); }";
        let code_tokens = counter.count_tokens(code);
        assert!(code_tokens > 0 && code_tokens < 20);
    }

    #[test]
    fn test_o200k_encoding() {
        let counter = TiktokenCounter::o200k();
        let phrase = "Artificial intelligence and knowledge graphs.";
        assert!(counter.count_tokens(phrase) > 0);
    }
}
