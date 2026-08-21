//! # GraphLite Core
//!
//! An embedded, single-file Graph and Vector database engine written in pure Rust.
//! Designed for local-first GraphRAG, AI memory, and low-latency knowledge graphs.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
