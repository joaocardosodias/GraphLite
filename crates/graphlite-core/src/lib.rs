//! # GraphLite Core
//!
//! An embedded, single-file Graph and Vector database engine written in pure Rust.
//! Designed for local-first GraphRAG, AI memory, and low-latency knowledge graphs.

pub mod error;
pub mod id;
pub mod interner;
pub mod record;

pub use error::{GraphLiteError, Result};
pub use id::{EdgeId, NodeId, StringId};
pub use interner::StringInterner;
pub use record::{EdgeRecord, NodeRecord, FLAG_ACTIVE, FLAG_DIRECTED, NO_VECTOR_OFFSET};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
