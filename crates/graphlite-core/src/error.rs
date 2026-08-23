use std::io;
use thiserror::Error;

use crate::id::{EdgeId, NodeId, StringId};

/// Centralized error enum for all GraphLite operations.
#[derive(Debug, Error)]
pub enum GraphLiteError {
    /// Underlying I/O error when reading or writing from disk.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// File header magic bytes do not match expected signature.
    #[error("Invalid magic bytes: expected {expected:?}, found {found:?}")]
    InvalidMagicBytes { expected: [u8; 4], found: [u8; 4] },

    /// Unsupported or incompatible binary file format version.
    #[error("Unsupported file version: expected {expected}, found {found}")]
    UnsupportedVersion { expected: u16, found: u16 },

    /// Data integrity check failed (CRC32 mismatch).
    #[error(
        "Integrity check failed: expected checksum {expected:#010x}, calculated {calculated:#010x}"
    )]
    ChecksumMismatch { expected: u32, calculated: u32 },

    /// Embedding vector dimension mismatch.
    #[error("Vector dimension mismatch: expected {expected} dimensions, found {found}")]
    VectorDimensionMismatch { expected: usize, found: usize },

    /// Node with the specified ID was not found in the database.
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    /// Edge with the specified ID was not found in the database.
    #[error("Edge not found: {0}")]
    EdgeNotFound(EdgeId),

    /// String with the specified ID was not found in the String Table.
    #[error("String not found for identifier: {0}")]
    StringNotFound(StringId),

    /// Binary layout or data structure corruption.
    #[error("Corrupted database format: {0}")]
    CorruptedFormat(String),

    /// Maximum capacity exceeded for nodes, edges, or string table.
    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Token budget constraint violation.
    #[error("Invalid token budget: requested {requested}, maximum allowable {limit}")]
    InvalidTokenBudget { requested: usize, limit: usize },

    /// Concurrency lock was poisoned.
    #[error("Lock poisoned: another thread panicked while holding the resource")]
    LockPoisoned,

    /// General operational or serialization error.
    #[error("Internal database error: {0}")]
    Internal(String),
}

/// Convenience alias for `std::result::Result<T, GraphLiteError>`.
pub type Result<T> = std::result::Result<T, GraphLiteError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_formatting() {
        let err = GraphLiteError::VectorDimensionMismatch {
            expected: 384,
            found: 768,
        };
        assert_eq!(
            err.to_string(),
            "Vector dimension mismatch: expected 384 dimensions, found 768"
        );

        let node_err = GraphLiteError::NodeNotFound(NodeId::new(42));
        assert_eq!(node_err.to_string(), "Node not found: NodeId(42)");

        let checksum_err = GraphLiteError::ChecksumMismatch {
            expected: 0x12345678,
            calculated: 0x87654321,
        };
        assert!(checksum_err.to_string().contains("0x12345678"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file missing");
        let err: GraphLiteError = io_err.into();
        assert!(matches!(err, GraphLiteError::Io(_)));
    }
}
