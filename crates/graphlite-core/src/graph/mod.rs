//! Graph topology representations, adjacency lists, and graph traversal engines.

pub mod adjacency;
pub mod csr;

pub use adjacency::AdjacencyGraph;
pub use csr::CsrGraph;
