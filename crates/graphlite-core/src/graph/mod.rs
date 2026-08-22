//! Graph topology representations, adjacency lists, and graph traversal engines.

pub mod adjacency;
pub mod csr;
pub mod traversal;

pub use adjacency::AdjacencyGraph;
pub use csr::CsrGraph;
pub use traversal::{
    bfs_adjacency, bfs_csr, TraversalConfig, TraversalDirection, TraversedNode,
};
