//! Graph topology representations, adjacency lists, and graph traversal engines.

pub mod adjacency;
pub mod csr;
pub mod hybrid_score;
pub mod subgraph;
pub mod traversal;

pub use adjacency::AdjacencyGraph;
pub use csr::CsrGraph;
pub use hybrid_score::{compute_hybrid_scores, HybridScoreConfig, ScoredEntity};
pub use subgraph::{
    extract_subgraph_adjacency, extract_subgraph_csr, ConnectedSubgraph,
};
pub use traversal::{
    bfs_adjacency, bfs_csr, TraversalConfig, TraversalDirection, TraversedNode,
};
