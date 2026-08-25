use crate::id::NodeId;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An immutable, high-performance Compressed Sparse Row (CSR) Graph topology.
///
/// Compresses the entire graph edge topology into two cache-contiguous arrays:
/// 1. `offsets`: Array of size $(N + 1)$ mapping each `NodeId(i)` to its edge range `[offsets[i]..offsets[i+1]]`.
/// 2. `edges`: Contiguous flat array storing all `EdgeRecord` structures sequentially.
///
/// This delivers $O(1)$ neighbor lookup with maximum CPU L1/L2 cache locality,
/// zero pointer chasing, and direct memory-mapped (`mmap`) zero-copy compatibility.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsrGraph {
    /// Array of offsets into the `edges` array. Length is always `node_count + 1`.
    offsets: Vec<u64>,
    /// Contiguous array of all edges ordered by source node ID.
    edges: Vec<EdgeRecord>,
    /// Total number of nodes represented in this topology.
    node_count: usize,
}

impl CsrGraph {
    /// Creates a new `CsrGraph` from pre-validated raw offsets and edge arrays.
    pub fn new(offsets: Vec<u64>, edges: Vec<EdgeRecord>, node_count: usize) -> Self {
        Self {
            offsets,
            edges,
            node_count,
        }
    }

    /// Returns a slice of all outgoing `EdgeRecord`s originating from the given `node_id`.
    ///
    /// Executes in $O(1)$ time by returning a direct subslice of the contiguous edge buffer.
    #[inline]
    pub fn out_edges(&self, node_id: NodeId) -> &[EdgeRecord] {
        let idx = node_id.as_usize();
        if idx >= self.node_count || idx + 1 >= self.offsets.len() {
            return &[];
        }

        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;

        if start > end || end > self.edges.len() {
            return &[];
        }

        &self.edges[start..end]
    }

    /// Returns the number of outgoing edges (out-degree) for a given `node_id`.
    #[inline]
    pub fn out_degree(&self, node_id: NodeId) -> usize {
        let idx = node_id.as_usize();
        if idx >= self.node_count || idx + 1 >= self.offsets.len() {
            return 0;
        }

        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;

        end.saturating_sub(start)
    }

    /// Returns a list of active target `NodeId`s reachable in 1 hop from `node_id`.
    pub fn out_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.out_edges(node_id)
            .iter()
            .filter(|e| e.is_active())
            .map(|e| e.target)
            .collect()
    }

    /// Returns the total number of nodes in this topology.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the total number of edges stored in this topology.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the graph topology contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Returns the raw slice of offsets.
    #[inline]
    pub fn raw_offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Returns the raw slice of contiguous edge records.
    #[inline]
    pub fn raw_edges(&self) -> &[EdgeRecord] {
        &self.edges
    }

    /// Returns the approximate memory footprint in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        (self.offsets.len() * std::mem::size_of::<u64>())
            + (self.edges.len() * EdgeRecord::BINARY_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, StringId};

    #[test]
    fn test_csr_graph_lookup_and_degree() {
        // Construct a mini CSR graph with 3 nodes (0, 1, 2)
        // Node 0 has edges to 1 and 2
        // Node 1 has edge to 2
        // Node 2 has 0 edges
        let e0 = EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(0),
            NodeId::new(1),
            StringId::new(10),
        );
        let e1 = EdgeRecord::new(
            EdgeId::new(2),
            NodeId::new(0),
            NodeId::new(2),
            StringId::new(11),
        );
        let e2 = EdgeRecord::new(
            EdgeId::new(3),
            NodeId::new(1),
            NodeId::new(2),
            StringId::new(12),
        );

        let offsets = vec![0, 2, 3, 3]; // Node 0: [0..2], Node 1: [2..3], Node 2: [3..3]
        let edges = vec![e0, e1, e2];
        let csr = CsrGraph::new(offsets, edges, 3);

        assert_eq!(csr.node_count(), 3);
        assert_eq!(csr.edge_count(), 3);

        // Node 0 out-edges
        assert_eq!(csr.out_degree(NodeId::new(0)), 2);
        assert_eq!(csr.out_edges(NodeId::new(0)), &[e0, e1]);
        assert_eq!(
            csr.out_neighbors(NodeId::new(0)),
            vec![NodeId::new(1), NodeId::new(2)]
        );

        // Node 1 out-edges
        assert_eq!(csr.out_degree(NodeId::new(1)), 1);
        assert_eq!(csr.out_edges(NodeId::new(1)), &[e2]);

        // Node 2 out-edges (0 edges)
        assert_eq!(csr.out_degree(NodeId::new(2)), 0);
        assert_eq!(csr.out_edges(NodeId::new(2)), &[]);

        // Out of bounds node
        assert_eq!(csr.out_degree(NodeId::new(99)), 0);
        assert_eq!(csr.out_edges(NodeId::new(99)), &[]);
    }
}
