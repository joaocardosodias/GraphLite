use std::collections::HashMap;

use crate::error::{GraphLiteError, Result};
use crate::graph::csr::CsrGraph;
use crate::id::{EdgeId, NodeId};
use crate::record::{EdgeRecord, NodeRecord};

/// A dynamic in-memory graph representation using bidirectional adjacency lists.
///
/// Optimized for fast insertions, deletions, and $O(1)$ lookup of incoming and outgoing neighbors.
#[derive(Debug, Clone, Default)]
pub struct AdjacencyGraph {
    /// Mapping of `NodeId` to its full `NodeRecord`.
    nodes: HashMap<NodeId, NodeRecord>,
    /// Mapping of `EdgeId` to its full `EdgeRecord`.
    edges: HashMap<EdgeId, EdgeRecord>,
    /// Outgoing adjacency list: `source_node -> [edge_id_1, edge_id_2, ...]`.
    out_edges: HashMap<NodeId, Vec<EdgeId>>,
    /// Incoming adjacency list: `target_node -> [edge_id_1, edge_id_2, ...]`.
    in_edges: HashMap<NodeId, Vec<EdgeId>>,
}

impl AdjacencyGraph {
    /// Creates a new, empty `AdjacencyGraph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new `AdjacencyGraph` with pre-allocated capacity for nodes and edges.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            nodes: HashMap::with_capacity(node_capacity),
            edges: HashMap::with_capacity(edge_capacity),
            out_edges: HashMap::with_capacity(node_capacity),
            in_edges: HashMap::with_capacity(node_capacity),
        }
    }

    /// Adds a `NodeRecord` to the graph.
    ///
    /// If a node with the same `NodeId` already exists, returns an error.
    pub fn add_node(&mut self, record: NodeRecord) -> Result<NodeId> {
        let id = record.id;
        if self.nodes.contains_key(&id) {
            return Err(GraphLiteError::Internal(format!(
                "Node with ID {} already exists in graph",
                id
            )));
        }

        self.nodes.insert(id, record);
        self.out_edges.entry(id).or_default();
        self.in_edges.entry(id).or_default();

        Ok(id)
    }

    /// Upserts a `NodeRecord` into the graph (replaces if exists, inserts otherwise).
    pub fn upsert_node(&mut self, record: NodeRecord) -> NodeId {
        let id = record.id;
        self.nodes.insert(id, record);
        self.out_edges.entry(id).or_default();
        self.in_edges.entry(id).or_default();
        id
    }

    /// Returns a reference to the `NodeRecord` with the given `NodeId`.
    #[inline]
    pub fn get_node(&self, id: NodeId) -> Option<&NodeRecord> {
        self.nodes.get(&id)
    }

    /// Returns a mutable reference to the `NodeRecord` with the given `NodeId`.
    #[inline]
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut NodeRecord> {
        self.nodes.get_mut(&id)
    }

    /// Returns `true` if the graph contains a node with the given `NodeId`.
    #[inline]
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Adds an `EdgeRecord` connecting two nodes in the graph.
    ///
    /// Validates that both `source` and `target` nodes exist before adding.
    pub fn add_edge(&mut self, record: EdgeRecord) -> Result<EdgeId> {
        let edge_id = record.id;
        let source = record.source;
        let target = record.target;

        if !self.contains_node(source) {
            return Err(GraphLiteError::NodeNotFound(source));
        }
        if !self.contains_node(target) {
            return Err(GraphLiteError::NodeNotFound(target));
        }
        if self.edges.contains_key(&edge_id) {
            return Err(GraphLiteError::Internal(format!(
                "Edge with ID {} already exists in graph",
                edge_id
            )));
        }

        self.out_edges.entry(source).or_default().push(edge_id);
        self.in_edges.entry(target).or_default().push(edge_id);
        self.edges.insert(edge_id, record);

        Ok(edge_id)
    }

    /// Returns a reference to the `EdgeRecord` with the given `EdgeId`.
    #[inline]
    pub fn get_edge(&self, id: EdgeId) -> Option<&EdgeRecord> {
        self.edges.get(&id)
    }

    /// Returns a mutable reference to the `EdgeRecord` with the given `EdgeId`.
    #[inline]
    pub fn get_edge_mut(&mut self, id: EdgeId) -> Option<&mut EdgeRecord> {
        self.edges.get_mut(&id)
    }

    /// Returns `true` if the graph contains an edge with the given `EdgeId`.
    #[inline]
    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.edges.contains_key(&id)
    }

    /// Returns a slice of outgoing `EdgeId`s originating from the given `node_id`.
    pub fn out_edges(&self, node_id: NodeId) -> &[EdgeId] {
        self.out_edges
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns a slice of incoming `EdgeId`s arriving at the given `node_id`.
    pub fn in_edges(&self, node_id: NodeId) -> &[EdgeId] {
        self.in_edges
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns a list of outgoing neighbor pairs `(target_node_id, edge_id)` for active edges.
    pub fn out_neighbors(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let mut neighbors = Vec::new();
        for &edge_id in self.out_edges(node_id) {
            if let Some(edge) = self.get_edge(edge_id) {
                if edge.is_active() {
                    neighbors.push((edge.target, edge_id));
                }
            }
        }
        neighbors
    }

    /// Returns a list of incoming neighbor pairs `(source_node_id, edge_id)` for active edges.
    pub fn in_neighbors(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let mut neighbors = Vec::new();
        for &edge_id in self.in_edges(node_id) {
            if let Some(edge) = self.get_edge(edge_id) {
                if edge.is_active() {
                    neighbors.push((edge.source, edge_id));
                }
            }
        }
        neighbors
    }

    /// Returns a combined list of all adjacent neighbors (both incoming and outgoing).
    pub fn all_neighbors(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let mut neighbors = self.out_neighbors(node_id);
        neighbors.extend(self.in_neighbors(node_id));
        neighbors
    }

    /// Removes a node and all connected incoming and outgoing edges from the graph.
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<NodeRecord> {
        let node = self.nodes.remove(&node_id)?;

        // Remove and cleanup all outgoing edges
        if let Some(out_e) = self.out_edges.remove(&node_id) {
            for edge_id in out_e {
                if let Some(edge) = self.edges.remove(&edge_id) {
                    if let Some(in_list) = self.in_edges.get_mut(&edge.target) {
                        in_list.retain(|&e| e != edge_id);
                    }
                }
            }
        }

        // Remove and cleanup all incoming edges
        if let Some(in_e) = self.in_edges.remove(&node_id) {
            for edge_id in in_e {
                if let Some(edge) = self.edges.remove(&edge_id) {
                    if let Some(out_list) = self.out_edges.get_mut(&edge.source) {
                        out_list.retain(|&e| e != edge_id);
                    }
                }
            }
        }

        Some(node)
    }

    /// Removes a specific edge from the graph.
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Option<EdgeRecord> {
        let edge = self.edges.remove(&edge_id)?;

        if let Some(out_list) = self.out_edges.get_mut(&edge.source) {
            out_list.retain(|&e| e != edge_id);
        }
        if let Some(in_list) = self.in_edges.get_mut(&edge.target) {
            in_list.retain(|&e| e != edge_id);
        }

        Some(edge)
    }

    /// Compiles this dynamic `AdjacencyGraph` into an immutable, contiguous `CsrGraph`.
    ///
    /// The resulting `CsrGraph` can be serialized to disk zero-copy or traversed with zero pointer chasing.
    pub fn to_csr(&self) -> CsrGraph {
        if self.nodes.is_empty() {
            return CsrGraph::new(vec![0], Vec::new(), 0);
        }

        let max_id = self.nodes.keys().map(|n| n.as_usize()).max().unwrap_or(0);
        let node_count = max_id + 1;

        let mut offsets = Vec::with_capacity(node_count + 1);
        let mut contiguous_edges = Vec::with_capacity(self.edges.len());

        let mut current_offset: u64 = 0;
        offsets.push(current_offset);

        for idx in 0..node_count {
            let node_id = NodeId::new(idx as u32);
            if let Some(edge_ids) = self.out_edges.get(&node_id) {
                for &edge_id in edge_ids {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if edge.is_active() {
                            contiguous_edges.push(*edge);
                            current_offset += 1;
                        }
                    }
                }
            }
            offsets.push(current_offset);
        }

        CsrGraph::new(offsets, contiguous_edges, node_count)
    }

    /// Returns the total number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of edges in the graph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the graph contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns an iterator over all `NodeRecord` references in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = &NodeRecord> {
        self.nodes.values()
    }

    /// Returns an iterator over all `EdgeRecord` references in the graph.
    pub fn edges(&self) -> impl Iterator<Item = &EdgeRecord> {
        self.edges.values()
    }

    /// Clears all nodes, edges, and adjacency lists from the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.out_edges.clear();
        self.in_edges.clear();
    }
}

impl From<&AdjacencyGraph> for CsrGraph {
    fn from(graph: &AdjacencyGraph) -> Self {
        graph.to_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::StringId;
    use crate::record::NO_VECTOR_OFFSET;

    #[test]
    fn test_add_nodes_and_edges() {
        let mut graph = AdjacencyGraph::new();

        let n1 = NodeRecord::new(
            NodeId::new(1),
            StringId::new(10),
            StringId::new(20),
            StringId::INVALID,
            NO_VECTOR_OFFSET,
        );
        let n2 = NodeRecord::new(
            NodeId::new(2),
            StringId::new(11),
            StringId::new(21),
            StringId::INVALID,
            NO_VECTOR_OFFSET,
        );

        assert_eq!(graph.add_node(n1).unwrap(), NodeId::new(1));
        assert_eq!(graph.add_node(n2).unwrap(), NodeId::new(2));
        assert_eq!(graph.node_count(), 2);

        // Cannot add duplicate node ID
        assert!(graph.add_node(n1).is_err());

        // Add edge
        let edge = EdgeRecord::new(
            EdgeId::new(100),
            NodeId::new(1),
            NodeId::new(2),
            StringId::new(99),
        )
        .with_weight(0.95);

        assert_eq!(graph.add_edge(edge).unwrap(), EdgeId::new(100));
        assert_eq!(graph.edge_count(), 1);

        // Check neighbors
        let out_n = graph.out_neighbors(NodeId::new(1));
        assert_eq!(out_n, vec![(NodeId::new(2), EdgeId::new(100))]);

        let in_n = graph.in_neighbors(NodeId::new(2));
        assert_eq!(in_n, vec![(NodeId::new(1), EdgeId::new(100))]);
    }

    #[test]
    fn test_edge_to_missing_node_fails() {
        let mut graph = AdjacencyGraph::new();
        let n1 = NodeRecord::new(
            NodeId::new(1),
            StringId::new(1),
            StringId::new(2),
            StringId::INVALID,
            NO_VECTOR_OFFSET,
        );
        graph.add_node(n1).unwrap();

        let edge = EdgeRecord::new(
            EdgeId::new(50),
            NodeId::new(1),
            NodeId::new(999),
            StringId::new(5),
        );

        let err = graph.add_edge(edge).unwrap_err();
        match err {
            GraphLiteError::NodeNotFound(id) => assert_eq!(id, NodeId::new(999)),
            _ => panic!("Expected NodeNotFound error"),
        }
    }

    #[test]
    fn test_remove_node_cascades_edges() {
        let mut graph = AdjacencyGraph::new();

        let n1 = NodeRecord::new(NodeId::new(1), StringId::new(1), StringId::new(2), StringId::INVALID, 0);
        let n2 = NodeRecord::new(NodeId::new(2), StringId::new(3), StringId::new(4), StringId::INVALID, 0);
        let n3 = NodeRecord::new(NodeId::new(3), StringId::new(5), StringId::new(6), StringId::INVALID, 0);

        graph.add_node(n1).unwrap();
        graph.add_node(n2).unwrap();
        graph.add_node(n3).unwrap();

        // 1 -> 2 and 2 -> 3
        graph.add_edge(EdgeRecord::new(EdgeId::new(1), NodeId::new(1), NodeId::new(2), StringId::new(10))).unwrap();
        graph.add_edge(EdgeRecord::new(EdgeId::new(2), NodeId::new(2), NodeId::new(3), StringId::new(11))).unwrap();

        assert_eq!(graph.edge_count(), 2);

        // Remove node 2 -> should automatically remove both edge 1 and edge 2
        let removed = graph.remove_node(NodeId::new(2));
        assert!(removed.is_some());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);

        assert_eq!(graph.out_neighbors(NodeId::new(1)), vec![]);
        assert_eq!(graph.in_neighbors(NodeId::new(3)), vec![]);
    }

    #[test]
    fn test_adjacency_to_csr_compilation() {
        let mut graph = AdjacencyGraph::new();

        let n0 = NodeRecord::new(NodeId::new(0), StringId::new(1), StringId::new(2), StringId::INVALID, 0);
        let n1 = NodeRecord::new(NodeId::new(1), StringId::new(3), StringId::new(4), StringId::INVALID, 0);
        let n2 = NodeRecord::new(NodeId::new(2), StringId::new(5), StringId::new(6), StringId::INVALID, 0);

        graph.add_node(n0).unwrap();
        graph.add_node(n1).unwrap();
        graph.add_node(n2).unwrap();

        // Node 0 -> Node 1 (weight 0.9)
        // Node 0 -> Node 2 (weight 0.8)
        // Node 1 -> Node 2 (weight 0.7)
        let e1 = EdgeRecord::new(EdgeId::new(1), NodeId::new(0), NodeId::new(1), StringId::new(10)).with_weight(0.9);
        let e2 = EdgeRecord::new(EdgeId::new(2), NodeId::new(0), NodeId::new(2), StringId::new(11)).with_weight(0.8);
        let e3 = EdgeRecord::new(EdgeId::new(3), NodeId::new(1), NodeId::new(2), StringId::new(12)).with_weight(0.7);

        graph.add_edge(e1).unwrap();
        graph.add_edge(e2).unwrap();
        graph.add_edge(e3).unwrap();

        // Compile to CSR
        let csr: CsrGraph = (&graph).into();

        assert_eq!(csr.node_count(), 3);
        assert_eq!(csr.edge_count(), 3);

        // Check CSR topology parity
        assert_eq!(csr.out_degree(NodeId::new(0)), 2);
        assert_eq!(csr.out_edges(NodeId::new(0)), &[e1, e2]);
        assert_eq!(csr.out_degree(NodeId::new(1)), 1);
        assert_eq!(csr.out_edges(NodeId::new(1)), &[e3]);
        assert_eq!(csr.out_degree(NodeId::new(2)), 0);
        assert_eq!(csr.out_edges(NodeId::new(2)), &[]);
    }
}
