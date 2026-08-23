use std::collections::HashSet;

use crate::graph::adjacency::AdjacencyGraph;
use crate::graph::csr::CsrGraph;
use crate::graph::hybrid_score::{compute_hybrid_scores, HybridScoreConfig, ScoredEntity};
use crate::graph::traversal::{bfs_adjacency, bfs_csr, TraversalConfig};
use crate::id::NodeId;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A cohesive, scored subgraph extracted for retrieval-augmented generation.
///
/// Contains all high-scoring entities and the active structural edges that connect them.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConnectedSubgraph {
    /// Ranked entities sorted in descending order of `final_score`.
    pub entities: Vec<ScoredEntity>,
    /// Interconnecting edges present between the extracted entities.
    pub edges: Vec<EdgeRecord>,
    /// The initial vector search seed nodes that originated the traversal.
    pub seed_ids: Vec<NodeId>,
}

impl ConnectedSubgraph {
    /// Returns the number of entities in this subgraph.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the number of edges in this subgraph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the subgraph contains no entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns `true` if the given `NodeId` is present in this subgraph.
    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.entities.iter().any(|e| e.node_id == node_id)
    }

    /// Returns a reference to the `ScoredEntity` for a given `NodeId` if present.
    pub fn get_entity(&self, node_id: NodeId) -> Option<&ScoredEntity> {
        self.entities.iter().find(|e| e.node_id == node_id)
    }

    /// Returns all edges originating from the given `node_id` within this subgraph.
    pub fn outgoing_edges_for(&self, node_id: NodeId) -> Vec<&EdgeRecord> {
        self.edges.iter().filter(|e| e.source == node_id).collect()
    }

    /// Returns all edges arriving at the given `node_id` within this subgraph.
    pub fn incoming_edges_for(&self, node_id: NodeId) -> Vec<&EdgeRecord> {
        self.edges.iter().filter(|e| e.target == node_id).collect()
    }

    /// Returns a list of all unique `NodeId`s present in this subgraph.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.entities.iter().map(|e| e.node_id).collect()
    }
}

/// Extracts a `ConnectedSubgraph` from an immutable `CsrGraph` given vector search seeds.
pub fn extract_subgraph_csr(
    csr: &CsrGraph,
    seeds: &[(NodeId, f32)],
    traversal_config: &TraversalConfig,
    hybrid_config: &HybridScoreConfig,
) -> ConnectedSubgraph {
    if seeds.is_empty() {
        return ConnectedSubgraph::default();
    }

    let seed_ids: Vec<NodeId> = seeds.iter().map(|(id, _)| *id).collect();

    // 1. Multi-hop BFS traversal from seeds
    let traversed = bfs_csr(csr, &seed_ids, traversal_config);

    // 2. Hybrid scoring and ranking
    let scored_entities = compute_hybrid_scores(seeds, &traversed, hybrid_config);

    if scored_entities.is_empty() {
        return ConnectedSubgraph {
            entities: Vec::new(),
            edges: Vec::new(),
            seed_ids,
        };
    }

    let selected_node_set: HashSet<NodeId> = scored_entities.iter().map(|e| e.node_id).collect();

    // 3. Extract all interconnecting edges between the selected entities
    let mut interconnecting_edges: Vec<EdgeRecord> = Vec::new();
    let mut seen_edge_ids: HashSet<crate::id::EdgeId> = HashSet::new();

    for entity in &scored_entities {
        let node_id = entity.node_id;
        for edge in csr.out_edges(node_id) {
            if edge.is_active()
                && selected_node_set.contains(&edge.target)
                && seen_edge_ids.insert(edge.id)
            {
                interconnecting_edges.push(*edge);
            }
        }
    }

    ConnectedSubgraph {
        entities: scored_entities,
        edges: interconnecting_edges,
        seed_ids,
    }
}

/// Extracts a `ConnectedSubgraph` from a dynamic `AdjacencyGraph` given vector search seeds.
pub fn extract_subgraph_adjacency(
    adj: &AdjacencyGraph,
    seeds: &[(NodeId, f32)],
    traversal_config: &TraversalConfig,
    hybrid_config: &HybridScoreConfig,
) -> ConnectedSubgraph {
    if seeds.is_empty() {
        return ConnectedSubgraph::default();
    }

    let seed_ids: Vec<NodeId> = seeds.iter().map(|(id, _)| *id).collect();

    // 1. Multi-hop BFS traversal from seeds
    let traversed = bfs_adjacency(adj, &seed_ids, traversal_config);

    // 2. Hybrid scoring and ranking
    let mut scored_entities = compute_hybrid_scores(seeds, &traversed, hybrid_config);
    for entity in &mut scored_entities {
        if let Some(record) = adj.get_node(entity.node_id) {
            entity.node_record = Some(*record);
        }
    }

    if scored_entities.is_empty() {
        return ConnectedSubgraph {
            entities: Vec::new(),
            edges: Vec::new(),
            seed_ids,
        };
    }

    let selected_node_set: HashSet<NodeId> = scored_entities.iter().map(|e| e.node_id).collect();

    // 3. Extract all interconnecting edges between the selected entities
    let mut interconnecting_edges: Vec<EdgeRecord> = Vec::new();
    let mut seen_edge_ids: HashSet<crate::id::EdgeId> = HashSet::new();

    for entity in &scored_entities {
        let node_id = entity.node_id;
        for &edge_id in adj.out_edges(node_id) {
            if let Some(edge) = adj.get_edge(edge_id) {
                if edge.is_active()
                    && selected_node_set.contains(&edge.target)
                    && seen_edge_ids.insert(edge.id)
                {
                    interconnecting_edges.push(*edge);
                }
            }
        }
    }

    ConnectedSubgraph {
        entities: scored_entities,
        edges: interconnecting_edges,
        seed_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, StringId};
    use crate::record::NodeRecord;

    fn build_test_graph() -> AdjacencyGraph {
        let mut g = AdjacencyGraph::new();

        for i in 0..4 {
            g.add_node(NodeRecord::new(
                NodeId::new(i),
                StringId::new(i),
                StringId::new(1),
                StringId::INVALID,
                0,
            ))
            .unwrap();
        }

        // 0 -> 1 (0.95), 1 -> 2 (0.90), 0 -> 2 (cross-edge: 0.85), 3 (isolated node)
        g.add_edge(
            EdgeRecord::new(
                EdgeId::new(10),
                NodeId::new(0),
                NodeId::new(1),
                StringId::new(1),
            )
            .with_weight(0.95),
        )
        .unwrap();
        g.add_edge(
            EdgeRecord::new(
                EdgeId::new(20),
                NodeId::new(1),
                NodeId::new(2),
                StringId::new(2),
            )
            .with_weight(0.90),
        )
        .unwrap();
        g.add_edge(
            EdgeRecord::new(
                EdgeId::new(30),
                NodeId::new(0),
                NodeId::new(2),
                StringId::new(3),
            )
            .with_weight(0.85),
        )
        .unwrap();

        g
    }

    #[test]
    fn test_extract_connected_subgraph() {
        let g = build_test_graph();
        let csr: CsrGraph = (&g).into();

        let seeds = vec![(NodeId::new(0), 0.95)];
        let t_config = TraversalConfig {
            max_depth: 2,
            min_edge_weight: 0.5,
            ..Default::default()
        };
        let h_config = HybridScoreConfig::default();

        let subgraph = extract_subgraph_csr(&csr, &seeds, &t_config, &h_config);

        assert_eq!(subgraph.entity_count(), 3); // Nodes 0, 1, 2
        assert_eq!(subgraph.edge_count(), 3); // Edges 10, 20, 30

        assert!(subgraph.contains_node(NodeId::new(0)));
        assert!(subgraph.contains_node(NodeId::new(1)));
        assert!(subgraph.contains_node(NodeId::new(2)));
        assert!(!subgraph.contains_node(NodeId::new(3))); // Isolated node 3 not included

        // Node 0 should have 2 outgoing edges (0->1 and 0->2)
        let out_0 = subgraph.outgoing_edges_for(NodeId::new(0));
        assert_eq!(out_0.len(), 2);
    }
}
