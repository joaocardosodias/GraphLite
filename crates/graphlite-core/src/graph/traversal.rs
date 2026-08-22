use std::collections::{HashSet, VecDeque};

use crate::graph::adjacency::AdjacencyGraph;
use crate::graph::csr::CsrGraph;
use crate::id::NodeId;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Direction of edge traversal during multi-hop exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TraversalDirection {
    /// Follow only outgoing edges ($A \to B$).
    #[default]
    Outgoing,
    /// Follow only incoming edges ($B \to A$).
    Incoming,
    /// Follow edges in both directions (undirected exploration).
    Undirected,
}

/// Configuration parameters for graph traversal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TraversalConfig {
    /// Maximum search depth in hops from the seed nodes (e.g. 1 or 2).
    pub max_depth: usize,
    /// Minimum edge weight threshold (edges with weight below this are ignored).
    pub min_edge_weight: f32,
    /// Maximum number of unique nodes to visit (circuit breaker to avoid explosive traversal).
    pub max_nodes: usize,
    /// Traversal direction (Outgoing, Incoming, or Undirected).
    pub direction: TraversalDirection,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            min_edge_weight: 0.0,
            max_nodes: 1000,
            direction: TraversalDirection::Outgoing,
        }
    }
}

/// Represents a node discovered during graph traversal.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TraversedNode {
    /// Unique identifier of the visited node.
    pub node_id: NodeId,
    /// Number of hops from the nearest seed node (0 for seed nodes).
    pub depth: usize,
    /// The edge record traversed to reach this node (`None` for seed nodes).
    pub incoming_edge: Option<EdgeRecord>,
    /// Accumulated path weight / decay score from the seed node.
    pub path_weight: f32,
}

/// Executes a multi-hop Breadth-First Search (BFS) on an immutable `CsrGraph`.
pub fn bfs_csr(
    csr: &CsrGraph,
    seeds: &[NodeId],
    config: &TraversalConfig,
) -> Vec<TraversedNode> {
    if seeds.is_empty() || config.max_nodes == 0 {
        return Vec::new();
    }

    let mut visited: HashSet<NodeId> = HashSet::with_capacity(seeds.len() * 4);
    let mut results: Vec<TraversedNode> = Vec::with_capacity(seeds.len() * 8);
    let mut queue: VecDeque<(NodeId, usize, Option<EdgeRecord>, f32)> =
        VecDeque::with_capacity(seeds.len() * 4);

    // Enqueue seed nodes at depth 0
    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back((seed, 0, None, 1.0));
            results.push(TraversedNode {
                node_id: seed,
                depth: 0,
                incoming_edge: None,
                path_weight: 1.0,
            });
            if results.len() >= config.max_nodes {
                return results;
            }
        }
    }

    while let Some((current_node, current_depth, _, current_weight)) = queue.pop_front() {
        if current_depth >= config.max_depth {
            continue;
        }

        let next_depth = current_depth + 1;
        let edges = csr.out_edges(current_node);

        for edge in edges {
            if !edge.is_active() || edge.weight < config.min_edge_weight {
                continue;
            }

            let neighbor = edge.target;
            if visited.insert(neighbor) {
                let next_weight = current_weight * edge.weight;
                let traversed = TraversedNode {
                    node_id: neighbor,
                    depth: next_depth,
                    incoming_edge: Some(*edge),
                    path_weight: next_weight,
                };

                results.push(traversed);
                if results.len() >= config.max_nodes {
                    return results;
                }

                if next_depth < config.max_depth {
                    queue.push_back((neighbor, next_depth, Some(*edge), next_weight));
                }
            }
        }
    }

    results
}

/// Executes a multi-hop Breadth-First Search (BFS) on a dynamic `AdjacencyGraph`.
pub fn bfs_adjacency(
    adj: &AdjacencyGraph,
    seeds: &[NodeId],
    config: &TraversalConfig,
) -> Vec<TraversedNode> {
    if seeds.is_empty() || config.max_nodes == 0 {
        return Vec::new();
    }

    let mut visited: HashSet<NodeId> = HashSet::with_capacity(seeds.len() * 4);
    let mut results: Vec<TraversedNode> = Vec::with_capacity(seeds.len() * 8);
    let mut queue: VecDeque<(NodeId, usize, Option<EdgeRecord>, f32)> =
        VecDeque::with_capacity(seeds.len() * 4);

    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back((seed, 0, None, 1.0));
            results.push(TraversedNode {
                node_id: seed,
                depth: 0,
                incoming_edge: None,
                path_weight: 1.0,
            });
            if results.len() >= config.max_nodes {
                return results;
            }
        }
    }

    while let Some((current_node, current_depth, _, current_weight)) = queue.pop_front() {
        if current_depth >= config.max_depth {
            continue;
        }

        let next_depth = current_depth + 1;

        let edge_ids = match config.direction {
            TraversalDirection::Outgoing => adj.out_edges(current_node),
            TraversalDirection::Incoming => adj.in_edges(current_node),
            TraversalDirection::Undirected => adj.out_edges(current_node),
        };

        // Helper to process an edge ID
        let mut process_edge = |edge_id: &crate::id::EdgeId, results: &mut Vec<TraversedNode>, queue: &mut VecDeque<(NodeId, usize, Option<EdgeRecord>, f32)>| -> bool {
            if let Some(edge) = adj.get_edge(*edge_id) {
                if !edge.is_active() || edge.weight < config.min_edge_weight {
                    return true; // continue
                }

                let neighbor = if edge.source == current_node {
                    edge.target
                } else {
                    edge.source
                };

                if visited.insert(neighbor) {
                    let next_weight = current_weight * edge.weight;
                    let traversed = TraversedNode {
                        node_id: neighbor,
                        depth: next_depth,
                        incoming_edge: Some(*edge),
                        path_weight: next_weight,
                    };

                    results.push(traversed);
                    if results.len() >= config.max_nodes {
                        return false; // stop
                    }

                    if next_depth < config.max_depth {
                        queue.push_back((neighbor, next_depth, Some(*edge), next_weight));
                    }
                }
            }
            true
        };

        for edge_id in edge_ids {
            if !process_edge(edge_id, &mut results, &mut queue) {
                return results;
            }
        }

        if config.direction == TraversalDirection::Undirected {
            for edge_id in adj.in_edges(current_node) {
                if !process_edge(edge_id, &mut results, &mut queue) {
                    return results;
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, StringId};
    use crate::record::NodeRecord;

    fn setup_sample_graph() -> AdjacencyGraph {
        let mut graph = AdjacencyGraph::new();

        // Nodes 0, 1, 2, 3, 4
        for i in 0..5 {
            graph
                .add_node(NodeRecord::new(
                    NodeId::new(i),
                    StringId::new(i * 10),
                    StringId::new(1),
                    StringId::INVALID,
                    0,
                ))
                .unwrap();
        }

        // 0 -> 1 (0.9), 1 -> 2 (0.8), 2 -> 3 (0.3 - weak), 0 -> 4 (0.95)
        graph
            .add_edge(EdgeRecord::new(EdgeId::new(1), NodeId::new(0), NodeId::new(1), StringId::new(100)).with_weight(0.9))
            .unwrap();
        graph
            .add_edge(EdgeRecord::new(EdgeId::new(2), NodeId::new(1), NodeId::new(2), StringId::new(101)).with_weight(0.8))
            .unwrap();
        graph
            .add_edge(EdgeRecord::new(EdgeId::new(3), NodeId::new(2), NodeId::new(3), StringId::new(102)).with_weight(0.3))
            .unwrap();
        graph
            .add_edge(EdgeRecord::new(EdgeId::new(4), NodeId::new(0), NodeId::new(4), StringId::new(103)).with_weight(0.95))
            .unwrap();

        graph
    }

    #[test]
    fn test_bfs_csr_multi_hop_traversal() {
        let adj = setup_sample_graph();
        let csr: CsrGraph = (&adj).into();

        // 2-hop traversal from Node 0
        let config = TraversalConfig {
            max_depth: 2,
            min_edge_weight: 0.5, // Ignores edge 2 -> 3 (weight 0.3)
            max_nodes: 50,
            direction: TraversalDirection::Outgoing,
        };

        let visited = bfs_csr(&csr, &[NodeId::new(0)], &config);

        // Discovered nodes should be:
        // Depth 0: Node 0
        // Depth 1: Node 1 (0.9), Node 4 (0.95)
        // Depth 2: Node 2 (0.9 * 0.8 = 0.72)
        // Node 3 is NOT reached because edge 2->3 (0.3) < min_weight (0.5)
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0].node_id, NodeId::new(0));
        assert_eq!(visited[0].depth, 0);

        let visited_ids: Vec<NodeId> = visited.iter().map(|n| n.node_id).collect();
        assert!(visited_ids.contains(&NodeId::new(1)));
        assert!(visited_ids.contains(&NodeId::new(4)));
        assert!(visited_ids.contains(&NodeId::new(2)));
        assert!(!visited_ids.contains(&NodeId::new(3)));
    }

    #[test]
    fn test_bfs_cyclic_graph_termination() {
        let mut graph = AdjacencyGraph::new();
        // Triangle cycle: 0 -> 1 -> 2 -> 0
        for i in 0..3 {
            graph
                .add_node(NodeRecord::new(NodeId::new(i), StringId::new(i), StringId::new(1), StringId::INVALID, 0))
                .unwrap();
        }
        graph.add_edge(EdgeRecord::new(EdgeId::new(1), NodeId::new(0), NodeId::new(1), StringId::new(10))).unwrap();
        graph.add_edge(EdgeRecord::new(EdgeId::new(2), NodeId::new(1), NodeId::new(2), StringId::new(11))).unwrap();
        graph.add_edge(EdgeRecord::new(EdgeId::new(3), NodeId::new(2), NodeId::new(0), StringId::new(12))).unwrap();

        let csr: CsrGraph = (&graph).into();
        let config = TraversalConfig {
            max_depth: 10,
            ..Default::default()
        };

        let visited = bfs_csr(&csr, &[NodeId::new(0)], &config);
        // Even with max_depth: 10, cycle detection visits exactly 3 unique nodes
        assert_eq!(visited.len(), 3);
    }
}
