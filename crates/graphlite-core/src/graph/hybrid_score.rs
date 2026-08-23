use std::collections::HashMap;

use crate::graph::traversal::TraversedNode;
use crate::id::NodeId;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration parameters for the hybrid vector + graph scoring formula.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HybridScoreConfig {
    /// Weight parameter $\alpha \in [0.0, 1.0]$ balancing vector similarity vs graph topology.
    /// - $\alpha = 1.0$: 100% Vector similarity only.
    /// - $\alpha = 0.0$: 100% Graph topological strength only.
    /// - $\alpha = 0.6$: Balanced hybrid (default).
    pub alpha: f32,
    /// Decay factor $\gamma \in (0.0, 1.0]$ applied per hop distance from seed nodes.
    ///
    /// For depth $d$, the structural weight is decayed by $\gamma^d$ (e.g. $0.85^1 = 0.85$, $0.85^2 = 0.7225$).
    pub depth_decay: f32,
    /// Minimum combined score threshold. Entities scoring below this are filtered out.
    pub min_score_threshold: f32,
}

impl Default for HybridScoreConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            depth_decay: 0.85,
            min_score_threshold: 0.05,
        }
    }
}

/// A node scored and ranked by the hybrid retrieval engine.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScoredEntity {
    /// Identifier of the entity node.
    pub node_id: NodeId,
    /// Combined final score: $\alpha \cdot \text{vector\_score} + (1 - \alpha) \cdot \text{graph\_score}$.
    pub final_score: f32,
    /// Normalized vector similarity score $[0.0, 1.0]$.
    pub vector_score: f32,
    /// Structural graph score adjusted by edge weights and depth decay.
    pub graph_score: f32,
    /// Hop distance from the nearest seed node (0 for direct vector seeds).
    pub depth: usize,
    /// The incoming edge traversed to discover this node (if any).
    pub path_edge: Option<EdgeRecord>,
    /// Underlying NodeRecord with name, type, and description IDs.
    pub node_record: Option<crate::record::NodeRecord>,
}

/// Combines vector search seed scores with graph traversal paths using the hybrid scoring formula.
///
/// Returns a list of `ScoredEntity` sorted in descending order of `final_score`.
pub fn compute_hybrid_scores(
    seeds: &[(NodeId, f32)],
    traversed: &[TraversedNode],
    config: &HybridScoreConfig,
) -> Vec<ScoredEntity> {
    let alpha = config.alpha.clamp(0.0, 1.0);
    let decay = config.depth_decay.clamp(0.0, 1.0);

    // Map seed NodeIds to their direct vector similarity score
    let seed_vector_scores: HashMap<NodeId, f32> = seeds.iter().copied().collect();

    // Map to track the best score per unique node (in case reached by multiple paths)
    let mut entity_map: HashMap<NodeId, ScoredEntity> = HashMap::with_capacity(traversed.len());

    for node in traversed {
        let node_id = node.node_id;
        let vector_score = seed_vector_scores.get(&node_id).copied().unwrap_or(0.0);

        // Compute decayed graph score
        let hop_decay = decay.powi(node.depth as i32);
        let graph_score = (node.path_weight * hop_decay).clamp(0.0, 1.0);

        // Hybrid linear interpolation formula
        let final_score = (alpha * vector_score) + ((1.0 - alpha) * graph_score);

        if final_score < config.min_score_threshold {
            continue;
        }

        let scored = ScoredEntity {
            node_id,
            final_score,
            vector_score,
            graph_score,
            depth: node.depth,
            path_edge: node.incoming_edge,
            node_record: None,
        };

        // If node was already visited, keep the one with higher final score
        match entity_map.get_mut(&node_id) {
            Some(existing) => {
                if scored.final_score > existing.final_score {
                    *existing = scored;
                }
            }
            None => {
                entity_map.insert(node_id, scored);
            }
        }
    }

    let mut ranked_entities: Vec<ScoredEntity> = entity_map.into_values().collect();

    // Sort descending by final score
    ranked_entities.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ranked_entities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, StringId};

    #[test]
    fn test_hybrid_scoring_calculation() {
        let seeds = vec![
            (NodeId::new(1), 0.90), // High vector similarity
            (NodeId::new(2), 0.80),
        ];

        let e1 = EdgeRecord::new(
            EdgeId::new(10),
            NodeId::new(1),
            NodeId::new(3),
            StringId::new(1),
        )
        .with_weight(0.95);

        let traversed = vec![
            TraversedNode {
                node_id: NodeId::new(1),
                depth: 0,
                incoming_edge: None,
                path_weight: 1.0,
            },
            TraversedNode {
                node_id: NodeId::new(3),
                depth: 1,
                incoming_edge: Some(e1),
                path_weight: 0.95,
            },
        ];

        let config = HybridScoreConfig {
            alpha: 0.6,
            depth_decay: 0.8,
            min_score_threshold: 0.0,
        };

        let scored = compute_hybrid_scores(&seeds, &traversed, &config);

        assert_eq!(scored.len(), 2);

        // Node 1: alpha * 0.90 + (1 - alpha) * 1.0 = 0.6*0.9 + 0.4*1.0 = 0.54 + 0.40 = 0.94
        assert_eq!(scored[0].node_id, NodeId::new(1));
        assert!((scored[0].final_score - 0.94).abs() < 1e-4);

        // Node 3: alpha * 0.0 + (1 - alpha) * (0.95 * 0.8) = 0.4 * 0.76 = 0.304
        assert_eq!(scored[1].node_id, NodeId::new(3));
        assert!((scored[1].final_score - 0.304).abs() < 1e-4);
    }

    #[test]
    fn test_min_score_threshold_filtering() {
        let seeds = vec![(NodeId::new(1), 0.1)];
        let traversed = vec![TraversedNode {
            node_id: NodeId::new(1),
            depth: 0,
            incoming_edge: None,
            path_weight: 0.1,
        }];

        let config = HybridScoreConfig {
            alpha: 0.5,
            depth_decay: 0.8,
            min_score_threshold: 0.5, // Threshold above 0.1
        };

        let scored = compute_hybrid_scores(&seeds, &traversed, &config);
        assert_eq!(scored.len(), 0);
    }
}
