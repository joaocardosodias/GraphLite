use std::collections::HashSet;

use crate::graph::hybrid_score::ScoredEntity;
use crate::graph::subgraph::ConnectedSubgraph;
use crate::id::{EdgeId, NodeId, StringId};
use crate::interner::StringInterner;
use crate::prompt::token_counter::TokenCounter;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A pruned subgraph whose text representation strictly fits within a given LLM token budget.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PrunedSubgraph {
    /// Entities retained within the token budget, ordered by relevance.
    pub entities: Vec<ScoredEntity>,
    /// Connecting edges retained between the selected entities.
    pub edges: Vec<EdgeRecord>,
    /// Actual total token footprint calculated by the `TokenCounter`.
    pub total_tokens: usize,
    /// The maximum token budget limit specified for this extraction.
    pub budget: usize,
}

impl PrunedSubgraph {
    /// Returns the number of entities retained.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the number of edges retained.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if no entities could fit within the budget.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

/// Greedily prunes a `ConnectedSubgraph` to strictly fit within `max_tokens`.
///
/// Prioritizes the highest-scoring entities and their connecting edges first,
/// stopping immediately when adding another entity or relation would exceed `max_tokens`.
pub fn prune_subgraph_by_budget(
    subgraph: &ConnectedSubgraph,
    interner: &StringInterner,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> PrunedSubgraph {
    if subgraph.is_empty() || max_tokens == 0 {
        return PrunedSubgraph {
            entities: Vec::new(),
            edges: Vec::new(),
            total_tokens: 0,
            budget: max_tokens,
        };
    }

    let header = "# Retrieved Knowledge Context:\n";
    let header_tokens = counter.count_tokens(header);

    if header_tokens >= max_tokens {
        return PrunedSubgraph {
            entities: Vec::new(),
            edges: Vec::new(),
            total_tokens: 0,
            budget: max_tokens,
        };
    }

    let mut current_tokens = header_tokens;
    let mut selected_entities: Vec<ScoredEntity> = Vec::new();
    let mut selected_node_ids: HashSet<NodeId> = HashSet::new();
    let mut selected_edges: Vec<EdgeRecord> = Vec::new();
    let mut included_edge_ids: HashSet<EdgeId> = HashSet::new();

    // Index all edges in this subgraph by both endpoints
    let mut incident_edges: std::collections::HashMap<NodeId, Vec<EdgeRecord>> =
        std::collections::HashMap::new();
    for edge in &subgraph.edges {
        incident_edges.entry(edge.source).or_default().push(*edge);
        incident_edges.entry(edge.target).or_default().push(*edge);
    }

    for entity in &subgraph.entities {
        let node_id = entity.node_id;
        let (node_name, entity_type, description) = if let Some(rec) = entity.node_record {
            (
                interner.resolve(rec.name_id).unwrap_or("Entidade"),
                interner.resolve(rec.type_id).filter(|s| !s.is_empty()),
                interner
                    .resolve(rec.description_id)
                    .filter(|s| !s.is_empty()),
            )
        } else {
            (
                interner
                    .resolve(StringId::new(node_id.as_u32()))
                    .unwrap_or("Entidade"),
                None,
                None,
            )
        };

        // Format candidate entity string to measure exact token cost
        let mut entity_line = if let Some(ty) = entity_type {
            format!(
                "- [{}] (Tipo: {}, Relevância: {:.2})\n",
                node_name, ty, entity.final_score
            )
        } else {
            format!(
                "- [{}] (Relevância: {:.2})\n",
                node_name, entity.final_score
            )
        };

        if let Some(desc) = description {
            entity_line.push_str(&format!("  Descrição: {}\n", desc));
        }

        let entity_tokens = counter.count_tokens(&entity_line);

        if current_tokens + entity_tokens > max_tokens {
            if selected_entities.is_empty() && max_tokens > current_tokens + 15 {
                // If it's the top match and slightly exceeds budget, truncate description to fit
                let remaining_budget = max_tokens - current_tokens;
                let truncated_desc: String = description
                    .unwrap_or("")
                    .chars()
                    .take(remaining_budget.saturating_mul(3))
                    .collect();
                let mut truncated_line = if let Some(ty) = entity_type {
                    format!(
                        "- [{}] (Tipo: {}, Relevância: {:.2})\n",
                        node_name, ty, entity.final_score
                    )
                } else {
                    format!(
                        "- [{}] (Relevância: {:.2})\n",
                        node_name, entity.final_score
                    )
                };
                if !truncated_desc.is_empty() {
                    truncated_line
                        .push_str(&format!("  Descrição: {}...\n", truncated_desc.trim()));
                }
                current_tokens += counter.count_tokens(&truncated_line);
                selected_entities.push(entity.clone());
                selected_node_ids.insert(node_id);
            }
            // Cannot fit more entities, stop greedily
            break;
        }

        current_tokens += entity_tokens;
        selected_entities.push(entity.clone());
        selected_node_ids.insert(node_id);

        // Try to include connecting edges between this new node and already selected nodes
        if let Some(edges) = incident_edges.get(&node_id) {
            for edge in edges {
                if !included_edge_ids.contains(&edge.id)
                    && selected_node_ids.contains(&edge.source)
                    && selected_node_ids.contains(&edge.target)
                {
                    let rel_name = interner.resolve(edge.relation_id).unwrap_or("RELACAO");
                    let target_name = interner
                        .resolve(StringId::new(edge.target.as_u32()))
                        .unwrap_or("Alvo");

                    let edge_line = format!(
                        "  - {} -> [{}] (Confiança: {:.2})\n",
                        rel_name, target_name, edge.weight
                    );
                    let edge_tokens = counter.count_tokens(&edge_line);

                    if current_tokens + edge_tokens <= max_tokens {
                        current_tokens += edge_tokens;
                        selected_edges.push(*edge);
                        included_edge_ids.insert(edge.id);
                    }
                }
            }
        }
    }

    PrunedSubgraph {
        entities: selected_entities,
        edges: selected_edges,
        total_tokens: current_tokens,
        budget: max_tokens,
    }
}

/// Calculates lexical Jaccard similarity between two text descriptions to measure redundancy without allocations.
fn text_jaccard_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let mut words_a: smallvec::SmallVec<[&str; 32]> = a.split_whitespace().collect();
    let mut words_b: smallvec::SmallVec<[&str; 32]> = b.split_whitespace().collect();

    words_a.sort_unstable();
    words_a.dedup();
    words_b.sort_unstable();
    words_b.dedup();

    let mut intersection = 0usize;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < words_a.len() && j < words_b.len() {
        match words_a[i].cmp(words_b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }

    let union = words_a.len() + words_b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Prunes a `ConnectedSubgraph` using Maximal Marginal Relevance (MMR) for maximum diversity and zero redundancy.
pub fn prune_subgraph_by_budget_mmr(
    subgraph: &ConnectedSubgraph,
    interner: &StringInterner,
    max_tokens: usize,
    counter: &dyn TokenCounter,
    mmr_lambda: f32,
) -> PrunedSubgraph {
    if subgraph.is_empty() || max_tokens == 0 {
        return PrunedSubgraph {
            entities: Vec::new(),
            edges: Vec::new(),
            total_tokens: 0,
            budget: max_tokens,
        };
    }

    let lambda = mmr_lambda.clamp(0.0, 1.0);

    // Fast path: if lambda is 1.0 (pure relevance) or single entity, bypass MMR overhead
    if (lambda - 1.0).abs() < 1e-4 || subgraph.entities.len() <= 1 {
        return prune_subgraph_by_budget(subgraph, interner, max_tokens, counter);
    }

    // Pre-resolve all string descriptions once to avoid repeated interner lookups and string clones
    let descriptions: Vec<&str> = subgraph
        .entities
        .iter()
        .map(|e| {
            if let Some(rec) = e.node_record {
                interner.resolve(rec.description_id).unwrap_or("")
            } else {
                ""
            }
        })
        .collect();

    let n = subgraph.entities.len();
    let mut remaining_indices: smallvec::SmallVec<[usize; 64]> = (0..n).collect();
    let mut mmr_ordered: Vec<ScoredEntity> = Vec::with_capacity(n);
    let mut selected_indices: smallvec::SmallVec<[usize; 64]> = smallvec::SmallVec::with_capacity(n);

    while !remaining_indices.is_empty() {
        let mut best_rem_pos = 0;
        let mut best_mmr_score = f32::NEG_INFINITY;

        for (rem_pos, &cand_idx) in remaining_indices.iter().enumerate() {
            let relevance = subgraph.entities[cand_idx].final_score;
            let cand_desc = descriptions[cand_idx];

            let max_sim = if selected_indices.is_empty() {
                0.0
            } else {
                selected_indices
                    .iter()
                    .map(|&sel_idx| text_jaccard_similarity(cand_desc, descriptions[sel_idx]))
                    .fold(0.0f32, f32::max)
            };

            let mmr_score = (lambda * relevance) - ((1.0 - lambda) * max_sim);
            if mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_rem_pos = rem_pos;
            }
        }

        let chosen_idx = remaining_indices.swap_remove(best_rem_pos);
        selected_indices.push(chosen_idx);
        mmr_ordered.push(subgraph.entities[chosen_idx].clone());
    }

    let mmr_subgraph = ConnectedSubgraph {
        entities: mmr_ordered,
        edges: subgraph.edges.clone(),
        seed_ids: subgraph.seed_ids.clone(),
    };

    prune_subgraph_by_budget(&mmr_subgraph, interner, max_tokens, counter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::token_counter::HeuristicTokenCounter;

    #[test]
    fn test_prune_subgraph_within_tight_budget() {
        let mut interner = StringInterner::new();
        let s0 = interner.intern("Projeto Titan");
        let s1 = interner.intern("Ana Silva");
        let s2 = interner.intern("Rust");
        let rel_lidera = interner.intern("LIDERA");
        let rel_escrito = interner.intern("ESCRITO_EM");

        let e0 = ScoredEntity {
            node_id: NodeId::new(s0.as_u32()),
            final_score: 0.95,
            vector_score: 0.90,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
            node_record: None,
        };
        let e1 = ScoredEntity {
            node_id: NodeId::new(s1.as_u32()),
            final_score: 0.85,
            vector_score: 0.20,
            graph_score: 0.95,
            depth: 1,
            path_edge: None,
            node_record: None,
        };
        let e2 = ScoredEntity {
            node_id: NodeId::new(s2.as_u32()),
            final_score: 0.75,
            vector_score: 0.15,
            graph_score: 0.90,
            depth: 1,
            path_edge: None,
            node_record: None,
        };

        let edge0 =
            EdgeRecord::new(EdgeId::new(1), e0.node_id, e1.node_id, rel_lidera).with_weight(0.95);
        let edge1 =
            EdgeRecord::new(EdgeId::new(2), e0.node_id, e2.node_id, rel_escrito).with_weight(0.90);

        let subgraph = ConnectedSubgraph {
            entities: vec![e0, e1, e2],
            edges: vec![edge0, edge1],
            seed_ids: vec![NodeId::new(s0.as_u32())],
        };

        let counter = HeuristicTokenCounter;

        // Generous budget: fits all 3 entities
        let pruned_full = prune_subgraph_by_budget(&subgraph, &interner, 500, &counter);
        assert_eq!(pruned_full.entity_count(), 3);
        assert!(pruned_full.total_tokens <= 500);

        // Extremely tight budget: fits only header + top 1 entity
        let pruned_tight = prune_subgraph_by_budget(&subgraph, &interner, 20, &counter);
        assert_eq!(pruned_tight.entity_count(), 1);
        assert_eq!(pruned_tight.entities[0].node_id, NodeId::new(s0.as_u32()));
        assert!(pruned_tight.total_tokens <= 20);
    }

    #[test]
    fn test_mmr_diversity_pruning() {
        let mut interner = StringInterner::new();
        let s0 = interner.intern("Document A");
        let s1 = interner.intern("Document B (Duplicate of A)");
        let s2 = interner.intern("Document C (Diverse Concept)");

        let desc_a = interner.intern("A quick brown fox jumps over the lazy dog in the forest");
        let desc_b = interner.intern("A quick brown fox jumps over the lazy dog in the woods"); // 80% similar to A
        let desc_c = interner.intern("Quantum computing algorithms using superconducting qubits"); // 0% similar to A

        let rec0 = crate::record::NodeRecord::new(
            NodeId::new(s0.as_u32()),
            s0,
            StringId::INVALID,
            desc_a,
            0,
        );
        let rec1 = crate::record::NodeRecord::new(
            NodeId::new(s1.as_u32()),
            s1,
            StringId::INVALID,
            desc_b,
            0,
        );
        let rec2 = crate::record::NodeRecord::new(
            NodeId::new(s2.as_u32()),
            s2,
            StringId::INVALID,
            desc_c,
            0,
        );

        let e0 = ScoredEntity {
            node_id: NodeId::new(s0.as_u32()),
            final_score: 0.95,
            vector_score: 0.95,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
            node_record: Some(rec0),
        };
        let e1 = ScoredEntity {
            node_id: NodeId::new(s1.as_u32()),
            final_score: 0.92, // High score, but near-duplicate of e0
            vector_score: 0.92,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
            node_record: Some(rec1),
        };
        let e2 = ScoredEntity {
            node_id: NodeId::new(s2.as_u32()),
            final_score: 0.88, // Slightly lower score, but completely diverse
            vector_score: 0.88,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
            node_record: Some(rec2),
        };

        let subgraph = ConnectedSubgraph {
            entities: vec![e0, e1, e2],
            edges: vec![],
            seed_ids: vec![NodeId::new(s0.as_u32())],
        };

        let counter = HeuristicTokenCounter;

        // With MMR (lambda = 0.5), e2 (diverse) is prioritized over e1 (duplicate)
        let pruned_mmr = prune_subgraph_by_budget_mmr(&subgraph, &interner, 500, &counter, 0.5);
        assert_eq!(pruned_mmr.entity_count(), 3);
        assert_eq!(pruned_mmr.entities[0].node_id, NodeId::new(s0.as_u32()));
        assert_eq!(
            pruned_mmr.entities[1].node_id,
            NodeId::new(s2.as_u32()),
            "Diverse Document C should be chosen before duplicate Document B"
        );
    }
}
