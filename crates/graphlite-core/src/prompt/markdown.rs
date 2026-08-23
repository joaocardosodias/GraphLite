use std::collections::{HashMap, HashSet};

use crate::graph::subgraph::ConnectedSubgraph;
use crate::id::{NodeId, StringId};
use crate::interner::StringInterner;
use crate::prompt::pruner::PrunedSubgraph;
use crate::record::EdgeRecord;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Visual styling format for generated Markdown context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MarkdownStyle {
    /// Hierarchical nested list with relations indented under their source entity (default).
    #[default]
    Hierarchical,
    /// Separate sections for Entities and Relationships.
    SeparatedSections,
}

/// Configuration options for the Markdown prompt context formatter.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MarkdownFormatConfig {
    /// Custom title for the context header.
    pub header_title: String,
    /// Whether to include numeric hybrid relevance scores next to entity names.
    pub include_scores: bool,
    /// Whether to include numeric edge weights next to relation names.
    pub include_edge_weights: bool,
    /// Output Markdown layout style.
    pub style: MarkdownStyle,
}

impl Default for MarkdownFormatConfig {
    fn default() -> Self {
        Self {
            header_title: "Contexto Recuperado do Conhecimento".to_string(),
            include_scores: true,
            include_edge_weights: true,
            style: MarkdownStyle::Hierarchical,
        }
    }
}

/// Formats a `PrunedSubgraph` into an LLM-optimized Markdown string.
pub fn format_pruned_subgraph_markdown(
    subgraph: &PrunedSubgraph,
    interner: &StringInterner,
    config: &MarkdownFormatConfig,
) -> String {
    if subgraph.is_empty() {
        return String::new();
    }

    match config.style {
        MarkdownStyle::Hierarchical => {
            format_hierarchical(&subgraph.entities, &subgraph.edges, interner, config)
        }
        MarkdownStyle::SeparatedSections => {
            format_separated(&subgraph.entities, &subgraph.edges, interner, config)
        }
    }
}

/// Formats an unpruned `ConnectedSubgraph` directly into a Markdown string.
pub fn format_connected_subgraph_markdown(
    subgraph: &ConnectedSubgraph,
    interner: &StringInterner,
    config: &MarkdownFormatConfig,
) -> String {
    if subgraph.is_empty() {
        return String::new();
    }

    match config.style {
        MarkdownStyle::Hierarchical => {
            format_hierarchical(&subgraph.entities, &subgraph.edges, interner, config)
        }
        MarkdownStyle::SeparatedSections => {
            format_separated(&subgraph.entities, &subgraph.edges, interner, config)
        }
    }
}

fn format_hierarchical(
    entities: &[crate::graph::hybrid_score::ScoredEntity],
    edges: &[EdgeRecord],
    interner: &StringInterner,
    config: &MarkdownFormatConfig,
) -> String {
    let mut output = format!("# {}:\n", config.header_title);

    // Map outgoing edges by source NodeId
    let mut outgoing_map: HashMap<NodeId, Vec<&EdgeRecord>> = HashMap::new();
    for edge in edges {
        outgoing_map.entry(edge.source).or_default().push(edge);
    }

    let node_set: HashSet<NodeId> = entities.iter().map(|e| e.node_id).collect();

    for entity in entities {
        let node_id = entity.node_id;
        let node_name = interner
            .resolve(StringId::new(node_id.as_u32()))
            .unwrap_or("Entidade");

        if config.include_scores {
            output.push_str(&format!(
                "- [{}] (Relevância: {:.2})\n",
                node_name, entity.final_score
            ));
        } else {
            output.push_str(&format!("- [{}]\n", node_name));
        }

        // Render outgoing relations to other selected entities
        if let Some(out_edges) = outgoing_map.get(&node_id) {
            for edge in out_edges {
                if node_set.contains(&edge.target) {
                    let rel_name = interner.resolve(edge.relation_id).unwrap_or("RELACAO");
                    let target_name = interner
                        .resolve(StringId::new(edge.target.as_u32()))
                        .unwrap_or("Alvo");

                    if config.include_edge_weights {
                        output.push_str(&format!(
                            "  - {} -> [{}] (Confiança: {:.2})\n",
                            rel_name, target_name, edge.weight
                        ));
                    } else {
                        output.push_str(&format!("  - {} -> [{}]\n", rel_name, target_name));
                    }
                }
            }
        }
    }

    output
}

fn format_separated(
    entities: &[crate::graph::hybrid_score::ScoredEntity],
    edges: &[EdgeRecord],
    interner: &StringInterner,
    config: &MarkdownFormatConfig,
) -> String {
    let mut output = format!("# {}:\n\n## Entidades:\n", config.header_title);

    for entity in entities {
        let node_name = interner
            .resolve(StringId::new(entity.node_id.as_u32()))
            .unwrap_or("Entidade");

        if config.include_scores {
            output.push_str(&format!(
                "- [{}] (Score: {:.2})\n",
                node_name, entity.final_score
            ));
        } else {
            output.push_str(&format!("- [{}]\n", node_name));
        }
    }

    if !edges.is_empty() {
        output.push_str("\n## Relações Estruturais:\n");
        for edge in edges {
            let source_name = interner
                .resolve(StringId::new(edge.source.as_u32()))
                .unwrap_or("Origem");
            let rel_name = interner.resolve(edge.relation_id).unwrap_or("RELACAO");
            let target_name = interner
                .resolve(StringId::new(edge.target.as_u32()))
                .unwrap_or("Destino");

            if config.include_edge_weights {
                output.push_str(&format!(
                    "- [{}] --{} ({:.2})--> [{}]\n",
                    source_name, rel_name, edge.weight, target_name
                ));
            } else {
                output.push_str(&format!(
                    "- [{}] --{}--> [{}]\n",
                    source_name, rel_name, target_name
                ));
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::hybrid_score::ScoredEntity;
    use crate::id::EdgeId;

    #[test]
    fn test_hierarchical_markdown_formatting() {
        let mut interner = StringInterner::new();
        let s_titan = interner.intern("Projeto Titan");
        let s_ana = interner.intern("Ana Silva");
        let rel_lidera = interner.intern("LIDERADO_POR");

        let e_titan = ScoredEntity {
            node_id: NodeId::new(s_titan.as_u32()),
            final_score: 0.95,
            vector_score: 0.90,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
        };
        let e_ana = ScoredEntity {
            node_id: NodeId::new(s_ana.as_u32()),
            final_score: 0.85,
            vector_score: 0.20,
            graph_score: 0.95,
            depth: 1,
            path_edge: None,
        };

        let edge = EdgeRecord::new(EdgeId::new(1), e_titan.node_id, e_ana.node_id, rel_lidera)
            .with_weight(0.98);

        let pruned = PrunedSubgraph {
            entities: vec![e_titan, e_ana],
            edges: vec![edge],
            total_tokens: 100,
            budget: 500,
        };

        let config = MarkdownFormatConfig::default();
        let md = format_pruned_subgraph_markdown(&pruned, &interner, &config);

        assert!(md.contains("# Contexto Recuperado do Conhecimento:"));
        assert!(md.contains("- [Projeto Titan] (Relevância: 0.95)"));
        assert!(md.contains("  - LIDERADO_POR -> [Ana Silva] (Confiança: 0.98)"));
        assert!(md.contains("- [Ana Silva] (Relevância: 0.85)"));
    }

    #[test]
    fn test_separated_sections_formatting() {
        let mut interner = StringInterner::new();
        let s1 = interner.intern("Rust");
        let s2 = interner.intern("Backend");
        let rel = interner.intern("USADO_EM");

        let e1 = ScoredEntity {
            node_id: NodeId::new(s1.as_u32()),
            final_score: 0.90,
            vector_score: 0.90,
            graph_score: 1.0,
            depth: 0,
            path_edge: None,
        };

        let edge = EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(s1.as_u32()),
            NodeId::new(s2.as_u32()),
            rel,
        )
        .with_weight(0.9);

        let pruned = PrunedSubgraph {
            entities: vec![e1],
            edges: vec![edge],
            total_tokens: 50,
            budget: 200,
        };

        let config = MarkdownFormatConfig {
            style: MarkdownStyle::SeparatedSections,
            include_scores: false,
            include_edge_weights: false,
            header_title: "Knowledge".to_string(),
        };

        let md = format_pruned_subgraph_markdown(&pruned, &interner, &config);

        assert!(md.contains("# Knowledge:"));
        assert!(md.contains("## Entidades:"));
        assert!(md.contains("- [Rust]"));
        assert!(md.contains("## Relações Estruturais:"));
        assert!(md.contains("- [Rust] --USADO_EM--> [Backend]"));
    }
}
