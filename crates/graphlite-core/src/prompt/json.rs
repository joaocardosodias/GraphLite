use crate::id::StringId;
use crate::interner::StringInterner;
use crate::prompt::pruner::PrunedSubgraph;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Structured JSON representation of an entity node for tool-calling agents.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JsonEntity {
    /// Unique integer identifier.
    pub id: u32,
    /// Human-readable resolved name.
    pub name: String,
    /// Hybrid relevance score.
    pub score: f32,
}

/// Structured JSON representation of an active relationship edge.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JsonRelation {
    /// Identifier of the source entity.
    pub source_id: u32,
    /// Resolved name of the source entity.
    pub source: String,
    /// Resolved relationship type name.
    pub relation: String,
    /// Identifier of the target entity.
    pub target_id: u32,
    /// Resolved name of the target entity.
    pub target: String,
    /// Confidence weight of the connection.
    pub weight: f32,
}

/// Complete machine-readable payload representing a retrieved knowledge subgraph.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JsonSubgraphPayload {
    /// List of retrieved entities.
    pub entities: Vec<JsonEntity>,
    /// List of interconnecting relations.
    pub relations: Vec<JsonRelation>,
    /// Exact token footprint.
    pub total_tokens: usize,
    /// Token budget constraint.
    pub budget: usize,
}

/// Converts a `PrunedSubgraph` into a `JsonSubgraphPayload` data structure.
pub fn to_json_payload(
    subgraph: &PrunedSubgraph,
    interner: &StringInterner,
) -> JsonSubgraphPayload {
    let mut entities = Vec::with_capacity(subgraph.entities.len());
    for e in &subgraph.entities {
        let name = interner
            .resolve(StringId::new(e.node_id.as_u32()))
            .unwrap_or("Entidade")
            .to_string();

        entities.push(JsonEntity {
            id: e.node_id.as_u32(),
            name,
            score: e.final_score,
        });
    }

    let mut relations = Vec::with_capacity(subgraph.edges.len());
    for edge in &subgraph.edges {
        let source_name = interner
            .resolve(StringId::new(edge.source.as_u32()))
            .unwrap_or("Origem")
            .to_string();
        let rel_name = interner
            .resolve(edge.relation_id)
            .unwrap_or("RELACAO")
            .to_string();
        let target_name = interner
            .resolve(StringId::new(edge.target.as_u32()))
            .unwrap_or("Destino")
            .to_string();

        relations.push(JsonRelation {
            source_id: edge.source.as_u32(),
            source: source_name,
            relation: rel_name,
            target_id: edge.target.as_u32(),
            target: target_name,
            weight: edge.weight,
        });
    }

    JsonSubgraphPayload {
        entities,
        relations,
        total_tokens: subgraph.total_tokens,
        budget: subgraph.budget,
    }
}

/// Formats a `PrunedSubgraph` into a serialized JSON string.
#[cfg(feature = "serde")]
pub fn format_subgraph_json(
    subgraph: &PrunedSubgraph,
    interner: &StringInterner,
) -> Result<String, serde_json::Error> {
    let payload = to_json_payload(subgraph, interner);
    serde_json::to_string_pretty(&payload)
}

/// Formats a `PrunedSubgraph` into a list of concise raw triples: `(Subject, Relation, Object)`.
pub fn format_subgraph_triples(
    subgraph: &PrunedSubgraph,
    interner: &StringInterner,
) -> Vec<String> {
    let mut triples = Vec::with_capacity(subgraph.edges.len());

    for edge in &subgraph.edges {
        let source_name = interner
            .resolve(StringId::new(edge.source.as_u32()))
            .unwrap_or("Origem");
        let rel_name = interner.resolve(edge.relation_id).unwrap_or("RELACAO");
        let target_name = interner
            .resolve(StringId::new(edge.target.as_u32()))
            .unwrap_or("Destino");

        triples.push(format!("({}, {}, {})", source_name, rel_name, target_name));
    }

    triples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::hybrid_score::ScoredEntity;
    use crate::id::{EdgeId, NodeId};
    use crate::record::EdgeRecord;

    #[test]
    fn test_to_json_payload_and_triples() {
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
            total_tokens: 80,
            budget: 500,
        };

        let payload = to_json_payload(&pruned, &interner);

        assert_eq!(payload.entities.len(), 2);
        assert_eq!(payload.entities[0].name, "Projeto Titan");
        assert_eq!(payload.relations.len(), 1);
        assert_eq!(payload.relations[0].relation, "LIDERADO_POR");
        assert_eq!(payload.relations[0].source, "Projeto Titan");
        assert_eq!(payload.relations[0].target, "Ana Silva");

        // Triples format
        let triples = format_subgraph_triples(&pruned, &interner);
        assert_eq!(triples, vec!["(Projeto Titan, LIDERADO_POR, Ana Silva)"]);

        #[cfg(feature = "serde")]
        {
            let json_str = format_subgraph_json(&pruned, &interner).unwrap();
            assert!(json_str.contains("\"name\": \"Projeto Titan\""));
            assert!(json_str.contains("\"relation\": \"LIDERADO_POR\""));
        }
    }
}
