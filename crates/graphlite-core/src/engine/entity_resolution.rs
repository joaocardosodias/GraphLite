use crate::engine::instance::GraphLiteEngine;
use crate::error::{GraphLiteError, Result};
use crate::id::{NodeId, StringId};
use crate::record::NodeRecord;

/// Configuration parameters for automatic real-time entity resolution and deduplication.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolutionConfig {
    /// Vector cosine similarity threshold above which two entities are considered identical (default: 0.92).
    pub similarity_threshold: f32,
    /// If `true`, requires matching entity types (e.g. both must be "Person" or "Technology") before merging.
    pub require_matching_type: bool,
    /// If `true`, concatenates new descriptions to existing ones upon merging.
    pub merge_descriptions: bool,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            require_matching_type: true,
            merge_descriptions: true,
        }
    }
}

/// The result of an entity resolution operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionResult {
    /// The canonical `NodeId` representing this entity in the graph.
    pub node_id: NodeId,
    /// `true` if this entity was merged into an existing semantically duplicate node.
    pub is_merged: bool,
    /// The ID of the existing node that was matched, if merged.
    pub matched_existing_id: Option<NodeId>,
}

impl GraphLiteEngine {
    /// Inserts an entity with real-time semantic deduplication and entity resolution.
    ///
    /// If an existing entity of matching type has vector cosine similarity $\ge$ `similarity_threshold`
    /// (e.g. "Rust Lang" vs "Rust Programming Language"), it automatically merges them into a single
    /// consolidated entity node, avoiding fragmentation in the knowledge graph.
    pub fn upsert_node_resolved(
        &self,
        name: &str,
        entity_type: &str,
        description: &str,
        vector: &[f32],
        config: Option<ResolutionConfig>,
    ) -> Result<ResolutionResult> {
        let conf = config.unwrap_or_default();
        let trimmed_name = name.trim();

        if trimmed_name.is_empty() {
            return Err(GraphLiteError::CorruptedFormat(
                "Entity name cannot be empty".to_string(),
            ));
        }

        if vector.len() != self.config().vector_dim {
            return Err(GraphLiteError::VectorDimensionMismatch {
                expected: self.config().vector_dim,
                found: vector.len(),
            });
        }

        let resolution_result = {
            let mut state = self.state.write();

            let name_id = state.interner.intern(trimmed_name);
            let type_id = if entity_type.trim().is_empty() {
                StringId::INVALID
            } else {
                state.interner.intern(entity_type.trim())
            };

            // 1. Exact string match check
            let exact_match_id = state
                .graph
                .nodes()
                .find(|n| n.name_id == name_id)
                .map(|n| n.id);

            if let Some(existing_id) = exact_match_id {
                // Exact name exists: update description and vector
                let mut full_desc = description.to_string();
                if !description.trim().is_empty() {
                    let desc_id = state.interner.intern(description.trim());
                    if let Some(record) = state.graph.get_node(existing_id).copied() {
                        let mut updated = record;
                        updated.description_id = desc_id;
                        updated.type_id = type_id;
                        state.graph.upsert_node(updated);
                    }
                    full_desc = description.trim().to_string();
                }
                state.vectors.insert(existing_id, vector)?;
                let bm25_text = format!("{} {} {}", trimmed_name, entity_type, full_desc);
                state.bm25.index_node(existing_id, &bm25_text);
                state.dirty = true;

                ResolutionResult {
                    node_id: existing_id,
                    is_merged: false,
                    matched_existing_id: None,
                }
            } else {
                // 2. Semantic vector proximity search for candidate merge targets
                let candidates = state.vectors.search(vector, 3)?;
                let mut best_merge_candidate: Option<NodeId> = None;

                for (cand_id, score) in candidates {
                    if score >= conf.similarity_threshold {
                        if let Some(cand_node) = state.graph.get_node(cand_id) {
                            if !conf.require_matching_type || cand_node.type_id == type_id {
                                best_merge_candidate = Some(cand_id);
                                break;
                            }
                        }
                    }
                }

                if let Some(target_id) = best_merge_candidate {
                    // Merge with existing candidate node!
                    let mut combined_desc = description.to_string();
                    if conf.merge_descriptions && !description.trim().is_empty() {
                        let old_desc = state
                            .graph
                            .get_node(target_id)
                            .and_then(|n| state.interner.resolve(n.description_id))
                            .unwrap_or("")
                            .to_string();

                        combined_desc = if old_desc.is_empty() {
                            description.trim().to_string()
                        } else if !old_desc.contains(description.trim()) {
                            format!("{}; {}", old_desc, description.trim())
                        } else {
                            old_desc
                        };

                        let combined_desc_id = state.interner.intern(&combined_desc);
                        if let Some(record) = state.graph.get_node(target_id).copied() {
                            let mut updated = record;
                            updated.description_id = combined_desc_id;
                            state.graph.upsert_node(updated);
                        }
                    }

                    let bm25_text = format!("{} {} {}", trimmed_name, entity_type, combined_desc);
                    state.bm25.index_node(target_id, &bm25_text);
                    state.dirty = true;
                    ResolutionResult {
                        node_id: target_id,
                        is_merged: true,
                        matched_existing_id: Some(target_id),
                    }
                } else {
                    // 3. No match found: create new node
                    let new_id = NodeId::new(name_id.as_u32());
                    let desc_id = if description.trim().is_empty() {
                        StringId::INVALID
                    } else {
                        state.interner.intern(description.trim())
                    };

                    let vector_offset =
                        (new_id.as_u32() as u64) * (8 + self.config().vector_dim as u64);
                    let record = NodeRecord::new(new_id, name_id, type_id, desc_id, vector_offset);

                    state.graph.add_node(record)?;
                    state.vectors.insert(new_id, vector)?;
                    let bm25_text = format!("{} {} {}", trimmed_name, entity_type, description);
                    state.bm25.index_node(new_id, &bm25_text);
                    state.dirty = true;

                    ResolutionResult {
                        node_id: new_id,
                        is_merged: false,
                        matched_existing_id: None,
                    }
                }
            }
        };

        if self.config().auto_flush {
            self.flush()?;
        }

        Ok(resolution_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::GraphLiteConfig;

    #[test]
    fn test_entity_resolution_and_merging() {
        let config = GraphLiteConfig::new().with_dim(4);
        let engine = GraphLiteEngine::in_memory(config).unwrap();

        let v1 = [1.0, 0.0, 0.0, 0.0];
        // Highly similar vector (cosine similarity ~ 0.999)
        let v2 = [0.999, 0.001, 0.0, 0.0];
        // Completely different vector
        let v3 = [0.0, 1.0, 0.0, 0.0];

        // 1. Insert "Linguagem Rust"
        let r1 = engine
            .upsert_node_resolved(
                "Linguagem Rust",
                "Tecnologia",
                "Linguagem focada em segurança",
                &v1,
                None,
            )
            .unwrap();

        assert!(!r1.is_merged);
        assert_eq!(engine.node_count(), 1);

        // 2. Insert "Rust Programming Language" (near-identical semantic vector)
        let r2 = engine
            .upsert_node_resolved(
                "Rust Programming Language",
                "Tecnologia",
                "Alta performance e zero-cost abstractions",
                &v2,
                Some(ResolutionConfig {
                    similarity_threshold: 0.95,
                    require_matching_type: true,
                    merge_descriptions: true,
                }),
            )
            .unwrap();

        // Must be merged into the existing node!
        assert!(r2.is_merged);
        assert_eq!(r2.node_id, r1.node_id);
        assert_eq!(r2.matched_existing_id, Some(r1.node_id));
        assert_eq!(engine.node_count(), 1); // No new node created!

        // Verify description was merged
        let node = engine.get_node_by_name("Linguagem Rust").unwrap();
        let desc = engine.resolve_string(node.description_id).unwrap();
        assert!(desc.contains("segurança"));
        assert!(desc.contains("zero-cost abstractions"));

        // 3. Insert "Python" (different vector)
        let r3 = engine
            .upsert_node_resolved("Python", "Tecnologia", "Linguagem dinâmica", &v3, None)
            .unwrap();

        assert!(!r3.is_merged);
        assert_eq!(engine.node_count(), 2);
    }
}
