use crate::engine::instance::GraphLiteEngine;
use crate::error::{GraphLiteError, Result};
use crate::id::{EdgeId, NodeId, StringId};
use crate::record::{EdgeRecord, NodeRecord, NO_VECTOR_OFFSET};

impl GraphLiteEngine {
    /// Inserts a new node or updates an existing one with the given attributes and optional vector.
    ///
    /// If a node with the same name already exists, its metadata and vector are updated in place.
    pub fn upsert_node(
        &self,
        name: &str,
        entity_type: &str,
        description: &str,
        vector: Option<&[f32]>,
    ) -> Result<NodeId> {
        let trimmed_name = name.trim();
        if trimmed_name.is_empty() {
            return Err(GraphLiteError::CorruptedFormat(
                "Node name cannot be empty".to_string(),
            ));
        }

        if let Some(v) = vector {
            if v.len() != self.config().vector_dim {
                return Err(GraphLiteError::VectorDimensionMismatch {
                    expected: self.config().vector_dim,
                    found: v.len(),
                });
            }
        }

        let node_id = {
            let mut state = self.state.write();

            let name_id = state.interner.intern(trimmed_name);
            let type_id = if entity_type.trim().is_empty() {
                StringId::INVALID
            } else {
                state.interner.intern(entity_type.trim())
            };
            let description_id = if description.trim().is_empty() {
                StringId::INVALID
            } else {
                state.interner.intern(description.trim())
            };

            // Check if node already exists by name_id
            let existing_id = state
                .graph
                .nodes()
                .find(|n| n.name_id == name_id)
                .map(|n| n.id);

            let target_node_id = match existing_id {
                Some(id) => {
                    // Update existing record
                    let mut record = *state.graph.get_node(id).unwrap();
                    record.type_id = type_id;
                    record.description_id = description_id;
                    if vector.is_some() {
                        record.vector_offset = (id.as_u32() as u64) * (8 + self.config().vector_dim as u64);
                    }
                    state.graph.add_node(record)?;
                    id
                }
                None => {
                    // Create new record (deterministic NodeId derived from name_id)
                    let new_id = NodeId::new(name_id.as_u32());
                    let vector_offset = if vector.is_some() {
                        (new_id.as_u32() as u64) * (8 + self.config().vector_dim as u64)
                    } else {
                        NO_VECTOR_OFFSET
                    };

                    let record = NodeRecord::new(
                        new_id,
                        name_id,
                        type_id,
                        description_id,
                        vector_offset,
                    );
                    state.graph.add_node(record)?;
                    new_id
                }
            };

            // Insert or update vector in VectorStore
            if let Some(v) = vector {
                state.vectors.insert(target_node_id, v)?;
            }

            state.dirty = true;
            target_node_id
        };

        if self.config().auto_flush {
            self.flush()?;
        }

        Ok(node_id)
    }

    /// Adds a directed or undirected relationship (edge) between two entities.
    pub fn add_edge(
        &self,
        source: NodeId,
        target: NodeId,
        relation: &str,
        weight: f32,
        directed: bool,
    ) -> Result<EdgeId> {
        let trimmed_relation = relation.trim();
        if trimmed_relation.is_empty() {
            return Err(GraphLiteError::CorruptedFormat(
                "Edge relation cannot be empty".to_string(),
            ));
        }

        let edge_id = {
            let mut state = self.state.write();

            if !state.graph.contains_node(source) {
                return Err(GraphLiteError::NodeNotFound(source));
            }
            if !state.graph.contains_node(target) {
                return Err(GraphLiteError::NodeNotFound(target));
            }

            let relation_id = state.interner.intern(trimmed_relation);
            let next_edge_id = EdgeId::new((state.graph.edge_count() + 1) as u32);

            let edge = EdgeRecord::new(next_edge_id, source, target, relation_id)
                .with_weight(weight.clamp(0.0, 1.0))
                .with_directed(directed);

            state.graph.add_edge(edge)?;
            state.dirty = true;
            next_edge_id
        };

        if self.config().auto_flush {
            self.flush()?;
        }

        Ok(edge_id)
    }

    /// Retrieves a `NodeRecord` by its textual name.
    pub fn get_node_by_name(&self, name: &str) -> Option<NodeRecord> {
        let state = self.state.read();
        let name_id = state.interner.get_id(name.trim())?;
        let node = state.graph.nodes().find(|n| n.name_id == name_id).copied();
        node
    }

    /// Resolves a `StringId` to an owned `String`.
    pub fn resolve_string(&self, id: StringId) -> Option<String> {
        let state = self.state.read();
        state.interner.resolve(id).map(|s| s.to_string())
    }

    /// Removes a node and cascades removal of all incident edges.
    pub fn remove_node(&self, id: NodeId) -> Result<bool> {
        let removed = {
            let mut state = self.state.write();
            let was_removed = state.graph.remove_node(id).is_some();
            if was_removed {
                state.dirty = true;
            }
            was_removed
        };

        if removed && self.config().auto_flush {
            self.flush()?;
        }

        Ok(removed)
    }

    /// Removes a specific edge by its `EdgeId`.
    pub fn remove_edge(&self, id: EdgeId) -> Result<bool> {
        let removed = {
            let mut state = self.state.write();
            let was_removed = state.graph.remove_edge(id).is_some();
            if was_removed {
                state.dirty = true;
            }
            was_removed
        };

        if removed && self.config().auto_flush {
            self.flush()?;
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::GraphLiteConfig;

    #[test]
    fn test_engine_upsert_and_mutation_workflow() {
        let config = GraphLiteConfig::new().with_dim(4);
        let engine = GraphLiteEngine::in_memory(config).unwrap();

        let v_titan = [0.1, 0.2, 0.3, 0.4];
        let v_ana = [0.5, 0.6, 0.7, 0.8];

        let id_titan = engine
            .upsert_node("Projeto Titan", "Projeto", "IA Generativa", Some(&v_titan))
            .unwrap();

        let id_ana = engine
            .upsert_node("Ana Silva", "Pessoa", "Tech Lead", Some(&v_ana))
            .unwrap();

        assert_eq!(engine.node_count(), 2);
        assert_eq!(engine.vector_count(), 2);

        // Add edge
        let edge_id = engine
            .add_edge(id_ana, id_titan, "LIDERA", 0.95, true)
            .unwrap();

        assert_eq!(engine.edge_count(), 1);

        // Test name lookup
        let fetched_titan = engine.get_node_by_name("Projeto Titan").unwrap();
        assert_eq!(fetched_titan.id, id_titan);
        assert_eq!(
            engine.resolve_string(fetched_titan.name_id),
            Some("Projeto Titan".to_string())
        );

        // Remove edge and node
        assert!(engine.remove_edge(edge_id).unwrap());
        assert_eq!(engine.edge_count(), 0);

        assert!(engine.remove_node(id_titan).unwrap());
        assert_eq!(engine.node_count(), 1);
    }
}
