use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::engine::config::GraphLiteConfig;
use crate::error::Result;
use crate::graph::adjacency::AdjacencyGraph;
use crate::graph::bm25::Bm25Index;
use crate::id::NodeId;
use crate::interner::StringInterner;
use crate::record::{EdgeRecord, NodeRecord};
use crate::storage::atomic_writer::write_database_atomic;
use crate::storage::mmap_reader::MmapGraphReader;
use crate::vector::store::VectorStore;

/// Internal mutable state of the GraphLite database engine.
pub(crate) struct EngineState {
    pub interner: StringInterner,
    pub graph: AdjacencyGraph,
    pub vectors: VectorStore,
    pub bm25: Bm25Index,
    pub dirty: bool,
}

/// The core public database engine handle for GraphLite.
///
/// Thread-safe (`Send + Sync`), cloneable via `Arc`, with concurrency control
/// managed by `parking_lot::RwLock` for high-throughput reads and writes.
#[derive(Clone)]
pub struct GraphLiteEngine {
    pub(crate) path: Option<PathBuf>,
    pub(crate) config: GraphLiteConfig,
    pub(crate) state: Arc<RwLock<EngineState>>,
}

impl GraphLiteEngine {
    /// Opens an existing `.graph` database file from disk, or creates a new one if it does not exist.
    ///
    /// # Example
    ///
    /// ```rust
    /// use graphlite_core::engine::{GraphLiteConfig, GraphLiteEngine};
    /// use tempfile::tempdir;
    ///
    /// let dir = tempdir().unwrap();
    /// let db_path = dir.path().join("demo.graph");
    /// let config = GraphLiteConfig::new().with_dim(4);
    ///
    /// let db = GraphLiteEngine::open_or_create(&db_path, config).unwrap();
    /// assert_eq!(db.node_count(), 0);
    /// ```
    pub fn open_or_create<P: AsRef<Path>>(path: P, config: GraphLiteConfig) -> Result<Self> {
        config.validate()?;
        let p = path.as_ref().to_path_buf();

        if p.exists() {
            // Load existing database from disk via zero-copy mmap
            let reader = MmapGraphReader::open(&p)?;

            // 1. Rebuild in-memory StringInterner
            let interner = reader
                .string_table()
                .map(|st| st.to_interner())
                .unwrap_or_else(|_| StringInterner::new());

            // 2. Rebuild in-memory dynamic AdjacencyGraph and BM25 index
            let mut graph = AdjacencyGraph::new();
            let mut bm25 = Bm25Index::new();
            let mut node_ids = Vec::new();

            if let Ok(node_block) = reader.nodes() {
                for i in 0..node_block.len() {
                    if let Some(node) = node_block.get_by_index(i) {
                        if node.is_active() {
                            graph.add_node(node)?;
                            node_ids.push(node.id);

                            let name = interner.resolve(node.name_id).unwrap_or("");
                            let ty = interner.resolve(node.type_id).unwrap_or("");
                            let desc = interner.resolve(node.description_id).unwrap_or("");
                            let text = format!("{} {} {}", name, ty, desc);
                            bm25.index_node(node.id, &text);
                        }
                    }
                }
            }

            if let Ok(csr_block) = reader.csr() {
                for node_id_u32 in 0..csr_block.node_count() as u32 {
                    for edge in csr_block.out_edges(NodeId::new(node_id_u32)) {
                        if edge.is_active() {
                            let _ = graph.add_edge(edge);
                        }
                    }
                }
            }

            // 3. Rebuild in-memory VectorStore
            let mut vectors = VectorStore::new(config.vector_dim, config.metric, config.quantization);

            if let Ok(vec_block) = reader.vectors() {
                for (i, node_id) in node_ids.into_iter().enumerate() {
                    if let Some(qv) = vec_block.get(i) {
                        vectors.insert_quantized(node_id, qv)?;
                    }
                }
            }

            let state = EngineState {
                interner,
                graph,
                vectors,
                bm25,
                dirty: false,
            };

            Ok(Self {
                path: Some(p),
                config,
                state: Arc::new(RwLock::new(state)),
            })
        } else {
            // Create fresh database
            let vectors = VectorStore::new(config.vector_dim, config.metric, config.quantization);

            let state = EngineState {
                interner: StringInterner::new(),
                graph: AdjacencyGraph::new(),
                vectors,
                bm25: Bm25Index::new(),
                dirty: true,
            };

            let engine = Self {
                path: Some(p),
                config,
                state: Arc::new(RwLock::new(state)),
            };

            if engine.config.auto_flush {
                engine.flush()?;
            }

            Ok(engine)
        }
    }

    /// Creates an ephemeral, in-memory only `GraphLiteEngine` instance without disk persistence.
    ///
    /// # Example
    ///
    /// ```rust
    /// use graphlite_core::engine::{GraphLiteConfig, GraphLiteEngine};
    ///
    /// let config = GraphLiteConfig::new().with_dim(4);
    /// let db = GraphLiteEngine::in_memory(config).unwrap();
    ///
    /// assert_eq!(db.node_count(), 0);
    /// assert_eq!(db.edge_count(), 0);
    /// ```
    pub fn in_memory(config: GraphLiteConfig) -> Result<Self> {
        config.validate()?;
        let vectors = VectorStore::new(config.vector_dim, config.metric, config.quantization);

        let state = EngineState {
            interner: StringInterner::new(),
            graph: AdjacencyGraph::new(),
            vectors,
            bm25: Bm25Index::new(),
            dirty: false,
        };

        Ok(Self {
            path: None,
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Flushes all in-memory mutations to disk atomically.
    pub fn flush(&self) -> Result<()> {
        let path = match &self.path {
            Some(p) => p,
            None => return Ok(()), // In-memory database has no disk backing
        };

        let mut state = self.state.write();
        if !state.dirty && path.exists() {
            return Ok(());
        }

        // 1. Compile in-memory AdjacencyGraph into compact CSR
        let csr = state.graph.to_csr();

        // 2. Extract active node records sorted by NodeId
        let mut nodes: Vec<NodeRecord> = state.graph.nodes().copied().collect();
        nodes.sort_by_key(|n| n.id.as_u32());

        // 3. Extract quantized vectors in NodeId order
        let mut vectors_vec = Vec::with_capacity(nodes.len());
        for node in &nodes {
            if let Some(qv) = state.vectors.get_quantized(node.id) {
                vectors_vec.push(qv);
            }
        }

        // 4. Atomically persist to disk with CRC32 checksum
        let metric_u8 = match self.config.metric {
            crate::vector::distance::Metric::Cosine => 0,
            crate::vector::distance::Metric::DotProduct => 1,
            crate::vector::distance::Metric::Euclidean => 2,
            crate::vector::distance::Metric::Manhattan => 3,
        };

        write_database_atomic(
            path,
            &nodes,
            &csr,
            &vectors_vec,
            &state.interner,
            self.config.vector_dim,
            metric_u8,
        )?;

        state.dirty = false;
        Ok(())
    }

    /// Returns a copy of the active database configuration.
    #[inline]
    pub fn config(&self) -> &GraphLiteConfig {
        &self.config
    }

    /// Returns the file path backing this engine, or `None` if in-memory.
    #[inline]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Returns the total number of active nodes in the database.
    pub fn node_count(&self) -> usize {
        self.state.read().graph.node_count()
    }

    /// Returns the total number of active edges in the database.
    pub fn edge_count(&self) -> usize {
        self.state.read().graph.edge_count()
    }

    /// Returns the total number of stored vectors in the database.
    pub fn vector_count(&self) -> usize {
        self.state.read().vectors.len()
    }

    /// Returns `true` if there are unsaved in-memory mutations.
    pub fn is_dirty(&self) -> bool {
        self.state.read().dirty
    }

    /// Returns a vector copy of all active `NodeRecord`s in the database.
    pub fn all_nodes(&self) -> Vec<NodeRecord> {
        self.state.read().graph.nodes().copied().collect()
    }

    /// Returns a vector copy of all active `EdgeRecord`s in the database.
    pub fn all_edges(&self) -> Vec<EdgeRecord> {
        self.state.read().graph.edges().copied().collect()
    }

    /// Returns the `NodeRecord` corresponding to `id` if found.
    pub fn get_node(&self, id: NodeId) -> Option<NodeRecord> {
        self.state.read().graph.get_node(id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_engine_in_memory_lifecycle() {
        let config = GraphLiteConfig::new().with_dim(128);
        let engine = GraphLiteEngine::in_memory(config).unwrap();

        assert_eq!(engine.node_count(), 0);
        assert_eq!(engine.edge_count(), 0);
        assert_eq!(engine.vector_count(), 0);
        assert!(engine.path().is_none());
        assert!(!engine.is_dirty());
    }

    #[test]
    fn test_engine_open_or_create_disk_lifecycle() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("lifecycle.graph");

        let config = GraphLiteConfig::new().with_dim(64);

        // 1. Create fresh database
        {
            let engine = GraphLiteEngine::open_or_create(&db_path, config.clone()).unwrap();
            assert_eq!(engine.node_count(), 0);
            assert!(db_path.exists());
        }

        // 2. Reopen existing database
        {
            let reopened = GraphLiteEngine::open_or_create(&db_path, config).unwrap();
            assert_eq!(reopened.node_count(), 0);
        }
    }
}
