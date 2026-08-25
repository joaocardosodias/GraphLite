//! # Graphite
//!
//! The official Rust client and embedded engine for Graphite: single-file GraphRAG,
//! knowledge graphs, and AI agent memory.
//!
//! ## Quickstart
//!
//! ```rust
//! use graphite::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Open or create an in-memory or disk-backed Graphite database
//!     let db = Graphite::in_memory()?;
//!
//!     // 2. Ingest entities and relations
//!     let auth_id = db.insert_node("AuthService", "Module", "Handles user authentication and JWTs", None)?;
//!     let db_id = db.insert_node("Database", "Infrastructure", "PostgreSQL primary cluster", None)?;
//!     db.connect("AuthService", "Database", "CONNECTS_TO", 0.95)?;
//!
//!     // 3. Query the knowledge graph with hybrid GraphRAG retrieval
//!     let result = db.query("How does authentication connect to the database?")?;
//!     println!("Retrieved tokens: {}", result.token_count);
//!     println!("{}", result.markdown);
//!
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(feature = "embedded")]
pub use graphite_core as core;

#[cfg(feature = "embedded")]
pub use graphite_core::engine::{GraphiteConfig, GraphiteEngine, QueryOptions, QueryResult};
#[cfg(feature = "embedded")]
pub use graphite_core::error::{GraphiteError, Result as CoreResult};
#[cfg(feature = "embedded")]
pub use graphite_core::id::{EdgeId, NodeId, StringId};
#[cfg(feature = "embedded")]
pub use graphite_core::vector::{Metric, Quantization};

pub mod prelude {
    pub use crate::Graphite;
    #[cfg(feature = "embedded")]
    pub use graphite_core::engine::{GraphiteConfig, GraphiteEngine, QueryOptions, QueryResult};
    #[cfg(feature = "embedded")]
    pub use graphite_core::id::{EdgeId, NodeId, StringId};
    #[cfg(feature = "embedded")]
    pub use graphite_core::vector::{Metric, Quantization};
}

/// Internal backend execution mode.
#[derive(Clone)]
enum Backend {
    #[cfg(feature = "embedded")]
    Embedded(GraphiteEngine),
    Binary {
        binary_path: PathBuf,
        db_path: PathBuf,
    },
}

/// High-level Graphite database handle.
///
/// Supports embedded in-process execution with zero-copy mmap or delegation to
/// the pre-installed system `graphite` CLI binary.
#[derive(Clone)]
pub struct Graphite {
    backend: Backend,
}

impl Graphite {
    /// Opens or creates an embedded `.graphite` database file with default configuration.
    #[cfg(feature = "embedded")]
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let config = GraphiteConfig::default();
        let engine = GraphiteEngine::open_or_create(path, config)?;
        Ok(Self {
            backend: Backend::Embedded(engine),
        })
    }

    /// Opens or creates an embedded `.graphite` database file with custom configuration.
    #[cfg(feature = "embedded")]
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: GraphiteConfig) -> anyhow::Result<Self> {
        let engine = GraphiteEngine::open_or_create(path, config)?;
        Ok(Self {
            backend: Backend::Embedded(engine),
        })
    }

    /// Creates an ephemeral in-memory database instance.
    #[cfg(feature = "embedded")]
    pub fn in_memory() -> anyhow::Result<Self> {
        let config = GraphiteConfig::default();
        let engine = GraphiteEngine::in_memory(config)?;
        Ok(Self {
            backend: Backend::Embedded(engine),
        })
    }

    /// Connects to a `.graphite` database by invoking the installed system `graphite` CLI binary on `$PATH`.
    pub fn system<P: AsRef<Path>>(db_path: P) -> anyhow::Result<Self> {
        let binary_path = which_binary("graphite")
            .ok_or_else(|| anyhow::anyhow!("Graphite binary not found on PATH. Run install.sh to install it."))?;
        Ok(Self {
            backend: Backend::Binary {
                binary_path,
                db_path: db_path.as_ref().to_path_buf(),
            },
        })
    }

    /// Connects to a `.graphite` database by specifying a custom binary path.
    pub fn from_binary<P: AsRef<Path>, B: AsRef<Path>>(db_path: P, binary_path: B) -> Self {
        Self {
            backend: Backend::Binary {
                binary_path: binary_path.as_ref().to_path_buf(),
                db_path: db_path.as_ref().to_path_buf(),
            },
        }
    }

    /// Queries the knowledge graph using hybrid vector and lexical search with GraphRAG synthesis.
    pub fn query(&self, text: &str) -> anyhow::Result<QueryResult> {
        match &self.backend {
            #[cfg(feature = "embedded")]
            Backend::Embedded(engine) => {
                let options = QueryOptions {
                    query_text: Some(text.to_string()),
                    max_tokens: Some(400),
                    ..Default::default()
                };

                // Use fastembed local embeddings if available
                #[cfg(feature = "fastembed")]
                {
                    let embedder = graphite_core::vector::embedding::LocalEmbedder::new_minilm()?;
                    let vector = embedder.embed_one(text)?;
                    let res = engine.retrieve_context(&vector, Some(options))?;
                    Ok(res)
                }

                #[cfg(not(feature = "fastembed"))]
                {
                    anyhow::bail!("FastEmbed feature is required for plain-text queries in embedded mode. Enable feature 'fastembed' or pass explicit query vectors.");
                }
            }
            Backend::Binary { binary_path, db_path } => {
                let output = Command::new(binary_path)
                    .arg("-d")
                    .arg(db_path)
                    .arg("query")
                    .arg(text)
                    .output()?;

                if !output.status.success() {
                    let err = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!("Graphite binary query failed: {}", err);
                }

                let markdown = String::from_utf8_lossy(&output.stdout).to_string();
                let token_count = markdown.split_whitespace().count();

                Ok(QueryResult {
                    markdown,
                    token_count,
                    entities_count: 0,
                    edges_count: 0,
                    scored_entities: Vec::new(),
                    pruned_subgraph: graphite_core::prompt::pruner::PrunedSubgraph::default(),
                })
            }
        }
    }

    /// Inserts or updates an entity node with optional embedding vector.
    #[cfg(feature = "embedded")]
    pub fn insert_node(
        &self,
        name: &str,
        entity_type: &str,
        description: &str,
        vector: Option<&[f32]>,
    ) -> anyhow::Result<NodeId> {
        match &self.backend {
            Backend::Embedded(engine) => {
                let id = engine.upsert_node(name, entity_type, description, vector)?;
                Ok(id)
            }
            Backend::Binary { .. } => {
                anyhow::bail!("Direct node insertion with raw vectors is supported in embedded mode.");
            }
        }
    }

    /// Connects two entity names with a directed, weighted relationship.
    #[cfg(feature = "embedded")]
    pub fn connect(
        &self,
        source_name: &str,
        target_name: &str,
        relation: &str,
        weight: f32,
    ) -> anyhow::Result<()> {
        match &self.backend {
            Backend::Embedded(engine) => {
                let src_id = engine.upsert_node(source_name, "Entity", "", None)?;
                let tgt_id = engine.upsert_node(target_name, "Entity", "", None)?;
                engine.add_edge(src_id, tgt_id, relation, weight, true)?;
                Ok(())
            }
            Backend::Binary { binary_path, db_path } => {
                let output = Command::new(binary_path)
                    .arg("-d")
                    .arg(db_path)
                    .arg("insert-edge")
                    .arg(source_name)
                    .arg(target_name)
                    .arg(relation)
                    .arg(weight.to_string())
                    .output()?;

                if !output.status.success() {
                    let err = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!("Graphite binary connect failed: {}", err);
                }
                Ok(())
            }
        }
    }

    /// Persists all pending in-memory mutations atomically to disk.
    #[cfg(feature = "embedded")]
    pub fn flush(&self) -> anyhow::Result<()> {
        match &self.backend {
            Backend::Embedded(engine) => {
                engine.flush()?;
                Ok(())
            }
            Backend::Binary { .. } => Ok(()),
        }
    }
}

fn which_binary(name: &str) -> Option<PathBuf> {
    if let Ok(paths) = std::env::var("PATH") {
        for path in std::env::split_paths(&paths) {
            let full_path = path.join(name);
            if full_path.is_file() {
                return Some(full_path);
            }
            #[cfg(windows)]
            {
                let exe_path = path.join(format!("{}.exe", name));
                if exe_path.is_file() {
                    return Some(exe_path);
                }
            }
        }
    }
    None
}
