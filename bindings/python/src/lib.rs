//! Python native bindings for Graphite GraphRAG Engine.

use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use graphite::engine::{GraphiteConfig, GraphiteEngine, QueryOptions, QueryResult};
use graphite::id::NodeId;
use graphite::vector::distance::Metric;
use graphite::vector::embedding::LocalEmbedder;
use graphite::vector::quantization::Quantization;

/// Structured result of a GraphRAG retrieval query.
#[pyclass(name = "QueryResult")]
#[derive(Clone)]
pub struct PyQueryResult {
    #[pyo3(get)]
    pub markdown: String,
    #[pyo3(get)]
    pub token_count: usize,
    #[pyo3(get)]
    pub entities_count: usize,
    #[pyo3(get)]
    pub edges_count: usize,
}

#[pymethods]
impl PyQueryResult {
    fn __repr__(&self) -> String {
        format!(
            "<QueryResult tokens={} entities={} edges={}>",
            self.token_count, self.entities_count, self.edges_count
        )
    }

    fn __str__(&self) -> String {
        self.markdown.clone()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("markdown", &self.markdown)?;
        dict.set_item("token_count", self.token_count)?;
        dict.set_item("entities_count", self.entities_count)?;
        dict.set_item("edges_count", self.edges_count)?;
        Ok(dict)
    }
}

impl From<QueryResult> for PyQueryResult {
    fn from(res: QueryResult) -> Self {
        Self {
            markdown: res.markdown,
            token_count: res.token_count,
            entities_count: res.entities_count,
            edges_count: res.edges_count,
        }
    }
}

/// Embedded Graphite database engine handle.
#[pyclass(name = "Graphite")]
pub struct PyGraphite {
    engine: Option<Arc<GraphiteEngine>>,
    embedder: Option<Arc<LocalEmbedder>>,
}

#[pymethods]
impl PyGraphite {
    /// Creates or opens a Graphite database.
    #[new]
    #[pyo3(signature = (path = None, dim = 384, metric = "cosine", quantization = "sq8", device = "auto"))]
    fn new(
        path: Option<String>,
        dim: usize,
        metric: &str,
        quantization: &str,
        device: &str,
    ) -> PyResult<Self> {
        let metric_enum = match metric.to_lowercase().as_str() {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            "manhattan" | "l1" => Metric::Manhattan,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported metric: {}",
                    other
                )))
            }
        };

        let quant_enum = match quantization.to_lowercase().as_str() {
            "sq8" | "int8" | "scalar" => Quantization::ScalarInt8,
            "none" | "float32" | "f32" => Quantization::None,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported quantization: {}",
                    other
                )))
            }
        };

        let device_type = graphite::vector::DeviceType::from_str_name(device);

        let config = GraphiteConfig::new()
            .with_dim(dim)
            .with_metric(metric_enum)
            .with_quantization(quant_enum)
            .with_device(device_type);

        let engine = match path {
            Some(p) => GraphiteEngine::open_or_create(Path::new(&p), config)
                .map_err(|e| PyIOError::new_err(e.to_string()))?,
            None => GraphiteEngine::in_memory(config)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };

        let embedder = LocalEmbedder::from_model_type_and_device(
            graphite::vector::EmbeddingModelType::AllMiniLML6V2,
            device_type,
        )
        .ok()
        .map(Arc::new);

        Ok(Self {
            engine: Some(Arc::new(engine)),
            embedder,
        })
    }

    /// Opens an existing or new `.graph` database file.
    #[staticmethod]
    #[pyo3(signature = (path, dim = 384, metric = "cosine", quantization = "sq8", device = "auto"))]
    fn open(
        path: String,
        dim: usize,
        metric: &str,
        quantization: &str,
        device: &str,
    ) -> PyResult<Self> {
        Self::new(Some(path), dim, metric, quantization, device)
    }

    /// Creates an in-memory ephemeral Graphite instance.
    #[staticmethod]
    #[pyo3(signature = (dim = 384, metric = "cosine", quantization = "sq8", device = "auto"))]
    fn in_memory(
        dim: usize,
        metric: &str,
        quantization: &str,
        device: &str,
    ) -> PyResult<Self> {
        Self::new(None, dim, metric, quantization, device)
    }

    /// Queries the knowledge graph using a plain text prompt with automatic local embedding.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (text, top_k = 10, threshold = None, min_score = None, relative_drop_off = Some(0.85), auto_k = None, max_depth = None, alpha = None))]
    fn query(
        &self,
        text: &str,
        top_k: usize,
        threshold: Option<f32>,
        min_score: Option<f32>,
        relative_drop_off: Option<f32>,
        auto_k: Option<f32>,
        max_depth: Option<usize>,
        alpha: Option<f32>,
    ) -> PyResult<PyQueryResult> {
        let engine = self.get_engine()?;

        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Local embedder is not available"))?;

        let vector = embedder
            .embed_one(text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut options = QueryOptions::default()
            .with_query_text(text)
            .with_top_k(top_k);

        if let Some(t) = threshold.or(min_score) {
            options = options.with_threshold(t);
        }
        if let Some(drop) = auto_k.or(relative_drop_off) {
            options = options.with_auto_k(drop);
        }
        if let Some(md) = max_depth {
            options = options.with_max_depth(md);
        }
        if let Some(a) = alpha {
            options = options.with_alpha(a);
        }

        let res = engine
            .retrieve_context(&vector, Some(options))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(res.into())
    }

    /// Retrieves prompt context using an explicit query vector.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vector, query_text = None, top_k = 10, threshold = None, min_score = None, relative_drop_off = Some(0.85), auto_k = None, max_depth = None, alpha = None))]
    fn retrieve_context(
        &self,
        vector: Vec<f32>,
        query_text: Option<String>,
        top_k: usize,
        threshold: Option<f32>,
        min_score: Option<f32>,
        relative_drop_off: Option<f32>,
        auto_k: Option<f32>,
        max_depth: Option<usize>,
        alpha: Option<f32>,
    ) -> PyResult<PyQueryResult> {
        let engine = self.get_engine()?;

        let mut options = QueryOptions::default().with_top_k(top_k);
        if let Some(t) = query_text {
            options = options.with_query_text(t);
        }
        if let Some(thresh) = threshold.or(min_score) {
            options = options.with_threshold(thresh);
        }
        if let Some(drop) = auto_k.or(relative_drop_off) {
            options = options.with_auto_k(drop);
        }
        if let Some(md) = max_depth {
            options = options.with_max_depth(md);
        }
        if let Some(a) = alpha {
            options = options.with_alpha(a);
        }

        let res = engine
            .retrieve_context(&vector, Some(options))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(res.into())
    }

    /// Inserts or updates an entity node with optional embedding vector.
    #[pyo3(signature = (name, entity_type = "Entity", description = "", vector = None))]
    fn insert(
        &self,
        name: &str,
        entity_type: &str,
        description: &str,
        vector: Option<Vec<f32>>,
    ) -> PyResult<u32> {
        let engine = self.get_engine()?;

        let vec_ref = vector.as_deref();
        let id = engine
            .upsert_node(name, entity_type, description, vec_ref)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(id.as_u32())
    }

    /// Connects two entity names with a directed relationship.
    #[pyo3(signature = (source_name, target_name, relation = "RELATES_TO", weight = 1.0))]
    fn connect(
        &self,
        source_name: &str,
        target_name: &str,
        relation: &str,
        weight: f32,
    ) -> PyResult<()> {
        let engine = self.get_engine()?;

        let src_id = engine
            .upsert_node(source_name, "Entity", "", None)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let tgt_id = engine
            .upsert_node(target_name, "Entity", "", None)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        engine
            .add_edge(src_id, tgt_id, relation, weight, true)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    /// Adds a relationship edge between two node IDs.
    #[pyo3(signature = (source_id, target_id, relation = "RELATES_TO", weight = 1.0, directed = true))]
    fn add_edge(
        &self,
        source_id: u32,
        target_id: u32,
        relation: &str,
        weight: f32,
        directed: bool,
    ) -> PyResult<()> {
        let engine = self.get_engine()?;

        engine
            .add_edge(
                NodeId::new(source_id),
                NodeId::new(target_id),
                relation,
                weight,
                directed,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }

    /// Persists all pending in-memory mutations atomically to disk.
    fn flush(&self) -> PyResult<()> {
        let engine = self.get_engine()?;
        engine
            .flush()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Returns statistics about the database.
    fn inspect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let engine = self.get_engine()?;

        let dict = PyDict::new(py);
        dict.set_item("nodes_count", engine.node_count())?;
        dict.set_item("edges_count", engine.edge_count())?;
        dict.set_item("vectors_count", engine.vector_count())?;
        dict.set_item("vector_dim", engine.config().vector_dim)?;
        dict.set_item("is_in_memory", engine.path().is_none())?;
        Ok(dict)
    }

    /// Closes the database and releases virtual memory mappings.
    fn close(&mut self) -> PyResult<()> {
        if let Some(engine) = self.engine.take() {
            let _ = engine.flush();
        }
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<()> {
        self.close()
    }
}

impl PyGraphite {
    fn get_engine(&self) -> PyResult<&Arc<GraphiteEngine>> {
        self.engine
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Database is closed"))
    }
}

/// Generates a 384-dimensional embedding vector locally on CPU using FastEmbed (MiniLM-L6-v2).
#[pyfunction]
fn embed(text: &str) -> PyResult<Vec<f32>> {
    let embedder =
        LocalEmbedder::new_minilm().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    embedder
        .embed_one(text)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Generates embedding vectors for a batch of text strings.
#[pyfunction]
fn embed_batch(texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
    let embedder =
        LocalEmbedder::new_minilm().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    embedder
        .embed_batch(&texts)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Native module initialization.
#[pymodule]
fn _graphite_db(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphite>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed_batch, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
