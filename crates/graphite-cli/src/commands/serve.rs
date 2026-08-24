use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tiny_http::{Header, Method, Response, Server, StatusCode};

use graphite_core::engine::config::GraphiteConfig;
use graphite_core::engine::instance::GraphiteEngine;
use graphite_core::engine::query::QueryOptions;
use graphite_core::storage::mmap_reader::MmapGraphReader;
use graphite_core::vector::distance::Metric;
use graphite_core::vector::quantization::Quantization;
use graphite_core::LocalEmbedder;

use crate::args::{IngestArgs, ServeArgs};
use crate::ingestion::run_ingest_pass;

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub max_tokens: Option<usize>,
    pub top_k_seeds: Option<usize>,
    pub type_filter: Option<Vec<String>>,
    #[allow(dead_code)]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub text: String,
    pub category: Option<String>,
    pub relate_to: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub path: String,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub extensions: Option<String>,
    pub force: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub query: String,
    pub token_count: usize,
    pub context: String,
    pub format: String,
    pub latency_ms: f64,
}

fn load_or_default_config(db_path: &Path) -> GraphiteConfig {
    if db_path.exists() {
        if let Ok(reader) = MmapGraphReader::open(db_path) {
            let dim = reader.header().vector_dim as usize;
            let metric = match reader.header().metric_type {
                1 => Metric::DotProduct,
                2 => Metric::Euclidean,
                3 => Metric::Manhattan,
                _ => Metric::Cosine,
            };
            let quant = if reader.header().is_quantized() {
                Quantization::ScalarInt8
            } else {
                Quantization::None
            };
            return GraphiteConfig::new()
                .with_dim(dim)
                .with_metric(metric)
                .with_quantization(quant);
        }
    }
    GraphiteConfig::new().with_dim(384)
}

fn json_response<T: Serialize>(data: &T, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(data).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = Response::from_data(body).with_status_code(StatusCode(status));
    if let Ok(h) = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]) {
        resp.add_header(h);
    }
    if let Ok(h) = Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]) {
        resp.add_header(h);
    }
    if let Ok(h) = Header::from_bytes(
        &b"Access-Control-Allow-Methods"[..],
        &b"GET, POST, OPTIONS"[..],
    ) {
        resp.add_header(h);
    }
    if let Ok(h) = Header::from_bytes(
        &b"Access-Control-Allow-Headers"[..],
        &b"Content-Type, Authorization"[..],
    ) {
        resp.add_header(h);
    }
    resp
}

fn handle_options() -> Response<std::io::Cursor<Vec<u8>>> {
    let mut resp = Response::from_data(Vec::new()).with_status_code(StatusCode(204));
    if let Ok(h) = Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]) {
        resp.add_header(h);
    }
    if let Ok(h) = Header::from_bytes(
        &b"Access-Control-Allow-Methods"[..],
        &b"GET, POST, OPTIONS"[..],
    ) {
        resp.add_header(h);
    }
    if let Ok(h) = Header::from_bytes(
        &b"Access-Control-Allow-Headers"[..],
        &b"Content-Type, Authorization"[..],
    ) {
        resp.add_header(h);
    }
    resp
}

pub fn execute_serve(db_path: &Path, args: &ServeArgs) -> Result<()> {
    let bind_addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&bind_addr)
        .map_err(|e| anyhow::anyhow!("Failed to bind HTTP server to {}: {}", bind_addr, e))?;

    let config = load_or_default_config(db_path);
    let engine = Arc::new(RwLock::new(GraphiteEngine::open_or_create(
        db_path, config,
    )?));
    let embedder = Arc::new(
        LocalEmbedder::new_minilm()
            .with_context(|| "Failed to initialize local ONNX embedding model")?,
    );

    println!("========================================================");
    println!("  🚀 Graphite Embedded REST API Server Running");
    println!("========================================================");
    println!("  • Base URL:     http://{}", bind_addr);
    println!("  • Database:     {}", db_path.display());
    println!("  • Total Nodes:  {}", engine.read().node_count());
    println!("  • Total Edges:  {}", engine.read().edge_count());
    println!("  • Endpoints:");
    println!("    - GET  /v1/health   -> Check server & database stats");
    println!("    - POST /v1/query    -> GraphRAG context retrieval");
    println!("    - POST /v1/remember -> Store agent memory / facts");
    println!("    - POST /v1/ingest   -> Ingest documents on demand");
    println!("========================================================");
    println!("Listening for requests... (Press Ctrl+C to stop)\n");

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        if method == Method::Options {
            let _ = request.respond(handle_options());
            continue;
        }

        let db_path_buf = db_path.to_path_buf();
        let engine_clone = Arc::clone(&engine);
        let embedder_clone = Arc::clone(&embedder);

        let mut body_bytes = Vec::new();
        let _ = request.as_reader().read_to_end(&mut body_bytes);

        let response = match (method, url.as_str()) {
            (Method::Get, "/health") | (Method::Get, "/v1/health") | (Method::Get, "/") => {
                let eng = engine_clone.read();
                let cache_stats = eng.cache_stats();
                json_response(
                    &json!({
                        "status": "ok",
                        "version": env!("CARGO_PKG_VERSION"),
                        "database": db_path_buf.display().to_string(),
                        "nodes": eng.node_count(),
                        "edges": eng.edge_count(),
                        "vectors": eng.vector_count(),
                        "cache": {
                            "capacity": cache_stats.capacity,
                            "entries": cache_stats.entries,
                            "hits": cache_stats.hits,
                            "misses": cache_stats.misses,
                            "hit_rate_pct": cache_stats.hit_rate
                        },
                        "engine": "Graphite GraphRAG (Pure Rust)"
                    }),
                    200,
                )
            }

            (Method::Post, "/v1/cache/clear") | (Method::Post, "/cache/clear") => {
                let eng = engine_clone.read();
                eng.clear_cache();
                json_response(
                    &json!({
                        "status": "success",
                        "message": "Query context LRU cache cleared successfully"
                    }),
                    200,
                )
            }

            (Method::Post, "/v1/query") | (Method::Post, "/query") => {
                match serde_json::from_slice::<QueryRequest>(&body_bytes) {
                    Ok(req) => {
                        let start = Instant::now();
                        match embedder_clone.embed_one(&req.query) {
                            Ok(query_vec) => {
                                let eng = engine_clone.read();
                                let options = QueryOptions {
                                    top_k_seeds: req.top_k_seeds.unwrap_or(5),
                                    max_tokens: req.max_tokens,
                                    type_filter: req.type_filter,
                                    redundancy_threshold: Some(0.82),
                                    ..Default::default()
                                };

                                match eng.retrieve_context(&query_vec, Some(options)) {
                                    Ok(res) => {
                                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                                        let resp_obj = QueryResponse {
                                            query: req.query,
                                            token_count: res.token_count,
                                            context: res.markdown,
                                            format: "markdown".to_string(),
                                            latency_ms: (elapsed_ms * 100.0).round() / 100.0,
                                        };
                                        json_response(&resp_obj, 200)
                                    }
                                    Err(e) => json_response(
                                        &json!({"error": format!("Retrieval failed: {}", e)}),
                                        500,
                                    ),
                                }
                            }
                            Err(e) => json_response(
                                &json!({"error": format!("Embedding failed: {}", e)}),
                                500,
                            ),
                        }
                    }
                    Err(e) => json_response(
                        &json!({"error": format!("Invalid JSON request payload: {}", e)}),
                        400,
                    ),
                }
            }

            (Method::Post, "/v1/remember") | (Method::Post, "/remember") => {
                match serde_json::from_slice::<RememberRequest>(&body_bytes) {
                    Ok(req) => {
                        let text = req.text.trim();
                        if text.is_empty() {
                            json_response(&json!({"error": "Memory text cannot be empty"}), 400)
                        } else {
                            let category =
                                req.category.unwrap_or_else(|| "AgentMemory".to_string());
                            match embedder_clone.embed_one(text) {
                                Ok(vector) => {
                                    let eng = engine_clone.write();
                                    let preview: String = text.chars().take(40).collect();
                                    let memory_name = format!("{}: {}", category, preview.trim());

                                    match eng.upsert_node(
                                        &memory_name,
                                        &category,
                                        text,
                                        Some(&vector),
                                    ) {
                                        Ok(node_id) => {
                                            if let Some(ref rel_name) = req.relate_to {
                                                if let Some(target) = eng.get_node_by_name(rel_name)
                                                {
                                                    let _ = eng.add_edge(
                                                        node_id,
                                                        target.id,
                                                        "RELATES_TO",
                                                        0.90,
                                                        true,
                                                    );
                                                }
                                            }
                                            let _ = eng.flush();
                                            json_response(
                                                &json!({
                                                    "status": "success",
                                                    "node_id": node_id.as_u32(),
                                                    "name": memory_name,
                                                    "category": category,
                                                    "total_nodes": eng.node_count(),
                                                }),
                                                201,
                                            )
                                        }
                                        Err(e) => json_response(
                                            &json!({"error": format!("Storage failed: {}", e)}),
                                            500,
                                        ),
                                    }
                                }
                                Err(e) => json_response(
                                    &json!({"error": format!("Embedding failed: {}", e)}),
                                    500,
                                ),
                            }
                        }
                    }
                    Err(e) => json_response(
                        &json!({"error": format!("Invalid JSON payload: {}", e)}),
                        400,
                    ),
                }
            }

            (Method::Post, "/v1/ingest") | (Method::Post, "/ingest") => {
                match serde_json::from_slice::<IngestRequest>(&body_bytes) {
                    Ok(req) => {
                        let ingest_args = IngestArgs {
                            path: std::path::PathBuf::from(&req.path),
                            chunk_size: req.chunk_size.unwrap_or(350),
                            chunk_overlap: req.chunk_overlap.unwrap_or(40),
                            extensions: req.extensions,
                            max_files: 1000,
                            watch: false,
                            force: req.force.unwrap_or(false),
                            no_tmp: false,
                        };

                        let start = Instant::now();
                        match run_ingest_pass(&db_path_buf, &ingest_args, &embedder_clone, false) {
                            Ok(modified) => {
                                if modified {
                                    let new_cfg = load_or_default_config(&db_path_buf);
                                    if let Ok(new_eng) =
                                        GraphiteEngine::open_or_create(&db_path_buf, new_cfg)
                                    {
                                        *engine_clone.write() = new_eng;
                                    }
                                }
                                let eng = engine_clone.read();
                                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                                json_response(
                                    &json!({
                                        "status": "success",
                                        "modified": modified,
                                        "total_nodes": eng.node_count(),
                                        "total_edges": eng.edge_count(),
                                        "elapsed_ms": (elapsed * 100.0).round() / 100.0,
                                    }),
                                    200,
                                )
                            }
                            Err(e) => json_response(
                                &json!({"error": format!("Ingestion failed: {}", e)}),
                                500,
                            ),
                        }
                    }
                    Err(e) => json_response(
                        &json!({"error": format!("Invalid JSON request: {}", e)}),
                        400,
                    ),
                }
            }

            _ => json_response(&json!({"error": "Endpoint not found", "path": url}), 404),
        };

        let _ = request.respond(response);
    }

    Ok(())
}
