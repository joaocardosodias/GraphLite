use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use graphite_core::engine::config::GraphiteConfig;
use graphite_core::engine::entity_resolution::ResolutionConfig;
use graphite_core::engine::instance::GraphiteEngine;
use graphite_core::engine::query::QueryOptions;
use graphite_core::prompt::markdown::MarkdownStyle;
use graphite_core::vector::distance::Metric;
use graphite_core::vector::embedding::LocalEmbedder;
use graphite_core::vector::quantization::Quantization;

/// Model Context Protocol (MCP) server exposing Graphite knowledge graphs to AI agents.
#[derive(Parser, Debug)]
#[command(name = "graphite-mcp", version, about = "Graphite MCP stdio server")]
struct ServerArgs {
    /// Path to the .graph database file.
    #[arg(short = 'd', long = "db", default_value = "graphite.graphite")]
    db_path: PathBuf,

    /// Vector dimension.
    #[arg(short = 'D', long, default_value_t = 384)]
    dim: usize,

    /// Ollama API endpoint for fallback embeddings.
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Ollama embedding model name.
    #[arg(long, default_value = "all-minilm")]
    embedding_model: String,
}

#[derive(Deserialize, Serialize, Debug)]
struct JsonRpcRequest {
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize, Debug)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize, Debug)]
struct JsonRpcError {
    code: i64,
    message: String,
}

/// Computes text embeddings with Primary Priority #1 given to Pure Rust LocalEmbedder (ONNX).
/// Falls back to Ollama HTTP only if local engine cannot be initialized.
fn compute_auto_embedding(
    local_embedder: Option<&LocalEmbedder>,
    ollama_url: &str,
    ollama_model: &str,
    text: &str,
) -> anyhow::Result<Vec<f32>> {
    // Priority #1: Pure Rust In-Memory ONNX Runtime (Zero External Services, ~2ms latency)
    if let Some(embedder) = local_embedder {
        if let Ok(v) = embedder.embed_one(text) {
            return Ok(v);
        }
    }

    // Priority #2: Fallback to local Ollama daemon
    let payload = json!({
        "model": ollama_model,
        "prompt": text
    });

    let endpoint = format!("{}/api/embeddings", ollama_url.trim_end_matches('/'));

    let output = Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg(&endpoint)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(payload.to_string())
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            if let Ok(res) = serde_json::from_slice::<Value>(&out.stdout) {
                if let Some(arr) = res.get("embedding").and_then(|v| v.as_array()) {
                    let vec: Result<Vec<f32>, _> = arr
                        .iter()
                        .map(|x| {
                            x.as_f64()
                                .map(|f| f as f32)
                                .ok_or_else(|| anyhow::anyhow!("Invalid float in embedding array"))
                        })
                        .collect();
                    if let Ok(v) = vec {
                        return Ok(v);
                    }
                }
            }
        }
    }

    anyhow::bail!("Failed to generate vector embedding via local FastEmbed or Ollama.")
}

fn list_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "graphite_retrieve",
                "description": "Retrieves verified architectural context, rules, and connected entities from the Graphite knowledge graph. You only need to pass a plain text query — embeddings are computed automatically in-memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Plain text search query or question (e.g. 'How does authentication work?')."
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum token budget for the returned Markdown context (default: 400)."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of entry seed entities to match (default: 3)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "graphite_remember",
                "description": "Stores or updates an entity (rule, architecture decision, struct, module, user preference) in the persistent graph with automatic real-time deduplication. You only need to pass plain text — embeddings are computed automatically in-memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The entity name or label (e.g. 'AuthService', 'RateLimitingRule')."
                        },
                        "type": {
                            "type": "string",
                            "description": "Entity category (e.g. 'Struct', 'Rule', 'Project', 'UserPreference')."
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description, convention, or architectural behavior in plain text."
                        }
                    },
                    "required": ["name", "description"]
                }
            },
            {
                "name": "graphite_connect",
                "description": "Creates a directed relationship connecting two entities in the knowledge graph.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Name of the source entity."
                        },
                        "target": {
                            "type": "string",
                            "description": "Name of the target entity."
                        },
                        "relation": {
                            "type": "string",
                            "description": "Relationship label (e.g. 'USES', 'LEADS', 'MUST_OBEY', 'DEPENDS_ON')."
                        },
                        "weight": {
                            "type": "number",
                            "description": "Connection weight between 0.0 and 1.0 (default: 0.95)."
                        }
                    },
                    "required": ["source", "target", "relation"]
                }
            }
        ]
    })
}

fn handle_tool_call(
    engine: &GraphiteEngine,
    local_embedder: Option<&LocalEmbedder>,
    args: &ServerArgs,
    tool_name: &str,
    params: &Value,
) -> anyhow::Result<String> {
    match tool_name {
        "graphite_retrieve" => {
            let query = params
                .get("query")
                .or_else(|| params.get("Query"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let max_tokens = params
                .get("max_tokens")
                .or_else(|| params.get("maxTokens"))
                .or_else(|| params.get("MaxTokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(400);
            let top_k = params
                .get("top_k")
                .or_else(|| params.get("topK"))
                .or_else(|| params.get("TopK"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(3);

            if query.trim().is_empty() {
                anyhow::bail!("Query parameter cannot be empty");
            }

            // High-priority automatic in-memory embedding
            let query_vec = compute_auto_embedding(
                local_embedder,
                &args.ollama_url,
                &args.embedding_model,
                query,
            )?;

            let type_filter: Option<Vec<String>> = params
                .get("entity_type")
                .or_else(|| params.get("type"))
                .and_then(|v| v.as_str())
                .map(|s| {
                    s.split(',')
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty())
                        .collect()
                });

            let options = QueryOptions {
                top_k_seeds: top_k,
                query_text: Some(query.to_string()),
                max_tokens: Some(max_tokens),
                markdown_style: MarkdownStyle::Hierarchical,
                max_depth: Some(2),
                min_score_threshold: Some(0.0),
                alpha: Some(0.6),
                relative_drop_off: Some(0.60),
                redundancy_threshold: Some(0.82),
                type_filter,
            };

            let res = engine.retrieve_context(&query_vec, Some(options))?;
            if res.markdown.trim().is_empty() {
                Ok("No relevant context found in Graphite database for this query.".to_string())
            } else {
                Ok(format!(
                    "Retrieved {} entities and {} relations ({} tokens):\n\n{}",
                    res.entities_count, res.edges_count, res.token_count, res.markdown
                ))
            }
        }
        "graphite_remember" => {
            let name = params
                .get("name")
                .or_else(|| params.get("Name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let entity_type = params
                .get("type")
                .or_else(|| params.get("Type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let description = params
                .get("description")
                .or_else(|| params.get("Description"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if name.trim().is_empty() || description.trim().is_empty() {
                anyhow::bail!(
                    "'name' and 'description' are required parameters for graphite_remember"
                );
            }

            let text_to_embed = format!("{} {}: {}", name, entity_type, description);
            // High-priority automatic in-memory embedding
            let vec = compute_auto_embedding(
                local_embedder,
                &args.ollama_url,
                &args.embedding_model,
                &text_to_embed,
            )?;

            let res_config = ResolutionConfig {
                similarity_threshold: 0.92,
                require_matching_type: true,
                merge_descriptions: true,
            };

            let res = engine.upsert_node_resolved(
                name,
                entity_type,
                description,
                &vec,
                Some(res_config),
            )?;
            engine.flush()?;

            if res.is_merged {
                Ok(format!(
                    "Entity '{}' was semantically resolved and merged into existing node ID {}.",
                    name, res.node_id
                ))
            } else {
                Ok(format!(
                    "Entity '{}' successfully created as a new node with ID {}.",
                    name, res.node_id
                ))
            }
        }
        "graphite_connect" => {
            let source = params
                .get("source")
                .or_else(|| params.get("Source"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let target = params
                .get("target")
                .or_else(|| params.get("Target"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let relation = params
                .get("relation")
                .or_else(|| params.get("Relation"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let weight = params
                .get("weight")
                .or_else(|| params.get("Weight"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.95);

            let src_node = engine.get_node_by_name(source).ok_or_else(|| {
                anyhow::anyhow!("Source node '{}' not found in Graphite database.", source)
            })?;

            let tgt_node = engine.get_node_by_name(target).ok_or_else(|| {
                anyhow::anyhow!("Target node '{}' not found in Graphite database.", target)
            })?;

            let edge_id = engine.add_edge(src_node.id, tgt_node.id, relation, weight, true)?;
            engine.flush()?;

            Ok(format!(
                "Created relationship: '{}' --[{}]--> '{}' (edge ID: {}).",
                source, relation, target, edge_id
            ))
        }
        _ => anyhow::bail!("Unknown tool: {}", tool_name),
    }
}

fn main() -> anyhow::Result<()> {
    let args = ServerArgs::parse();

    let config = GraphiteConfig::new()
        .with_dim(args.dim)
        .with_metric(Metric::Cosine)
        .with_quantization(Quantization::ScalarInt8)
        .with_auto_flush(true);

    let db_path = if args.db_path == std::path::Path::new("graphite.graphite")
        && !args.db_path.exists()
        && std::path::Path::new("app.graphite").exists()
    {
        PathBuf::from("app.graphite")
    } else {
        args.db_path.clone()
    };

    let engine = Arc::new(GraphiteEngine::open_or_create(&db_path, config)?);

    // Initialize in-memory LocalEmbedder once at startup for sub-millisecond embeddings
    let local_embedder = LocalEmbedder::new_minilm().ok();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line_str = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let trimmed = line_str.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err_res = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                    }),
                };
                let _ = writeln!(stdout, "{}", serde_json::to_string(&err_res)?);
                let _ = stdout.flush();
                continue;
            }
        };

        let req_id = req.id;
        let is_notification = req_id.is_none();

        let response = match req.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req_id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "graphite-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {
                        "tools": {}
                    }
                })),
                error: None,
            },
            "notifications/initialized" | "initialized" => {
                continue;
            }
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req_id,
                result: Some(json!({})),
                error: None,
            },
            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req_id,
                result: Some(list_tools()),
                error: None,
            },
            "resources/list" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req_id,
                result: Some(json!({ "resources": [] })),
                error: None,
            },
            "prompts/list" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req_id,
                result: Some(json!({ "prompts": [] })),
                error: None,
            },
            "tools/call" => {
                let params = req.params.unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").unwrap_or(&Value::Null);

                match handle_tool_call(
                    &engine,
                    local_embedder.as_ref(),
                    &args,
                    tool_name,
                    arguments,
                ) {
                    Ok(text) => JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: req_id,
                        result: Some(json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": text
                                }
                            ]
                        })),
                        error: None,
                    },
                    Err(e) => JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: req_id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32000,
                            message: e.to_string(),
                        }),
                    },
                }
            }
            other => {
                if is_notification {
                    // JSON-RPC 2.0: Do not reply to notifications
                    continue;
                }
                JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: req_id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32601,
                        message: format!("Method '{}' not found", other),
                    }),
                }
            }
        };

        if !is_notification {
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }
    }

    Ok(())
}
