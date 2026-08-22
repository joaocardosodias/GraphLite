use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::entity_resolution::ResolutionConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::engine::query::QueryOptions;
use graphlite_core::prompt::markdown::MarkdownStyle;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

/// Model Context Protocol (MCP) server exposing GraphLite knowledge graphs to AI agents.
#[derive(Parser, Debug)]
#[command(name = "graphlite-mcp", version, about = "GraphLite MCP stdio server")]
struct ServerArgs {
    /// Path to the .graph database file.
    #[arg(short = 'd', long = "db", default_value = "graphlite.graph")]
    db_path: PathBuf,

    /// Vector dimension.
    #[arg(short = 'D', long, default_value_t = 384)]
    dim: usize,

    /// Ollama API endpoint for local embeddings.
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

fn fetch_ollama_embedding(ollama_url: &str, model: &str, text: &str) -> anyhow::Result<Vec<f32>> {
    let payload = json!({
        "model": model,
        "prompt": text
    });

    let endpoint = format!("{}/api/embeddings", ollama_url.trim_end_matches('/'));

    // Execute curl for zero-dependency HTTP call
    let output = Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg(&endpoint)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(payload.to_string())
        .output()?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Ollama request failed: {}", err_msg);
    }

    let res: Value = serde_json::from_slice(&output.stdout)?;
    if let Some(arr) = res.get("embedding").and_then(|v| v.as_array()) {
        let vec: Result<Vec<f32>, _> = arr.iter().map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| anyhow::anyhow!("Invalid float"))).collect();
        return vec;
    }

    anyhow::bail!("Failed to parse embedding from Ollama response: {}", res)
}

fn list_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "graphlite_retrieve",
                "description": "Retrieves verified architectural context, rules, and connected entities from the GraphLite knowledge graph within a token budget.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or question about the system / architecture."
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
                "name": "graphlite_remember",
                "description": "Stores or updates an entity (rule, architecture decision, struct, module, user preference) in the persistent graph with automatic real-time deduplication.",
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
                            "description": "Detailed description, convention, or architectural behavior."
                        }
                    },
                    "required": ["name", "description"]
                }
            },
            {
                "name": "graphlite_connect",
                "description": "Creates a directed or bidirectional relationship connecting two entities in the knowledge graph.",
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
    engine: &GraphLiteEngine,
    args: &ServerArgs,
    tool_name: &str,
    params: &Value,
) -> anyhow::Result<String> {
    match tool_name {
        "graphlite_retrieve" => {
            let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let max_tokens = params.get("max_tokens").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(400);
            let top_k = params.get("top_k").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(3);

            let query_vec = fetch_ollama_embedding(&args.ollama_url, &args.embedding_model, query)?;

            let options = QueryOptions {
                top_k_seeds: top_k,
                max_tokens: Some(max_tokens),
                markdown_style: MarkdownStyle::Hierarchical,
                max_depth: Some(2),
                min_score_threshold: Some(0.05),
                alpha: Some(0.6),
            };

            let res = engine.retrieve_context(&query_vec, Some(options))?;
            if res.markdown.trim().is_empty() {
                Ok("No relevant context found in GraphLite database for this query.".to_string())
            } else {
                Ok(format!(
                    "Retrieved {} entities and {} relations ({} tokens):\n\n{}",
                    res.entities_count, res.edges_count, res.token_count, res.markdown
                ))
            }
        }
        "graphlite_remember" => {
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let entity_type = params.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let description = params.get("description").and_then(|v| v.as_str()).unwrap_or("");

            let text_to_embed = format!("{} {}: {}", name, entity_type, description);
            let vec = fetch_ollama_embedding(&args.ollama_url, &args.embedding_model, &text_to_embed)?;

            let res_config = ResolutionConfig {
                similarity_threshold: 0.92,
                require_matching_type: true,
                merge_descriptions: true,
            };

            let res = engine.upsert_node_resolved(name, entity_type, description, &vec, Some(res_config))?;
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
        "graphlite_connect" => {
            let source = params.get("source").and_then(|v| v.as_str()).unwrap_or("");
            let target = params.get("target").and_then(|v| v.as_str()).unwrap_or("");
            let relation = params.get("relation").and_then(|v| v.as_str()).unwrap_or("");
            let weight = params.get("weight").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(0.95);

            let src_node = engine.get_node_by_name(source).ok_or_else(|| {
                anyhow::anyhow!("Source node '{}' not found in GraphLite database.", source)
            })?;

            let tgt_node = engine.get_node_by_name(target).ok_or_else(|| {
                anyhow::anyhow!("Target node '{}' not found in GraphLite database.", target)
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

    let config = GraphLiteConfig::new()
        .with_dim(args.dim)
        .with_metric(Metric::Cosine)
        .with_quantization(Quantization::ScalarInt8)
        .with_auto_flush(true);

    let engine = Arc::new(GraphLiteEngine::open_or_create(&args.db_path, config)?);

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

        let response = match req.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "graphlite-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "capabilities": {
                        "tools": {}
                    }
                })),
                error: None,
            },
            "notifications/initialized" => {
                // Client initialized notification, no response required
                continue;
            }
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: Some(json!({})),
                error: None,
            },
            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: Some(list_tools()),
                error: None,
            },
            "tools/call" => {
                let params = req.params.unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").unwrap_or(&Value::Null);

                match handle_tool_call(&engine, &args, tool_name, arguments) {
                    Ok(text) => JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: req.id,
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
                        id: req.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32000,
                            message: e.to_string(),
                        }),
                    },
                }
            }
            other => JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method '{}' not found", other),
                }),
            },
        };

        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }

    Ok(())
}
