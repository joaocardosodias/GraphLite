use std::path::Path;
use anyhow::{bail, Result};
use serde::Serialize;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliDumpFormat, DumpArgs};

#[derive(Serialize)]
struct FullGraphDumpJson {
    nodes: Vec<NodeDumpJson>,
    edges: Vec<EdgeDumpJson>,
}

#[derive(Serialize)]
struct NodeDumpJson {
    id: u32,
    name: String,
    entity_type: String,
    description: String,
}

#[derive(Serialize)]
struct EdgeDumpJson {
    id: u32,
    source_name: String,
    target_name: String,
    relation: String,
    weight: f32,
    directed: bool,
}

fn load_or_default_config(db_path: &Path) -> GraphLiteConfig {
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
            return GraphLiteConfig::new()
                .with_dim(dim)
                .with_metric(metric)
                .with_quantization(quant);
        }
    }
    GraphLiteConfig::default()
}

pub fn execute_dump(db_path: &Path, args: &DumpArgs) -> Result<()> {
    if !db_path.exists() {
        bail!("Database file '{:?}' not found.", db_path);
    }

    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let nodes = engine.all_nodes();
    let edges = engine.all_edges();

    match args.format {
        CliDumpFormat::Json => {
            let json_nodes: Vec<NodeDumpJson> = nodes
                .iter()
                .map(|n| {
                    let name = engine.resolve_string(n.name_id).unwrap_or_default();
                    let entity_type = engine.resolve_string(n.type_id).unwrap_or_default();
                    let description = engine.resolve_string(n.description_id).unwrap_or_default();
                    NodeDumpJson {
                        id: n.id.as_u32(),
                        name,
                        entity_type,
                        description,
                    }
                })
                .collect();

            let json_edges: Vec<EdgeDumpJson> = edges
                .iter()
                .map(|e| {
                    let source_name = engine
                        .get_node(e.source)
                        .and_then(|n| engine.resolve_string(n.name_id))
                        .unwrap_or_default();

                    let target_name = engine
                        .get_node(e.target)
                        .and_then(|n| engine.resolve_string(n.name_id))
                        .unwrap_or_default();

                    let relation = engine.resolve_string(e.relation_id).unwrap_or_default();

                    EdgeDumpJson {
                        id: e.id.as_u32(),
                        source_name,
                        target_name,
                        relation,
                        weight: e.weight,
                        directed: e.is_directed(),
                    }
                })
                .collect();

            let dump_payload = FullGraphDumpJson {
                nodes: json_nodes,
                edges: json_edges,
            };

            println!("{}", serde_json::to_string_pretty(&dump_payload)?);
        }
        CliDumpFormat::Markdown => {
            println!("# GraphLite Knowledge Base\n");
            println!("## Entities ({} nodes):\n", nodes.len());
            for n in &nodes {
                let name = engine.resolve_string(n.name_id).unwrap_or_default();
                let entity_type = engine.resolve_string(n.type_id).unwrap_or_default();
                let description = engine.resolve_string(n.description_id).unwrap_or_default();
                if !entity_type.is_empty() {
                    println!("### {} [{}]", name, entity_type);
                } else {
                    println!("### {}", name);
                }
                if !description.is_empty() {
                    println!("> {}\n", description);
                } else {
                    println!();
                }
            }

            println!("## Relationships ({} edges):\n", edges.len());
            for e in &edges {
                let source_name = engine
                    .get_node(e.source)
                    .and_then(|n| engine.resolve_string(n.name_id))
                    .unwrap_or_default();

                let target_name = engine
                    .get_node(e.target)
                    .and_then(|n| engine.resolve_string(n.name_id))
                    .unwrap_or_default();

                let relation = engine.resolve_string(e.relation_id).unwrap_or_default();
                println!("- **{}** --[{}]--> **{}** (weight: {:.2})", source_name, relation, target_name, e.weight);
            }
        }
        CliDumpFormat::Triples => {
            for e in &edges {
                let source_name = engine
                    .get_node(e.source)
                    .and_then(|n| engine.resolve_string(n.name_id))
                    .unwrap_or_default();

                let target_name = engine
                    .get_node(e.target)
                    .and_then(|n| engine.resolve_string(n.name_id))
                    .unwrap_or_default();

                let relation = engine.resolve_string(e.relation_id).unwrap_or_default();
                println!("(\"{}\") -[:{}]-> (\"{})\";", source_name, relation, target_name);
            }
        }
    }

    Ok(())
}
