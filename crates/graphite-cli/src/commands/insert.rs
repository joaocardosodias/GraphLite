use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

use graphite::engine::config::GraphiteConfig;
use graphite::engine::entity_resolution::ResolutionConfig;
use graphite::engine::instance::GraphiteEngine;
use graphite::storage::mmap_reader::MmapGraphReader;
use graphite::vector::distance::Metric;
use graphite::vector::quantization::Quantization;

use crate::args::{InsertEdgeArgs, InsertNodeArgs};

fn parse_vector_arg(raw: Option<&str>) -> Result<Option<Vec<f32>>> {
    let s = match raw {
        Some(v) => v.trim(),
        None => return Ok(None),
    };

    if s.is_empty() {
        return Ok(None);
    }

    if (s.ends_with(".json") || Path::new(s).exists()) && Path::new(s).is_file() {
        let content = fs::read_to_string(s)
            .with_context(|| format!("Failed to read vector JSON file: {}", s))?;
        let vector: Vec<f32> = serde_json::from_str(&content).with_context(|| {
            "Invalid JSON format: expected an array of numbers (e.g. [0.1, 0.2, ...])"
        })?;
        return Ok(Some(vector));
    }

    let mut vector = Vec::new();
    let cleaned = s.trim_start_matches('[').trim_end_matches(']');
    for part in cleaned.split([',', ' ']) {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let val = trimmed.parse::<f32>().with_context(|| {
                format!("Invalid numeric value '{}' in vector argument. Expected comma-separated floats.", trimmed)
            })?;
            vector.push(val);
        }
    }

    Ok(Some(vector))
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
                .with_quantization(quant)
                .with_models(
                    reader.header().embedding_model_id(),
                    reader.header().reranker_model_id(),
                );
        }
    }
    GraphiteConfig::default()
}

pub fn execute_insert_node(db_path: &Path, args: &InsertNodeArgs) -> Result<()> {
    let config = load_or_default_config(db_path);
    let engine = GraphiteEngine::open_or_create(db_path, config)?;

    let parsed_vector = if let Some(raw) = args.vector.as_deref() {
        parse_vector_arg(Some(raw))?
    } else if args.auto_embed {
        let text_to_embed = format!("{} {}: {}", args.name, args.entity_type, args.description);
        let emb_type = graphite::vector::embedding::EmbeddingModelType::from_id(
            engine.config().embedding_model_id,
            engine.config().vector_dim,
        );
        let embedder = graphite::LocalEmbedder::from_model_type(emb_type).with_context(|| {
            format!(
                "Failed to initialize local ONNX embedding model ({})",
                emb_type.name()
            )
        })?;
        Some(embedder.embed_one(&text_to_embed)?)
    } else {
        None
    };

    if let Some(ref v) = parsed_vector {
        if v.len() != engine.config().vector_dim {
            bail!(
                "Vector dimension mismatch: database expects {} dimensions, but received {}.",
                engine.config().vector_dim,
                v.len()
            );
        }
    }

    if args.resolve {
        let res_config = ResolutionConfig {
            similarity_threshold: 0.92,
            require_matching_type: true,
            merge_descriptions: true,
        };

        let result = if let Some(ref v) = parsed_vector {
            engine.upsert_node_resolved(
                &args.name,
                &args.entity_type,
                &args.description,
                v,
                Some(res_config),
            )?
        } else {
            let id = engine.upsert_node(&args.name, &args.entity_type, &args.description, None)?;
            graphite::engine::ResolutionResult {
                node_id: id,
                is_merged: false,
                matched_existing_id: None,
            }
        };

        engine.flush()?;

        if result.is_merged {
            println!("Entity resolved and merged into existing node.");
            println!("  Node ID:     {}", result.node_id);
            println!("  Name:        '{}'", args.name);
            println!("  Merged With: ID {:?}", result.matched_existing_id);
        } else {
            println!("Node inserted successfully.");
            println!("  Node ID:     {}", result.node_id);
            println!("  Name:        '{}'", args.name);
            println!("  Type:        '{}'", args.entity_type);
        }
    } else {
        let node_id = engine.upsert_node(
            &args.name,
            &args.entity_type,
            &args.description,
            parsed_vector.as_deref(),
        )?;
        engine.flush()?;

        println!("Node inserted successfully.");
        println!("  Node ID:     {}", node_id);
        println!("  Name:        '{}'", args.name);
        println!("  Type:        '{}'", args.entity_type);
    }

    Ok(())
}

pub fn execute_insert_edge(db_path: &Path, args: &InsertEdgeArgs) -> Result<()> {
    if !db_path.exists() {
        bail!(
            "Database file '{:?}' not found. Initialize with 'graphite init' first.",
            db_path
        );
    }

    let config = load_or_default_config(db_path);
    let engine = GraphiteEngine::open_or_create(db_path, config)?;

    let source_node = engine
        .get_node_by_name(&args.source)
        .with_context(|| format!("Source node '{}' not found in database.", args.source))?;

    let target_node = engine
        .get_node_by_name(&args.target)
        .with_context(|| format!("Target node '{}' not found in database.", args.target))?;

    let edge_id = engine.add_edge(
        source_node.id,
        target_node.id,
        &args.relation,
        args.weight,
        args.directed,
    )?;

    engine.flush()?;

    println!("Edge created successfully.");
    println!("  Edge ID:     {}", edge_id);
    println!(
        "  Connection:  '{}' --[{}]--> '{}'",
        args.source, args.relation, args.target
    );
    println!("  Weight:      {:.2}", args.weight);
    println!(
        "  Directed:    {}",
        if args.directed {
            "true"
        } else {
            "false (bidirectional)"
        }
    );

    Ok(())
}
