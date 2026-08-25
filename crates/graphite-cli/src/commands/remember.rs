//! Agent Long-Term Memory and Knowledge Recording command.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use graphite::engine::config::GraphiteConfig;
use graphite::engine::instance::GraphiteEngine;
use graphite::storage::mmap_reader::MmapGraphReader;
use graphite::vector::distance::Metric;
use graphite::vector::quantization::Quantization;
use graphite::LocalEmbedder;

use crate::args::RememberArgs;

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

pub fn execute_remember(db_path: &Path, args: &RememberArgs) -> Result<()> {
    let start_time = Instant::now();
    let memory_text = args.text.trim();
    if memory_text.is_empty() {
        anyhow::bail!("Memory text cannot be empty.");
    }

    let config = load_or_default_config(db_path);
    let engine = GraphiteEngine::open_or_create(db_path, config)?;

    // Embed memory text using configured model
    let emb_type = graphite::vector::embedding::EmbeddingModelType::from_id(
        engine.config().embedding_model_id,
        engine.config().vector_dim,
    );
    let embedder = LocalEmbedder::from_model_type(emb_type).with_context(|| {
        format!(
            "Failed to initialize local ONNX embedding model ({})",
            emb_type.name()
        )
    })?;
    let vector = embedder.embed_one(memory_text)?;

    // Format memory node label
    let memory_preview: String = memory_text.chars().take(40).collect();
    let memory_name = format!("{}: {}", args.category, memory_preview.trim());

    let node_id = engine.upsert_node(&memory_name, &args.category, memory_text, Some(&vector))?;

    // Optional link to related entity
    if let Some(ref rel_name) = args.relate_to {
        if let Some(target) = engine.get_node_by_name(rel_name) {
            engine.add_edge(node_id, target.id, "RELATES_TO", 0.90, true)?;
            println!("  Connected to existing entity '{}'", rel_name);
        }
    }

    engine.flush()?;

    let elapsed = start_time.elapsed();
    println!("=== Memory Stored Successfully ===");
    println!("  Node ID:     {}", node_id.as_u32());
    println!("  Label:       '{}'", memory_name);
    println!("  Category:    '{}'", args.category);
    println!("  Database:    '{}'", db_path.display());
    println!("  Time:        {:.2?}", elapsed);

    Ok(())
}
