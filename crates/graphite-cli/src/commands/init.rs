use anyhow::{bail, Result};
use std::fs;
use std::path::Path;

use graphite::engine::config::GraphiteConfig;
use graphite::engine::instance::GraphiteEngine;
use graphite::vector::distance::Metric;
use graphite::vector::embedding::{EmbeddingModelType, LocalEmbedder};
use graphite::vector::quantization::Quantization;
use graphite::vector::reranker::{LocalReranker, RerankerModelType};

use crate::args::{CliMetric, CliQuantization, InitArgs};
use crate::commands::wizard::run_interactive_wizard;

pub fn execute_init(db_path: &Path, args: &InitArgs) -> Result<()> {
    if args.interactive {
        let (final_path, config) = run_interactive_wizard(db_path)?;
        if final_path.exists() {
            let _ = fs::remove_file(&final_path);
        }
        let engine = GraphiteEngine::open_or_create(&final_path, config.clone())?;
        engine.flush()?;

        let emb_type = EmbeddingModelType::from_id(config.embedding_model_id, config.vector_dim);
        let rerank_type = RerankerModelType::from_id(config.reranker_model_id);

        println!("------------------------------------------------------------");
        println!("  Database initialized: {}", final_path.display());
        println!("  Embedding model:      {} ({}d)", emb_type.name(), config.vector_dim);
        println!("  Reranker model:       {}", rerank_type.name());
        println!("  Distance metric:      {:?}", config.metric);
        println!("  Quantization:         {:?}", config.quantization);
        println!("------------------------------------------------------------");
        return Ok(());
    }

    if db_path.exists() {
        if !args.force {
            bail!(
                "Database file '{:?}' already exists. Use '--force' (-f) to overwrite.",
                db_path
            );
        } else {
            fs::remove_file(db_path)?;
        }
    }

    // 1. Resolve Embedding Model & Dimensionality
    let emb_type = if let Some(ref name) = args.embedding_model {
        EmbeddingModelType::from_str_name(name)
            .unwrap_or_else(|| EmbeddingModelType::Custom(args.dim.unwrap_or(384)))
    } else if let Some(dim) = args.dim {
        EmbeddingModelType::Custom(dim)
    } else {
        EmbeddingModelType::AllMiniLML6V2
    };

    let dim = emb_type.dimension();

    // 2. Resolve Reranker Model
    let rerank_type = if let Some(ref name) = args.reranker_model {
        RerankerModelType::from_str_name(name).unwrap_or(RerankerModelType::BGERerankerBase)
    } else {
        RerankerModelType::BGERerankerBase
    };

    // 3. Pre-download models if requested
    if args.download {
        if let EmbeddingModelType::Custom(_) = emb_type {
            // custom dimension has no local download
        } else if emb_type.is_cached() {
            println!("  Embedding model '{}' is already cached.", emb_type.name());
        } else {
            println!("  Downloading Embedding Model: {}...", emb_type.name());
            LocalEmbedder::from_model_type(emb_type)?;
        }

        if rerank_type != RerankerModelType::None {
            if rerank_type.is_cached() {
                println!("  Reranker model '{}' is already cached.", rerank_type.name());
            } else {
                println!("  Downloading Reranker Model: {}...", rerank_type.name());
                LocalReranker::from_model_type(rerank_type)?;
            }
        }
    }

    let metric = match args.metric {
        CliMetric::Cosine => Metric::Cosine,
        CliMetric::DotProduct => Metric::DotProduct,
        CliMetric::Euclidean => Metric::Euclidean,
        CliMetric::Manhattan => Metric::Manhattan,
    };

    let quantization = match args.quantization {
        CliQuantization::None => Quantization::None,
        CliQuantization::ScalarInt8 => Quantization::ScalarInt8,
    };

    let config = GraphiteConfig::new()
        .with_dim(dim)
        .with_metric(metric)
        .with_quantization(quantization)
        .with_models(emb_type.id(), rerank_type.id())
        .with_max_tokens(args.max_tokens)
        .with_auto_flush(true);

    let engine = GraphiteEngine::open_or_create(db_path, config)?;
    engine.flush()?;

    println!("------------------------------------------------------------");
    println!("  Database initialized: {}", db_path.display());
    println!("  Embedding model:      {} ({}d)", emb_type.name(), dim);
    println!("  Reranker model:       {}", rerank_type.name());
    println!("  Distance metric:      {:?}", args.metric);
    println!("  Quantization:         {:?}", args.quantization);
    println!("------------------------------------------------------------");

    Ok(())
}
