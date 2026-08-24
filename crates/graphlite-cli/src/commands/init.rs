use anyhow::{bail, Result};
use std::fs;
use std::path::Path;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliMetric, CliQuantization, InitArgs};

pub fn execute_init(db_path: &Path, args: &InitArgs) -> Result<()> {
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

    let config = GraphLiteConfig::new()
        .with_dim(args.dim)
        .with_metric(metric)
        .with_quantization(quantization)
        .with_max_tokens(args.max_tokens)
        .with_auto_flush(true);

    let engine = GraphLiteEngine::open_or_create(db_path, config)?;
    engine.flush()?;

    println!("=== GraphLite Database Initialized ===");
    println!("  Database File:        {:?}", db_path);
    println!("  Vector Dimension:     {} dimensions", args.dim);
    println!("  Distance Metric:      {:?}", args.metric);
    println!("  Quantization:         {:?}", args.quantization);
    println!("  Default Token Budget: {}", args.max_tokens);

    Ok(())
}
