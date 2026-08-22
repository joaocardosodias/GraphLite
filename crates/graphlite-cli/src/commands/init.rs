use std::fs;
use std::path::Path;
use anyhow::{bail, Result};

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliMetric, CliQuantization, InitArgs};

pub fn execute_init(db_path: &Path, args: &InitArgs) -> Result<()> {
    if db_path.exists() {
        if !args.force {
            bail!(
                "O arquivo de banco de dados '{:?}' já existe. Use a flag '--force' (-f) se desejar sobrescrever.",
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

    println!("✨ Banco de dados GraphLite inicializado com sucesso!");
    println!("   📁 Arquivo: {:?}", db_path);
    println!("   📐 Dimensão Vetorial: {} dimensões", args.dim);
    println!("   📏 Métrica: {:?}", args.metric);
    println!("   🗜️  Quantização: {:?}", args.quantization);
    println!("   🪙 Orçamento Padrão de Tokens: {}", args.max_tokens);

    Ok(())
}
