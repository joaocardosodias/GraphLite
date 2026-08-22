use std::fs;
use std::path::Path;
use anyhow::{bail, Context, Result};

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::entity_resolution::ResolutionConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{InsertEdgeArgs, InsertNodeArgs};

/// Parses a vector argument from either a comma-separated float string or a JSON file.
fn parse_vector_arg(raw: Option<&str>) -> Result<Option<Vec<f32>>> {
    let s = match raw {
        Some(v) => v.trim(),
        None => return Ok(None),
    };

    if s.is_empty() {
        return Ok(None);
    }

    // Check if argument is a JSON file path
    if (s.ends_with(".json") || Path::new(s).exists()) && Path::new(s).is_file() {
        let content = fs::read_to_string(s)
            .with_context(|| format!("Falha ao ler arquivo de vetor: {}", s))?;
        let vector: Vec<f32> = serde_json::from_str(&content)
            .with_context(|| "Formato JSON inválido. Esperava um array de números (ex: [0.1, 0.2, ...])")?;
        return Ok(Some(vector));
    }

    // Parse as comma-separated or space-separated floats
    let mut vector = Vec::new();
    let cleaned = s.trim_start_matches('[').trim_end_matches(']');
    for part in cleaned.split([',', ' ']) {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let val = trimmed.parse::<f32>().with_context(|| {
                format!("Número inválido '{}' no vetor. Esperava floats separados por vírgula.", trimmed)
            })?;
            vector.push(val);
        }
    }

    Ok(Some(vector))
}

/// Reads the existing database configuration from header if file exists, or returns standard default.
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

pub fn execute_insert_node(db_path: &Path, args: &InsertNodeArgs) -> Result<()> {
    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let parsed_vector = parse_vector_arg(args.vector.as_deref())?;

    if let Some(ref v) = parsed_vector {
        if v.len() != engine.config().vector_dim {
            bail!(
                "Dimensão do vetor incompatível: o banco espera {} dimensões, mas foram fornecidas {}.",
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
            let id = engine.upsert_node(
                &args.name,
                &args.entity_type,
                &args.description,
                None,
            )?;
            graphlite_core::engine::ResolutionResult {
                node_id: id,
                is_merged: false,
                matched_existing_id: None,
            }
        };

        engine.flush()?;

        if result.is_merged {
            println!("🔗 Entidade fundida automaticamente com nó existente (ID: {})!", result.node_id);
            println!("   🏷️  Nome: '{}' -> Fundido com ID pré-existente", args.name);
        } else {
            println!("✅ Nó cadastrado com sucesso!");
            println!("   🆔 ID: {}", result.node_id);
            println!("   🏷️  Nome: '{}'", args.name);
            println!("   📂 Tipo: '{}'", args.entity_type);
        }
    } else {
        let node_id = engine.upsert_node(
            &args.name,
            &args.entity_type,
            &args.description,
            parsed_vector.as_deref(),
        )?;
        engine.flush()?;

        println!("✅ Nó cadastrado com sucesso!");
        println!("   🆔 ID: {}", node_id);
        println!("   🏷️  Nome: '{}'", args.name);
        println!("   📂 Tipo: '{}'", args.entity_type);
    }

    Ok(())
}

pub fn execute_insert_edge(db_path: &Path, args: &InsertEdgeArgs) -> Result<()> {
    if !db_path.exists() {
        bail!("O banco de dados '{:?}' não foi encontrado. Inicialize com 'graphlite init' primeiro.", db_path);
    }

    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let source_node = engine
        .get_node_by_name(&args.source)
        .with_context(|| format!("Nó de origem '{}' não foi encontrado no banco de dados.", args.source))?;

    let target_node = engine
        .get_node_by_name(&args.target)
        .with_context(|| format!("Nó de destino '{}' não foi encontrado no banco de dados.", args.target))?;

    let edge_id = engine.add_edge(
        source_node.id,
        target_node.id,
        &args.relation,
        args.weight,
        args.directed,
    )?;

    engine.flush()?;

    println!("✅ Aresta criada com sucesso!");
    println!("   🆔 ID da Aresta: {}", edge_id);
    println!("   🔗 Conexão: '{}' ──[{}]──► '{}'", args.source, args.relation, args.target);
    println!("   ⚖️  Peso: {:.2}", args.weight);
    println!("   🧭 Direcionada: {}", if args.directed { "Sim" } else { "Não (Bidirecional)" });

    Ok(())
}
