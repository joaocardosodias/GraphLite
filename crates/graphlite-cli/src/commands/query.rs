use std::fs;
use std::path::Path;
use std::time::Instant;
use anyhow::{bail, Context, Result};

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::engine::query::QueryOptions;
use graphlite_core::prompt::json::to_json_payload;
use graphlite_core::prompt::markdown::MarkdownStyle;
use graphlite_core::prompt::format_subgraph_triples;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliOutputFormat, QueryArgs};

/// Parses a vector argument from either a comma-separated float string or a JSON file.
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
            .with_context(|| format!("Falha ao ler arquivo de vetor: {}", s))?;
        let vector: Vec<f32> = serde_json::from_str(&content)
            .with_context(|| "Formato JSON inválido. Esperava um array de números (ex: [0.1, 0.2, ...])")?;
        return Ok(Some(vector));
    }

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

pub fn execute_query(db_path: &Path, args: &QueryArgs, verbose: bool) -> Result<()> {
    if !db_path.exists() {
        bail!("O banco de dados '{:?}' não foi encontrado. Use 'graphlite init' para criá-lo.", db_path);
    }

    let start_time = Instant::now();
    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let query_vector = parse_vector_arg(args.vector.as_deref())?;

    let seed_names: Option<Vec<String>> = args.seeds.as_ref().map(|s| {
        s.split(',')
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .collect()
    });

    if query_vector.is_none() && seed_names.as_ref().map_or(true, |v| v.is_empty()) {
        bail!("Informe ao menos um vetor de busca (-V / --vector) ou sementes textuais (-s / --seeds).");
    }

    let options = QueryOptions {
        top_k_seeds: args.top_k,
        max_tokens: args.tokens,
        markdown_style: MarkdownStyle::Hierarchical,
        max_depth: args.depth,
        min_score_threshold: None,
        alpha: args.alpha,
    };

    let result = if let Some(ref v) = query_vector {
        engine.retrieve_context(v, Some(options))?
    } else if let Some(ref seeds) = seed_names {
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        engine.retrieve_context_by_seed_names(&seed_refs, Some(options))?
    } else {
        unreachable!()
    };

    let elapsed = start_time.elapsed();

    // Format output
    match args.format {
        CliOutputFormat::Markdown => {
            println!("{}", result.markdown);
        }
        CliOutputFormat::Json => {
            let state = engine.get_node_by_name("").map(|_| ()); // test state
            let _ = state;
            // Build structured JSON payload
            let interner = {
                if let Ok(reader) = MmapGraphReader::open(db_path) {
                    reader.string_table().map(|st| st.to_interner()).unwrap_or_default()
                } else {
                    graphlite_core::interner::StringInterner::new()
                }
            };
            let payload = to_json_payload(&result.pruned_subgraph, &interner);
            println!("{}", serde_json::to_string_pretty(&payload)?);
        }
        CliOutputFormat::Triples => {
            let interner = {
                if let Ok(reader) = MmapGraphReader::open(db_path) {
                    reader.string_table().map(|st| st.to_interner()).unwrap_or_default()
                } else {
                    graphlite_core::interner::StringInterner::new()
                }
            };
            let triples = format_subgraph_triples(&result.pruned_subgraph, &interner);
            println!("{}", triples.join("\n"));
        }
    }

    if verbose {
        eprintln!("\n--- [Estatísticas da Consulta GraphLite] ---");
        eprintln!("⏱️  Latência Total: {:.2?}", elapsed);
        eprintln!("🪙 Tokens no Prompt: {}", result.token_count);
        eprintln!("🌐 Entidades Retidas: {}", result.entities_count);
        eprintln!("🔗 Arestas Retidas: {}", result.edges_count);
    }

    Ok(())
}
