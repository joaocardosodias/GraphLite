use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::time::Instant;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::engine::query::QueryOptions;
use graphlite_core::prompt::format_subgraph_triples;
use graphlite_core::prompt::json::to_json_payload;
use graphlite_core::prompt::markdown::MarkdownStyle;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

use crate::args::{CliOutputFormat, QueryArgs};

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
        bail!(
            "Database file '{:?}' not found. Initialize with 'graphlite init' first.",
            db_path
        );
    }

    let start_time = Instant::now();
    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let query_vector = if let Some(raw) = args.vector.as_deref() {
        parse_vector_arg(Some(raw))?
    } else if let Some(ref text) = args.query_text {
        let embedder = graphlite_core::LocalEmbedder::new_minilm()
            .with_context(|| "Failed to initialize local ONNX embedding model")?;
        Some(embedder.embed_one(text)?)
    } else {
        None
    };

    let seed_names: Option<Vec<String>> = args.seeds.as_ref().map(|s| {
        s.split(',')
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .collect()
    });

    let has_seeds = match &seed_names {
        Some(v) => !v.is_empty(),
        None => false,
    };

    if query_vector.is_none() && !has_seeds {
        bail!("Provide at least one query vector (-V / --vector), plain text query (-T / --text), or textual seed entities (-s / --seeds).");
    }

    let type_filter = args.entity_type.as_ref().map(|s| {
        s.split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect()
    });

    let options = QueryOptions {
        top_k_seeds: args.top_k,
        query_text: args.query_text.clone(),
        max_tokens: args.tokens,
        markdown_style: MarkdownStyle::Hierarchical,
        max_depth: args.depth,
        min_score_threshold: None,
        alpha: args.alpha,
        relative_drop_off: Some(0.60),
        redundancy_threshold: Some(0.82),
        type_filter,
    };

    let mut result = if let Some(ref v) = query_vector {
        engine.retrieve_context(v, Some(options))?
    } else if let Some(ref seeds) = seed_names {
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        engine.retrieve_context_by_seed_names(&seed_refs, Some(options))?
    } else {
        unreachable!()
    };

    if args.rerank {
        if let Some(ref q_text) = args.query_text {
            let reranker = graphlite_core::LocalReranker::new_bge_base()
                .with_context(|| "Failed to initialize local ONNX Cross-Encoder reranker")?;

            let candidate_docs: Vec<String> = result
                .scored_entities
                .iter()
                .map(|e| {
                    if let Some(rec) = e.node_record {
                        let name = engine.resolve_string(rec.name_id).unwrap_or_default();
                        let desc = engine
                            .resolve_string(rec.description_id)
                            .unwrap_or_default();
                        format!("{} - {}", name, desc)
                    } else {
                        String::new()
                    }
                })
                .collect();

            if !candidate_docs.is_empty() {
                let rerank_res = reranker.rerank(q_text, &candidate_docs)?;
                let mut reranked_entities = Vec::new();
                for r in rerank_res {
                    if r.index < result.scored_entities.len() {
                        let mut entity = result.scored_entities[r.index].clone();
                        entity.final_score = r.score;
                        reranked_entities.push(entity);
                    }
                }

                let interner = {
                    if let Ok(reader) = MmapGraphReader::open(db_path) {
                        reader
                            .string_table()
                            .map(|st| st.to_interner())
                            .unwrap_or_default()
                    } else {
                        graphlite_core::interner::StringInterner::new()
                    }
                };

                let connected_subgraph = graphlite_core::ConnectedSubgraph {
                    entities: reranked_entities,
                    edges: result.pruned_subgraph.edges.clone(),
                    seed_ids: Vec::new(),
                };

                let token_budget = args.tokens.unwrap_or(engine.config().default_max_tokens);
                let token_counter = graphlite_core::TiktokenCounter::cl100k();
                let pruned = graphlite_core::prune_subgraph_by_budget_mmr(
                    &connected_subgraph,
                    &interner,
                    token_budget,
                    &token_counter,
                    engine.config().mmr_lambda,
                );

                let format_config = graphlite_core::MarkdownFormatConfig {
                    header_title: "Retrieved Knowledge Context (Reranked)".to_string(),
                    include_scores: true,
                    include_edge_weights: true,
                    style: MarkdownStyle::Hierarchical,
                };
                let markdown = graphlite_core::format_pruned_subgraph_markdown(
                    &pruned,
                    &interner,
                    &format_config,
                );

                result = graphlite_core::QueryResult {
                    markdown,
                    token_count: pruned.total_tokens,
                    entities_count: pruned.entities.len(),
                    edges_count: pruned.edges.len(),
                    scored_entities: connected_subgraph.entities,
                    pruned_subgraph: pruned,
                };
            }
        }
    }

    let elapsed = start_time.elapsed();

    match args.format {
        CliOutputFormat::Markdown => {
            println!("{}", result.markdown);
        }
        CliOutputFormat::Json => {
            let interner = {
                if let Ok(reader) = MmapGraphReader::open(db_path) {
                    reader
                        .string_table()
                        .map(|st| st.to_interner())
                        .unwrap_or_default()
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
                    reader
                        .string_table()
                        .map(|st| st.to_interner())
                        .unwrap_or_default()
                } else {
                    graphlite_core::interner::StringInterner::new()
                }
            };
            let triples = format_subgraph_triples(&result.pruned_subgraph, &interner);
            println!("{}", triples.join("\n"));
        }
    }

    if verbose {
        eprintln!("\n--- GraphLite Query Metrics ---");
        eprintln!("Total Latency:     {:.2?}", elapsed);
        eprintln!("Tokens in Prompt:  {}", result.token_count);
        eprintln!("Retained Entities: {}", result.entities_count);
        eprintln!("Retained Edges:    {}", result.edges_count);
    }

    Ok(())
}
