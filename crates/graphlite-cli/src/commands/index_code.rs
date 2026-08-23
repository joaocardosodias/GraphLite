//! Command execution logic for `graphlite index-code`.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::entity_resolution::ResolutionConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::id::NodeId;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::embedding::LocalEmbedder;
use graphlite_core::vector::quantization::Quantization;

use crate::args::IndexCodeArgs;
use crate::indexer::{parse_file, scan_directory, ExtractedSymbol};

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
    GraphLiteConfig::new().with_dim(384)
}

pub fn execute_index_code(db_path: &Path, args: &IndexCodeArgs) -> Result<()> {
    let target_dir = &args.path;
    if !target_dir.exists() {
        bail!("Target directory '{:?}' does not exist.", target_dir);
    }

    let start_time = Instant::now();
    println!("Scanning codebase in '{}'...", target_dir.display());

    let default_exts = vec!["rs", "py", "ts", "tsx", "js", "jsx", "go"];
    let extensions: Vec<&str> = if let Some(ref ext_str) = args.extensions {
        ext_str.split(',').map(|s| s.trim()).collect()
    } else {
        default_exts
    };

    let max_files = args.max_files;
    let files = scan_directory(target_dir, &extensions, max_files);

    if files.is_empty() {
        println!(
            "No source code files matching extensions {:?} found in '{}'.",
            extensions,
            target_dir.display()
        );
        return Ok(());
    }

    println!(
        "Found {} source files. Parsing symbols and extracting AST...",
        files.len()
    );

    let mut all_symbols: Vec<ExtractedSymbol> = Vec::new();
    for file in &files {
        if let Ok(mut symbols) = parse_file(file) {
            all_symbols.append(&mut symbols);
        }
    }

    println!(
        "Extracted {} code symbols. Initializing local ONNX embedding engine...",
        all_symbols.len()
    );

    let embedder = LocalEmbedder::new_minilm()
        .with_context(|| "Failed to initialize embedded FastEmbed ONNX model")?;

    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let resolution_cfg = ResolutionConfig {
        similarity_threshold: 0.94,
        require_matching_type: true,
        merge_descriptions: true,
    };

    let pb = indicatif::ProgressBar::new(all_symbols.len() as u64);
    if let Ok(style) = indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
    {
        pb.set_style(style.progress_chars("█▓▒░ "));
    }

    let mut symbol_to_node_id: HashMap<String, NodeId> = HashMap::new();
    let mut indexed_count = 0;
    let mut merged_count = 0;

    for sym in &all_symbols {
        pb.set_message(format!("{}: {}", sym.symbol_type, sym.name));
        let text_to_embed = format!("{} {}: {}", sym.name, sym.symbol_type, sym.description);
        if let Ok(vec) = embedder.embed_one(&text_to_embed) {
            let res = engine.upsert_node_resolved(
                &sym.name,
                &sym.symbol_type,
                &sym.description,
                &vec,
                Some(resolution_cfg),
            )?;

            symbol_to_node_id.insert(sym.name.clone(), res.node_id);
            if res.is_merged {
                merged_count += 1;
            } else {
                indexed_count += 1;
            }
        }
        pb.inc(1);
    }

    pb.finish_with_message("Graph embedding complete");

    // 2. Connect relationships (METHOD_OF, IMPLEMENTS, USES)
    let mut created_edges = 0;
    for sym in &all_symbols {
        if let Some(src_id) = symbol_to_node_id.get(&sym.name) {
            for (target_name, relation, weight) in &sym.relations {
                if let Some(tgt_id) = symbol_to_node_id.get(target_name) {
                    if src_id != tgt_id {
                        let _ = engine.add_edge(*src_id, *tgt_id, relation, *weight, true);
                        created_edges += 1;
                    }
                }
            }
        }
    }

    engine.flush()?;
    let duration = start_time.elapsed();

    println!("\n=== Codebase Indexing Complete ===");
    println!("  Database:           '{}'", db_path.display());
    println!("  Files Scanned:      {}", files.len());
    println!("  New Entities:       {}", indexed_count);
    println!("  Merged Entities:    {}", merged_count);
    println!("  Edges Created:      {}", created_edges);
    println!("  Total Graph Nodes:  {}", engine.node_count());
    println!("  Total Graph Edges:  {}", engine.edge_count());
    println!("  Elapsed Time:       {:.2?}", duration);

    Ok(())
}
