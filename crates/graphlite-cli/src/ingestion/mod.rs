//! Document Ingestion and GraphRAG Knowledge Base Builder.

pub mod chunker;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::storage::mmap_reader::MmapGraphReader;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;
use graphlite_core::LocalEmbedder;

use self::chunker::{chunk_markdown_document, chunk_plain_document, ChunkConfig, DocumentChunk};
use crate::args::IngestArgs;

/// Recursively scans directory collecting relevant document files.
pub fn scan_documents<P: AsRef<Path>>(
    root: P,
    allowed_exts: &[&str],
    max_files: usize,
) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs_to_visit = vec![root.as_ref().to_path_buf()];

    while let Some(current_dir) = dirs_to_visit.pop() {
        if let Ok(entries) = fs::read_dir(&current_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                // Skip hidden folders, version control, build caches, and node_modules
                if file_name.starts_with('.')
                    || file_name == "target"
                    || file_name == "node_modules"
                    || file_name == "venv"
                    || file_name == ".venv"
                    || file_name == "dist"
                    || file_name == "build"
                {
                    continue;
                }

                if path.is_dir() {
                    dirs_to_visit.push(path);
                } else if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if allowed_exts.contains(&ext) {
                            files.push(path);
                            if files.len() >= max_files {
                                return files;
                            }
                        }
                    }
                }
            }
        }
    }

    files
}

/// Parses a document into structured semantic chunks based on file type.
pub fn parse_document(path: &Path, config: &ChunkConfig) -> Result<Vec<DocumentChunk>> {
    let relative_path = path.to_string_lossy().to_string();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let content = match ext {
        "pdf" => {
            // For PDF files, attempt basic extraction or read raw text
            fs::read_to_string(path).unwrap_or_else(|_| format!("PDF Document: {}", relative_path))
        }
        _ => fs::read_to_string(path)?,
    };

    let chunks = match ext {
        "md" | "markdown" => chunk_markdown_document(&content, &relative_path, config),
        "json" | "yaml" | "yml" | "csv" | "txt" => {
            chunk_plain_document(&content, &relative_path, config)
        }
        _ => chunk_plain_document(&content, &relative_path, config),
    };

    Ok(chunks)
}

/// Auto-links chunks across documents that mention shared entities or concepts.
pub fn link_related_chunks(chunks: &mut [DocumentChunk]) {
    let mut concept_to_chunks: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        let tokens = graphlite_core::graph::bm25::Bm25Index::tokenize(&chunk.content);
        let mut unique_concepts: HashSet<String> = HashSet::new();

        for token in tokens {
            if token.len() >= 4 {
                unique_concepts.insert(token);
            }
        }

        for concept in unique_concepts {
            concept_to_chunks.entry(concept).or_default().push(idx);
        }
    }

    // Connect chunks that share multiple key concepts
    let mut relations_to_add: Vec<(usize, String, String, f32)> = Vec::new();

    for (_concept, chunk_indices) in concept_to_chunks {
        if chunk_indices.len() > 1 && chunk_indices.len() <= 5 {
            for i in 0..chunk_indices.len() {
                for j in (i + 1)..chunk_indices.len() {
                    let idx_a = chunk_indices[i];
                    let idx_b = chunk_indices[j];

                    // Only connect if they belong to different documents or sections
                    if chunks[idx_a].file_path != chunks[idx_b].file_path {
                        relations_to_add.push((
                            idx_a,
                            chunks[idx_b].chunk_id.clone(),
                            "RELATES_TO".to_string(),
                            0.75,
                        ));
                    }
                }
            }
        }
    }

    for (src_idx, tgt_name, rel, weight) in relations_to_add {
        if !chunks[src_idx]
            .relations
            .iter()
            .any(|(t, _, _)| t == &tgt_name)
        {
            chunks[src_idx].relations.push((tgt_name, rel, weight));
        }
    }
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
    GraphLiteConfig::new().with_dim(384)
}

/// Executes the end-to-end document ingestion command.
pub fn execute_ingest(db_path: &Path, args: &IngestArgs) -> Result<()> {
    let target_path = &args.path;
    if !target_path.exists() {
        bail!("Target path '{:?}' does not exist.", target_path);
    }

    let start_time = Instant::now();
    println!("Scanning documents in '{}'...", target_path.display());

    let default_exts = vec!["md", "markdown", "txt", "pdf", "json", "yaml", "yml", "csv"];
    let extensions: Vec<&str> = if let Some(ref ext_str) = args.extensions {
        ext_str.split(',').map(|s| s.trim()).collect()
    } else {
        default_exts
    };

    let files = if target_path.is_file() {
        vec![target_path.clone()]
    } else {
        scan_documents(target_path, &extensions, args.max_files)
    };

    if files.is_empty() {
        println!(
            "No documents matching extensions {:?} found in '{}'.",
            extensions,
            target_path.display()
        );
        return Ok(());
    }

    println!(
        "Found {} document files. Parsing semantic chunks...",
        files.len()
    );

    let chunk_config = ChunkConfig {
        target_chars: args.chunk_size * 4,
        overlap_chars: args.chunk_overlap * 4,
    };

    let mut all_chunks = Vec::new();
    for file in &files {
        match parse_document(file, &chunk_config) {
            Ok(mut chunks) => all_chunks.append(&mut chunks),
            Err(e) => eprintln!("Warning: Failed to parse '{:?}': {}", file, e),
        }
    }

    if all_chunks.is_empty() {
        println!("No chunks extracted from documents.");
        return Ok(());
    }

    println!("Linking cross-document relations...");
    link_related_chunks(&mut all_chunks);

    println!(
        "Extracted {} knowledge chunks with relational edges. Initializing local ONNX embedding engine...",
        all_chunks.len()
    );

    let embedder = LocalEmbedder::new_minilm()
        .with_context(|| "Failed to initialize local ONNX embedding model")?;

    let config = load_or_default_config(db_path);
    let engine = GraphLiteEngine::open_or_create(db_path, config)?;

    let pb = ProgressBar::new(all_chunks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
            .expect("Valid template")
            .progress_chars("#>-"),
    );

    let mut name_to_id = HashMap::new();
    let mut new_entities_count = 0;
    let mut merged_entities_count = 0;
    let mut edges_created_count = 0;

    for chunk in &all_chunks {
        pb.set_message(format!("{}: {}", chunk.chunk_type, chunk.title));

        let embedding_text = format!("{} {}: {}", chunk.chunk_type, chunk.title, chunk.content);
        let vector = embedder.embed_one(&embedding_text)?;

        let initial_node_count = engine.node_count();
        let node_id = engine.upsert_node(
            &chunk.chunk_id,
            &chunk.chunk_type,
            &chunk.content,
            Some(&vector),
        )?;
        let final_node_count = engine.node_count();

        if final_node_count > initial_node_count {
            new_entities_count += 1;
        } else {
            merged_entities_count += 1;
        }

        name_to_id.insert(chunk.chunk_id.clone(), node_id);
        pb.inc(1);
    }

    // Insert relational edges connecting the knowledge graph
    for chunk in &all_chunks {
        if let Some(&src_id) = name_to_id.get(&chunk.chunk_id) {
            for (tgt_name, rel_label, weight) in &chunk.relations {
                if let Some(&tgt_id) = name_to_id.get(tgt_name) {
                    if engine
                        .add_edge(src_id, tgt_id, rel_label, *weight, true)
                        .is_ok()
                    {
                        edges_created_count += 1;
                    }
                }
            }
        }
    }

    pb.finish_with_message("Document embedding & graph construction complete");

    engine.flush()?;

    let elapsed = start_time.elapsed();
    println!("\n=== Document Ingestion Complete ===");
    println!("  Database:           '{}'", db_path.display());
    println!("  Files Ingested:     {}", files.len());
    println!("  New Chunks:         {}", new_entities_count);
    println!("  Updated Chunks:     {}", merged_entities_count);
    println!("  Graph Edges:        {}", edges_created_count);
    println!("  Total Graph Nodes:  {}", engine.node_count());
    println!("  Total Graph Edges:  {}", engine.edge_count());
    println!("  Elapsed Time:       {:.2?}", elapsed);

    Ok(())
}
