//! Document Ingestion and GraphRAG Knowledge Base Builder with Incremental Hashing and Watch Mode.

pub mod chunker;

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use graphite::engine::config::GraphiteConfig;
use graphite::engine::instance::GraphiteEngine;
use graphite::storage::mmap_reader::MmapGraphReader;
use graphite::vector::distance::Metric;
use graphite::vector::quantization::Quantization;
use graphite::LocalEmbedder;

use self::chunker::{chunk_markdown_document, chunk_plain_document, ChunkConfig, DocumentChunk};
use crate::args::IngestArgs;

/// Computes a fast 64-bit content hash formatted as a hex string.
pub fn compute_file_hash(content: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

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
pub fn parse_document(
    path: &Path,
    file_hash: &str,
    config: &ChunkConfig,
) -> Result<Vec<DocumentChunk>> {
    let relative_path = path.to_string_lossy().to_string();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let content = match ext {
        "pdf" => {
            fs::read_to_string(path).unwrap_or_else(|_| format!("PDF Document: {}", relative_path))
        }
        _ => fs::read_to_string(path)?,
    };

    let chunks = match ext {
        "md" | "markdown" => chunk_markdown_document(&content, &relative_path, file_hash, config),
        "json" | "yaml" | "yml" | "csv" | "txt" => {
            chunk_plain_document(&content, &relative_path, file_hash, config)
        }
        _ => chunk_plain_document(&content, &relative_path, file_hash, config),
    };

    Ok(chunks)
}

/// Auto-links chunks across documents that mention shared entities or concepts.
pub fn link_related_chunks(chunks: &mut [DocumentChunk]) {
    let mut concept_to_chunks: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        let tokens = graphite::graph::bm25::Bm25Index::tokenize(&chunk.content);
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
                .with_quantization(quant);
        }
    }
    GraphiteConfig::new().with_dim(384)
}

/// Performs a single pass of incremental ingestion.
pub fn run_ingest_pass(
    db_path: &Path,
    args: &IngestArgs,
    embedder: &LocalEmbedder,
    is_watch_pass: bool,
) -> Result<bool> {
    let target_path = &args.path;
    if !target_path.exists() {
        bail!("Target path '{:?}' does not exist.", target_path);
    }

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
        if !is_watch_pass {
            println!(
                "No documents matching extensions {:?} found in '{}'.",
                extensions,
                target_path.display()
            );
        }
        return Ok(false);
    }

    let config = load_or_default_config(db_path)
        .with_direct_write(args.no_tmp)
        .with_auto_flush(false);
    let engine = GraphiteEngine::open_or_create(db_path, config)?;

    let chunk_config = ChunkConfig {
        target_chars: args.chunk_size * 4,
        overlap_chars: args.chunk_overlap * 4,
    };

    // Filter files by content hash to only process new or modified documents
    let mut files_to_process: Vec<(PathBuf, String)> = Vec::new();
    let mut unchanged_count = 0;

    for file in &files {
        let content_bytes = match fs::read(file) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let hash = compute_file_hash(&content_bytes);

        let file_basename = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let doc_root_name = format!("Doc: {}", file_basename);

        if !args.force {
            if let Some(doc_node) = engine.get_node_by_name(&doc_root_name) {
                if let Some(desc) = engine.resolve_string(doc_node.description_id) {
                    if desc.contains(&format!("Hash: {}", hash)) {
                        unchanged_count += 1;
                        continue;
                    }
                }
            }
        }

        files_to_process.push((file.clone(), hash));
    }

    if files_to_process.is_empty() {
        if !is_watch_pass {
            println!(
                "All {} document(s) are up to date with cached hashes (0 modifications).",
                unchanged_count
            );
        }
        return Ok(false);
    }

    let start_time = Instant::now();
    println!(
        "{} Processing {} modified/new document(s) ({} cached unchanged)...",
        if is_watch_pass { "[Watch]" } else { "" },
        files_to_process.len(),
        unchanged_count
    );

    let mut all_chunks: Vec<DocumentChunk> = files_to_process
        .par_iter()
        .filter_map(
            |(file, hash)| match parse_document(file, hash, &chunk_config) {
                Ok(chunks) => Some(chunks),
                Err(e) => {
                    eprintln!("Warning: Failed to parse '{:?}': {}", file, e);
                    None
                }
            },
        )
        .flatten()
        .collect();

    if all_chunks.is_empty() {
        return Ok(false);
    }

    link_related_chunks(&mut all_chunks);

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

    // Process nodes in vectorized batches of 64 for massive ONNX SIMD throughput
    let batch_size = 64;
    for chunk_batch in all_chunks.chunks(batch_size) {
        let texts: Vec<String> = chunk_batch
            .iter()
            .map(|c| format!("{} {}: {}", c.chunk_type, c.title, c.content))
            .collect();

        let vectors = embedder.embed_batch(&texts)?;

        for (chunk, vector) in chunk_batch.iter().zip(vectors.into_iter()) {
            pb.set_message(format!("{}: {}", chunk.chunk_type, chunk.title));

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
    }

    // Insert relational edges connecting the knowledge graph
    pb.set_message("Linking knowledge graph relations...");
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

    pb.set_message("Syncing binary storage to disk...");
    engine.flush()?;
    pb.finish_with_message("Sync complete");

    let elapsed = start_time.elapsed();
    println!(
        "{} Ingestion completed in {:.2?} | Nodes: {} (+{} new, {} updated), Edges: {} (+{})",
        if is_watch_pass { "[Watch]" } else { "===" },
        elapsed,
        engine.node_count(),
        new_entities_count,
        merged_entities_count,
        engine.edge_count(),
        edges_created_count
    );

    Ok(true)
}

/// Executes the end-to-end document ingestion command with optional watch loop.
pub fn execute_ingest(db_path: &Path, args: &IngestArgs) -> Result<()> {
    let target_path = &args.path;
    if !target_path.exists() {
        bail!("Target path '{:?}' does not exist.", target_path);
    }

    println!("Scanning documents in '{}'...", target_path.display());

    let emb_type = if db_path.exists() {
        if let Ok(reader) = graphite::storage::mmap_reader::MmapGraphReader::open(db_path) {
            graphite::vector::embedding::EmbeddingModelType::from_id(
                reader.header().embedding_model_id(),
                reader.header().vector_dim as usize,
            )
        } else {
            graphite::vector::embedding::EmbeddingModelType::AllMiniLML6V2
        }
    } else {
        graphite::vector::embedding::EmbeddingModelType::AllMiniLML6V2
    };

    let embedder = LocalEmbedder::from_model_type(emb_type).with_context(|| {
        format!(
            "Failed to initialize local ONNX embedding model ({})",
            emb_type.name()
        )
    })?;

    // 1. Initial ingestion pass
    run_ingest_pass(db_path, args, &embedder, false)?;

    // 2. Continuous Watch mode if requested
    if args.watch {
        println!(
            "\n[Watch] Watching '{}' for changes every 1s... (Press Ctrl+C to exit)",
            target_path.display()
        );

        loop {
            sleep(Duration::from_millis(1000));
            if let Err(e) = run_ingest_pass(db_path, args, &embedder, true) {
                eprintln!("[Watch] Error during sync: {}", e);
            }
        }
    }

    Ok(())
}
