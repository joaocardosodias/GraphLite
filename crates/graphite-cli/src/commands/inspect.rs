use anyhow::{bail, Result};
use serde::Serialize;
use std::fs;
use std::path::Path;

use graphite::storage::checksum::verify_file_integrity;
use graphite::storage::mmap_reader::MmapGraphReader;

use crate::args::InspectArgs;

#[derive(Serialize)]
struct DatabaseInspectionJson {
    file_path: String,
    file_size_bytes: u64,
    integrity_valid: bool,
    checksum_crc32: u32,
    magic: String,
    version: u16,
    vector_dim: u16,
    metric: String,
    quantization: String,
    node_count: u32,
    edge_count: u32,
    string_bytes_len: u32,
    sections: SectionOffsetsJson,
}

#[derive(Serialize)]
struct SectionOffsetsJson {
    node_section_offset: u64,
    edge_section_offset: u64,
    vector_section_offset: u64,
    string_section_offset: u64,
}

pub fn execute_inspect(db_path: &Path, args: &InspectArgs) -> Result<()> {
    if !db_path.exists() {
        bail!("Database file '{:?}' not found.", db_path);
    }

    let file_metadata = fs::metadata(db_path)?;
    let file_size = file_metadata.len();

    let reader = MmapGraphReader::open(db_path)?;
    let header = reader.header();

    let is_valid = verify_file_integrity(reader.as_bytes(), header.checksum).is_ok();

    let metric_str = match header.metric_type {
        0 => "Cosine Similarity",
        1 => "Dot Product",
        2 => "Euclidean Distance (L2)",
        3 => "Manhattan Distance (L1)",
        _ => "Unknown Metric",
    };

    let quant_str = if header.is_quantized() {
        "Scalar Quantization Int8 (SQ8)"
    } else {
        "None (Float32)"
    };

    let magic_str = String::from_utf8_lossy(&header.magic).to_string();

    if args.json {
        let inspect_data = DatabaseInspectionJson {
            file_path: db_path.to_string_lossy().to_string(),
            file_size_bytes: file_size,
            integrity_valid: is_valid,
            checksum_crc32: header.checksum,
            magic: magic_str,
            version: header.version,
            vector_dim: header.vector_dim,
            metric: metric_str.to_string(),
            quantization: quant_str.to_string(),
            node_count: header.node_count,
            edge_count: header.edge_count,
            string_bytes_len: header.string_bytes_len,
            sections: SectionOffsetsJson {
                node_section_offset: header.node_section_offset,
                edge_section_offset: header.edge_section_offset,
                vector_section_offset: header.vector_section_offset,
                string_section_offset: header.string_section_offset,
            },
        };
        println!("{}", serde_json::to_string_pretty(&inspect_data)?);
    } else {
        println!("============================================================");
        println!("                GRAPHITE DATABASE DIAGNOSTICS              ");
        println!("============================================================");
        println!("Database File:         {:?}", db_path);
        println!(
            "Disk File Size:        {} bytes ({:.2} KB)",
            file_size,
            file_size as f64 / 1024.0
        );
        println!(
            "CRC32 Integrity:       {} (Checksum: 0x{:08X})",
            if is_valid { "VALID" } else { "CORRUPTED" },
            header.checksum
        );
        let emb_type = graphite::vector::embedding::EmbeddingModelType::from_id(
            header.embedding_model_id(),
            header.vector_dim as usize,
        );
        let rerank_type = graphite::vector::reranker::RerankerModelType::from_id(
            header.reranker_model_id(),
        );

        println!("------------------------------------------------------------");
        println!(
            "Binary Format:         {} (Version: {})",
            magic_str, header.version
        );
        println!("Embedding Model:       {} ({}d)", emb_type.name(), header.vector_dim);
        println!("Reranker Model:        {}", rerank_type.name());
        println!("Distance Metric:       {}", metric_str);
        println!("Quantization Mode:     {}", quant_str);
        println!("------------------------------------------------------------");
        println!("Total Nodes:           {}", header.node_count);
        println!("Total Edges:           {}", header.edge_count);
        println!("String Pool Size:      {} bytes", header.string_bytes_len);
        println!("------------------------------------------------------------");
        println!("Section Offsets (Zero-Copy Mmap):");
        println!(
            "  NodeBlock:           offset 0x{:08X} ({} bytes)",
            header.node_section_offset,
            header.node_count as u64 * 32
        );
        println!(
            "  Edge/CSR Block:      offset 0x{:08X}",
            header.edge_section_offset
        );
        println!(
            "  VectorBlock:         offset 0x{:08X}",
            header.vector_section_offset
        );
        println!(
            "  StringTable:         offset 0x{:08X} ({} bytes)",
            header.string_section_offset, header.string_bytes_len
        );
        println!("============================================================");
    }

    Ok(())
}
