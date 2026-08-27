use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use crate::error::Result;
use crate::graph::csr::CsrGraph;
use crate::interner::StringInterner;
use crate::record::NodeRecord;
use crate::storage::checksum::compute_file_checksum;
use crate::storage::csr_block::serialize_csr_block;
use crate::storage::header::{GraphHeader, FLAG_QUANTIZED_SQ8, HEADER_SIZE};
use crate::storage::node_block::serialize_node_block;
use crate::storage::string_table::serialize_string_table;
use crate::storage::vector_block::serialize_quantized_vector_block;
use crate::vector::quantization::QuantizedVector;

/// Serializes all database components into a complete in-memory `.graph` file buffer.
#[allow(clippy::too_many_arguments)]
pub fn serialize_database(
    nodes: &[NodeRecord],
    csr: &CsrGraph,
    vectors: &[QuantizedVector],
    interner: &StringInterner,
    vector_dim: usize,
    metric_type: u8,
    embedding_model_id: u8,
    reranker_model_id: u8,
) -> Vec<u8> {
    let node_bytes = serialize_node_block(nodes);
    let edge_bytes = serialize_csr_block(csr);
    let vector_bytes = serialize_quantized_vector_block(vectors, vector_dim);
    let string_bytes = serialize_string_table(interner);

    let node_offset = HEADER_SIZE as u64;
    let edge_offset = node_offset + node_bytes.len() as u64;
    let vector_offset = edge_offset + edge_bytes.len() as u64;
    let string_offset = vector_offset + vector_bytes.len() as u64;

    let mut header = GraphHeader::new(vector_dim as u16, metric_type, 1)
        .with_models(embedding_model_id, reranker_model_id);
    header.flags = FLAG_QUANTIZED_SQ8;
    header.node_count = nodes.len() as u32;
    header.edge_count = csr.edge_count() as u32;
    header.string_bytes_len = string_bytes.len() as u32;
    header.node_section_offset = node_offset;
    header.edge_section_offset = edge_offset;
    header.vector_section_offset = vector_offset;
    header.string_section_offset = string_offset;

    let total_size =
        HEADER_SIZE + node_bytes.len() + edge_bytes.len() + vector_bytes.len() + string_bytes.len();

    let mut full_file = Vec::with_capacity(total_size);
    full_file.extend_from_slice(&header.to_bytes());
    full_file.extend_from_slice(&node_bytes);
    full_file.extend_from_slice(&edge_bytes);
    full_file.extend_from_slice(&vector_bytes);
    full_file.extend_from_slice(&string_bytes);

    // Compute CRC32 checksum and write into header (bytes 56..60)
    let checksum = compute_file_checksum(&full_file);
    full_file[56..60].copy_from_slice(&checksum.to_le_bytes());

    full_file
}

/// Atomically writes a `.graph` database file to disk with crash-resilience.
///
/// Write Pipeline:
/// 1. Assembles binary blocks and calculates CRC32 checksum.
/// 2. Writes to a temporary staging file (`<path>.tmp.<pid>`).
/// 3. Flushes and executes `sync_all` (fsync) to guarantee physical persistence.
/// 4. Atomically renames the temporary file to the final destination.
#[allow(clippy::too_many_arguments)]
pub fn write_database_atomic<P: AsRef<Path>>(
    target_path: P,
    nodes: &[NodeRecord],
    csr: &CsrGraph,
    vectors: &[QuantizedVector],
    interner: &StringInterner,
    vector_dim: usize,
    metric_type: u8,
    embedding_model_id: u8,
    reranker_model_id: u8,
) -> Result<()> {
    let target = target_path.as_ref();
    let parent_dir = target.parent().unwrap_or_else(|| Path::new("."));

    // Ensure parent directory exists
    if !parent_dir.exists() {
        fs::create_dir_all(parent_dir)?;
    }

    let file_payload = serialize_database(
        nodes,
        csr,
        vectors,
        interner,
        vector_dim,
        metric_type,
        embedding_model_id,
        reranker_model_id,
    );

    // Unique temporary filename to prevent race conditions
    let pid = std::process::id();
    let tmp_path = parent_dir.join(format!(
        "{}.tmp.{}",
        target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("database"),
        pid
    ));

    // 1. Write to temporary file
    {
        let mut tmp_file = File::create(&tmp_path)?;
        tmp_file.write_all(&file_payload)?;
        tmp_file.flush()?;
        tmp_file.sync_all()?;
    }

    // 2. Atomic rename to destination path (POSIX atomic replacement)
    fs::rename(&tmp_path, target)?;

    Ok(())
}

/// Directly writes a `.graph` database file to disk without creating any temporary staging files.
#[allow(clippy::too_many_arguments)]
pub fn write_database_direct<P: AsRef<Path>>(
    target_path: P,
    nodes: &[NodeRecord],
    csr: &CsrGraph,
    vectors: &[QuantizedVector],
    interner: &StringInterner,
    vector_dim: usize,
    metric_type: u8,
    embedding_model_id: u8,
    reranker_model_id: u8,
) -> Result<()> {
    let target = target_path.as_ref();
    let parent_dir = target.parent().unwrap_or_else(|| Path::new("."));

    if !parent_dir.exists() {
        fs::create_dir_all(parent_dir)?;
    }

    let file_payload = serialize_database(
        nodes,
        csr,
        vectors,
        interner,
        vector_dim,
        metric_type,
        embedding_model_id,
        reranker_model_id,
    );

    let mut file = File::create(target)?;
    file.write_all(&file_payload)?;
    file.flush()?;
    file.sync_all()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    use crate::id::{EdgeId, NodeId, StringId};
    use crate::record::EdgeRecord;
    use crate::storage::mmap_reader::MmapGraphReader;

    #[test]
    fn test_atomic_writer_and_mmap_reader_integration() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_knowledge.graph");

        let mut interner = StringInterner::new();
        let s0 = interner.intern("Graphite Engine");
        let s1 = interner.intern("Fast RAG");
        let rel_enables = interner.intern("ENABLES");

        let node0 = NodeRecord::new(
            NodeId::new(s0.as_u32()),
            s0,
            StringId::new(1),
            StringId::INVALID,
            0,
        );
        let node1 = NodeRecord::new(
            NodeId::new(s1.as_u32()),
            s1,
            StringId::new(1),
            StringId::INVALID,
            384,
        );

        let edge0 =
            EdgeRecord::new(EdgeId::new(1), node0.id, node1.id, rel_enables).with_weight(0.99);
        let csr = CsrGraph::new(vec![0, 1, 1], vec![edge0], 2);

        let v0 = QuantizedVector {
            data: vec![5; 16],
            scale: 0.1,
            norm: 1.0, // módulo do vetor
        };
        let v1 = QuantizedVector {
            data: vec![-5; 16],
            scale: 0.1,
            norm: 1.0,
        };

        // Write atomically to disk
        write_database_atomic(
            &db_path,
            &[node0, node1],
            &csr,
            &[v0.clone(), v1.clone()],
            &interner,
            16,
            0, // Cosine metric
            0, // Embedding model ID
            1, // Reranker model ID
        )
        .unwrap();

        assert!(db_path.exists());

        // Reopen with MmapGraphReader and verify
        let reader = MmapGraphReader::open(&db_path).unwrap();
        assert_eq!(reader.header().node_count, 2);
        assert_eq!(reader.header().edge_count, 1);
        assert_eq!(reader.header().vector_dim, 16);
        assert_eq!(reader.header().embedding_model_id(), 0);
        assert_eq!(reader.header().reranker_model_id(), 1);

        assert_eq!(reader.resolve_string(s0), Some("Graphite Engine"));
        assert_eq!(reader.resolve_string(s1), Some("Fast RAG"));

        let out_edges = reader.get_out_edges(node0.id);
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].weight, 0.99);

        let fetched_v0 = reader.get_vector(0).unwrap();
        assert_eq!(fetched_v0, v0);
    }

    #[test]
    fn test_direct_writer_and_mmap_reader_integration() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_direct_knowledge.graph");

        let mut interner = StringInterner::new();
        let s0 = interner.intern("Direct Write Node");
        let node0 = NodeRecord::new(
            NodeId::new(s0.as_u32()),
            s0,
            StringId::new(1),
            StringId::INVALID,
            0,
        );
        let csr = CsrGraph::new(vec![0], vec![], 1);
        let v0 = QuantizedVector {
            data: vec![1; 16],
            scale: 0.1,
            norm: 1.0, // módulo do vetor
        };

        // Write directly to disk without .tmp
        write_database_direct(
            &db_path,
            &[node0],
            &csr,
            std::slice::from_ref(&v0),
            &interner,
            16,
            0,
            0,
            1,
        )
        .unwrap();

        assert!(db_path.exists());
        let reader = MmapGraphReader::open(&db_path).unwrap();
        assert_eq!(reader.header().node_count, 1);
        assert_eq!(reader.resolve_string(s0), Some("Direct Write Node"));
    }
}
