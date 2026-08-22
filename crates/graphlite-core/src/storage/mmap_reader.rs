use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{GraphLiteError, Result};
use crate::id::{NodeId, StringId};
use crate::record::{EdgeRecord, NodeRecord};
use crate::storage::checksum::verify_file_integrity;
use crate::storage::csr_block::ZeroCopyCsrBlock;
use crate::storage::header::{GraphHeader, HEADER_SIZE};
use crate::storage::node_block::ZeroCopyNodeBlock;
use crate::storage::string_table::ZeroCopyStringTable;
use crate::storage::vector_block::ZeroCopyVectorBlock;
use crate::vector::quantization::QuantizedVector;

/// A high-performance, zero-copy reader over a memory-mapped `.graph` database file.
///
/// Maps the entire file into virtual memory in sub-millisecond time, allowing
/// direct traversal, vector search, and string lookups without allocating memory on the Heap.
pub struct MmapGraphReader {
    mmap: Mmap,
    header: GraphHeader,
}

impl std::fmt::Debug for MmapGraphReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapGraphReader")
            .field("header", &self.header)
            .field("file_size", &self.mmap.len())
            .finish()
    }
}

impl MmapGraphReader {
    /// Opens a `.graph` database file from disk with zero-copy memory mapping.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(GraphLiteError::CorruptedFormat(format!(
                "File too small for header: expected >= {} bytes, got {}",
                HEADER_SIZE,
                mmap.len()
            )));
        }

        let header = GraphHeader::from_bytes(&mmap[0..HEADER_SIZE])?;

        // Verify CRC32 checksum if present
        if header.checksum != 0 {
            verify_file_integrity(&mmap, header.checksum)?;
        }

        Ok(Self { mmap, header })
    }

    /// Returns a reference to the parsed 64-byte database header.
    #[inline]
    pub fn header(&self) -> &GraphHeader {
        &self.header
    }

    /// Total size in bytes of the mapped database file.
    #[inline]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Returns the raw memory-mapped byte slice of the entire database file.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns a zero-copy viewer over the String Table.
    pub fn string_table(&self) -> Result<ZeroCopyStringTable<'_>> {
        let offset = self.header.string_section_offset as usize;
        if offset > self.mmap.len() {
            return Err(GraphLiteError::CorruptedFormat(
                "Invalid string section offset".to_string(),
            ));
        }
        ZeroCopyStringTable::from_bytes(&self.mmap[offset..])
    }

    /// Returns a zero-copy viewer over the Node records block.
    pub fn nodes(&self) -> Result<ZeroCopyNodeBlock<'_>> {
        let offset = self.header.node_section_offset as usize;
        if offset > self.mmap.len() {
            return Err(GraphLiteError::CorruptedFormat(
                "Invalid node section offset".to_string(),
            ));
        }
        ZeroCopyNodeBlock::from_bytes(&self.mmap[offset..])
    }

    /// Returns a zero-copy viewer over the CSR Graph topology block.
    pub fn csr(&self) -> Result<ZeroCopyCsrBlock<'_>> {
        let offset = self.header.edge_section_offset as usize;
        if offset > self.mmap.len() {
            return Err(GraphLiteError::CorruptedFormat(
                "Invalid edge section offset".to_string(),
            ));
        }
        ZeroCopyCsrBlock::from_bytes(&self.mmap[offset..])
    }

    /// Returns a zero-copy viewer over the Quantized Vector block.
    pub fn vectors(&self) -> Result<ZeroCopyVectorBlock<'_>> {
        let offset = self.header.vector_section_offset as usize;
        if offset > self.mmap.len() {
            return Err(GraphLiteError::CorruptedFormat(
                "Invalid vector section offset".to_string(),
            ));
        }
        ZeroCopyVectorBlock::from_bytes(&self.mmap[offset..])
    }

    /// Resolves a `StringId` to a zero-copy borrowed `&str` in $O(1)$ time.
    pub fn resolve_string(&self, id: StringId) -> Option<&str> {
        let table = self.string_table().ok()?;
        table.get(id)
    }

    /// Retrieves a `NodeRecord` by its `NodeId` in $O(1)$ time.
    pub fn get_node(&self, id: NodeId) -> Option<NodeRecord> {
        let nodes = self.nodes().ok()?;
        nodes.get_by_id(id)
    }

    /// Retrieves all outgoing active edges for a given `NodeId`.
    pub fn get_out_edges(&self, id: NodeId) -> Vec<EdgeRecord> {
        match self.csr() {
            Ok(csr) => csr.out_edges(id),
            Err(_) => Vec::new(),
        }
    }

    /// Retrieves a `QuantizedVector` by direct index.
    pub fn get_vector(&self, index: usize) -> Option<QuantizedVector> {
        let vectors = self.vectors().ok()?;
        vectors.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    use crate::graph::csr::CsrGraph;
    use crate::id::{EdgeId, StringId};
    use crate::interner::StringInterner;
    use crate::record::{NO_VECTOR_OFFSET};
    use crate::storage::checksum::compute_file_checksum;
    use crate::storage::csr_block::serialize_csr_block;
    use crate::storage::header::FLAG_QUANTIZED_SQ8;
    use crate::storage::node_block::serialize_node_block;
    use crate::storage::string_table::serialize_string_table;
    use crate::storage::vector_block::serialize_quantized_vector_block;

    #[test]
    fn test_mmap_reader_end_to_end() {
        let mut temp_file = NamedTempFile::new().unwrap();

        // 1. Build components
        let mut interner = StringInterner::new();
        let s0 = interner.intern("Projeto Titan");
        let s1 = interner.intern("Ana Silva");
        let rel_lidera = interner.intern("LIDERADO_POR");
        let string_bytes = serialize_string_table(&interner);

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
            NO_VECTOR_OFFSET,
        );
        let node_bytes = serialize_node_block(&[node0, node1]);

        let edge0 = EdgeRecord::new(EdgeId::new(1), node0.id, node1.id, rel_lidera).with_weight(0.95);
        let csr = CsrGraph::new(vec![0, 1, 1], vec![edge0], 2);
        let edge_bytes = serialize_csr_block(&csr);

        let v0 = QuantizedVector {
            data: vec![10, 20, -30, 40],
            scale: 0.05,
            norm: 1.0, // módulo do vetor
        };
        let vector_bytes = serialize_quantized_vector_block(std::slice::from_ref(&v0), 4);

        // 2. Assemble file buffer
        let node_offset = HEADER_SIZE as u64;
        let edge_offset = node_offset + node_bytes.len() as u64;
        let vector_offset = edge_offset + edge_bytes.len() as u64;
        let string_offset = vector_offset + vector_bytes.len() as u64;

        let mut header = GraphHeader::new(4, 0, 1);
        header.flags = FLAG_QUANTIZED_SQ8;
        header.node_count = 2;
        header.edge_count = 1;
        header.string_bytes_len = string_bytes.len() as u32;
        header.node_section_offset = node_offset;
        header.edge_section_offset = edge_offset;
        header.vector_section_offset = vector_offset;
        header.string_section_offset = string_offset;

        let mut full_file = Vec::new();
        full_file.extend_from_slice(&header.to_bytes());
        full_file.extend_from_slice(&node_bytes);
        full_file.extend_from_slice(&edge_bytes);
        full_file.extend_from_slice(&vector_bytes);
        full_file.extend_from_slice(&string_bytes);

        // Compute checksum
        let checksum = compute_file_checksum(&full_file);
        full_file[56..60].copy_from_slice(&checksum.to_le_bytes());

        // Write to disk
        temp_file.write_all(&full_file).unwrap();
        temp_file.flush().unwrap();

        // 3. Open via MmapGraphReader
        let reader = MmapGraphReader::open(temp_file.path()).unwrap();

        assert_eq!(reader.header().node_count, 2);
        assert_eq!(reader.header().edge_count, 1);
        assert_eq!(reader.header().vector_dim, 4);

        // Test string resolution
        assert_eq!(reader.resolve_string(s0), Some("Projeto Titan"));
        assert_eq!(reader.resolve_string(s1), Some("Ana Silva"));

        // Test node lookup
        let n0 = reader.get_node(node0.id).unwrap();
        assert_eq!(n0.name_id, s0);
        assert_eq!(n0.vector_offset, 0);

        // Test edge lookup
        let out_edges = reader.get_out_edges(node0.id);
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].target, node1.id);
        assert_eq!(out_edges[0].weight, 0.95);

        // Test vector lookup
        let fetched_v0 = reader.get_vector(0).unwrap();
        assert_eq!(fetched_v0, v0);
    }
}
