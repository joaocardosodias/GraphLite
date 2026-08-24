use crate::error::{GraphiteError, Result};
use crate::graph::csr::CsrGraph;
use crate::id::NodeId;
use crate::record::{EdgeRecord, EDGE_RECORD_SIZE};

/// Serializes a `CsrGraph` into a contiguous zero-copy binary block.
///
/// Binary Layout:
/// - `4 bytes` : `node_count` (u32)
/// - `4 bytes` : `edge_count` (u32)
/// - `(node_count + 1) * 8 bytes` : CSR `offsets` array (u64 each)
/// - `edge_count * 32 bytes` : Contiguous array of 32-byte `EdgeRecord`s
pub fn serialize_csr_block(csr: &CsrGraph) -> Vec<u8> {
    let node_count = csr.node_count() as u32;
    let edge_count = csr.edge_count() as u32;

    let offsets_bytes_len = ((node_count as usize) + 1) * 8;
    let edges_bytes_len = (edge_count as usize) * EDGE_RECORD_SIZE;
    let total_bytes = 8 + offsets_bytes_len + edges_bytes_len;

    let mut buffer = Vec::with_capacity(total_bytes);

    // 1. Header
    buffer.extend_from_slice(&node_count.to_le_bytes());
    buffer.extend_from_slice(&edge_count.to_le_bytes());

    // 2. Offsets Array (u64)
    for &offset in csr.raw_offsets() {
        buffer.extend_from_slice(&offset.to_le_bytes());
    }

    // 3. Edges Array (32B each)
    for edge in csr.raw_edges() {
        buffer.extend_from_slice(&edge.to_bytes());
    }

    buffer
}

/// A zero-copy reader over a memory-mapped binary CSR graph topology block.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyCsrBlock<'a> {
    node_count: usize,
    edge_count: usize,
    offsets_slice: &'a [u8],
    edges_slice: &'a [u8],
}

impl<'a> ZeroCopyCsrBlock<'a> {
    /// Creates a `ZeroCopyCsrBlock` from a raw byte slice.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self {
                node_count: 0,
                edge_count: 0,
                offsets_slice: &[],
                edges_slice: &[],
            });
        }

        if bytes.len() < 8 {
            return Err(GraphiteError::CorruptedFormat(
                "CSR block too short for header".to_string(),
            ));
        }

        let node_count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let edge_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;

        let offsets_bytes_len = (node_count + 1) * 8;
        let edges_bytes_len = edge_count * EDGE_RECORD_SIZE;
        let expected_total = 8 + offsets_bytes_len + edges_bytes_len;

        if bytes.len() < expected_total {
            return Err(GraphiteError::CorruptedFormat(format!(
                "CSR block payload truncated: expected {} bytes, got {}",
                expected_total,
                bytes.len()
            )));
        }

        let offsets_slice = &bytes[8..8 + offsets_bytes_len];
        let edges_slice = &bytes[8 + offsets_bytes_len..expected_total];

        Ok(Self {
            node_count,
            edge_count,
            offsets_slice,
            edges_slice,
        })
    }

    /// Number of nodes indexed in this CSR topology.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Total number of directed edges stored in this CSR topology.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Returns `true` if the CSR topology contains no edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// Retrieves an offset for a given node index in $O(1)$ time.
    #[inline]
    pub fn get_offset(&self, node_idx: usize) -> Option<u64> {
        if node_idx > self.node_count {
            return None;
        }
        let start = node_idx * 8;
        let bytes = &self.offsets_slice[start..start + 8];
        Some(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Returns all outgoing active edge records for a given `NodeId`.
    pub fn out_edges(&self, node: NodeId) -> Vec<EdgeRecord> {
        let idx = node.as_u32() as usize;
        if idx >= self.node_count {
            return Vec::new();
        }

        let start = match self.get_offset(idx) {
            Some(o) => o as usize,
            None => return Vec::new(),
        };
        let end = match self.get_offset(idx + 1) {
            Some(o) => o as usize,
            None => return Vec::new(),
        };

        if start >= end || end > self.edge_count {
            return Vec::new();
        }

        let mut edges = Vec::with_capacity(end - start);
        for i in start..end {
            let byte_start = i * EDGE_RECORD_SIZE;
            let byte_end = byte_start + EDGE_RECORD_SIZE;
            let slice = &self.edges_slice[byte_start..byte_end];
            let raw_bytes: [u8; EDGE_RECORD_SIZE] = slice.try_into().unwrap();
            let edge = EdgeRecord::from_bytes(raw_bytes);
            if edge.is_active() {
                edges.push(edge);
            }
        }

        edges
    }

    /// Converts this zero-copy block into an owned in-memory `CsrGraph`.
    pub fn to_csr_graph(&self) -> CsrGraph {
        let mut offsets = Vec::with_capacity(self.node_count + 1);
        for i in 0..=self.node_count {
            if let Some(o) = self.get_offset(i) {
                offsets.push(o);
            }
        }

        let mut edges = Vec::with_capacity(self.edge_count);
        for i in 0..self.edge_count {
            let byte_start = i * EDGE_RECORD_SIZE;
            let byte_end = byte_start + EDGE_RECORD_SIZE;
            let slice = &self.edges_slice[byte_start..byte_end];
            let raw_bytes: [u8; EDGE_RECORD_SIZE] = slice.try_into().unwrap();
            edges.push(EdgeRecord::from_bytes(raw_bytes));
        }

        CsrGraph::new(offsets, edges, self.node_count)
    }
}

/// Deserializes a binary CSR block slice into an owned `CsrGraph`.
pub fn deserialize_csr_block(bytes: &[u8]) -> Result<CsrGraph> {
    let viewer = ZeroCopyCsrBlock::from_bytes(bytes)?;
    Ok(viewer.to_csr_graph())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{EdgeId, StringId};

    #[test]
    fn test_csr_block_serialization_and_zero_copy_traversal() {
        let edge0 = EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(0),
            NodeId::new(1),
            StringId::new(10),
        )
        .with_weight(0.95);
        let edge1 = EdgeRecord::new(
            EdgeId::new(2),
            NodeId::new(0),
            NodeId::new(2),
            StringId::new(20),
        )
        .with_weight(0.85);
        let edge2 = EdgeRecord::new(
            EdgeId::new(3),
            NodeId::new(1),
            NodeId::new(2),
            StringId::new(30),
        )
        .with_weight(0.75);

        let offsets = vec![0, 2, 3, 3];
        let edges = vec![edge0, edge1, edge2];
        let csr = CsrGraph::new(offsets, edges, 3);

        let serialized = serialize_csr_block(&csr);
        assert!(!serialized.is_empty());

        let viewer = ZeroCopyCsrBlock::from_bytes(&serialized).unwrap();
        assert_eq!(viewer.node_count(), 3);
        assert_eq!(viewer.edge_count(), 3);

        // Verify out edges of node 0
        let node0_edges = viewer.out_edges(NodeId::new(0));
        assert_eq!(node0_edges.len(), 2);
        assert_eq!(node0_edges[0].target, NodeId::new(1));
        assert_eq!(node0_edges[1].target, NodeId::new(2));

        // Verify out edges of node 1
        let node1_edges = viewer.out_edges(NodeId::new(1));
        assert_eq!(node1_edges.len(), 1);
        assert_eq!(node1_edges[0].target, NodeId::new(2));

        // Verify out edges of node 2 (leaf node)
        let node2_edges = viewer.out_edges(NodeId::new(2));
        assert!(node2_edges.is_empty());
    }

    #[test]
    fn test_csr_block_deserialize_roundtrip() {
        let offsets = vec![0, 1, 1];
        let edges = vec![EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(0),
            NodeId::new(1),
            StringId::new(5),
        )];

        let csr = CsrGraph::new(offsets, edges, 2);
        let serialized = serialize_csr_block(&csr);
        let restored = deserialize_csr_block(&serialized).unwrap();

        assert_eq!(csr.node_count(), restored.node_count());
        assert_eq!(csr.edge_count(), restored.edge_count());
        assert_eq!(csr.raw_offsets(), restored.raw_offsets());
        assert_eq!(csr.raw_edges(), restored.raw_edges());
    }
}
