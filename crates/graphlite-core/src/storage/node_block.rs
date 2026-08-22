use crate::error::{GraphLiteError, Result};
use crate::id::NodeId;
use crate::record::{NodeRecord, NODE_RECORD_SIZE};

/// Serializes a slice of `NodeRecord`s into a contiguous binary block.
///
/// Binary Layout:
/// - `4 bytes` : `count` (u32)
/// - `4 bytes` : `_reserved` (padding for 8-byte alignment)
/// - `count * 32 bytes` : Contiguous array of 32-byte `NodeRecord`s
pub fn serialize_node_block(nodes: &[NodeRecord]) -> Vec<u8> {
    let count = nodes.len() as u32;
    let total_bytes = 8 + (nodes.len() * NODE_RECORD_SIZE);
    let mut buffer = Vec::with_capacity(total_bytes);

    buffer.extend_from_slice(&count.to_le_bytes());
    buffer.extend_from_slice(&[0u8; 4]); // 8-byte alignment padding

    for node in nodes {
        buffer.extend_from_slice(&node.to_bytes());
    }

    buffer
}

/// A zero-copy reader over a memory-mapped binary node records block.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyNodeBlock<'a> {
    count: usize,
    data_slice: &'a [u8],
}

impl<'a> ZeroCopyNodeBlock<'a> {
    /// Creates a `ZeroCopyNodeBlock` from a raw byte slice.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self {
                count: 0,
                data_slice: &[],
            });
        }

        if bytes.len() < 8 {
            return Err(GraphLiteError::CorruptedFormat(
                "Node block too short for header".to_string(),
            ));
        }

        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let expected_payload = count * NODE_RECORD_SIZE;

        if bytes.len() < 8 + expected_payload {
            return Err(GraphLiteError::CorruptedFormat(format!(
                "Node block payload truncated: expected {} bytes, got {}",
                8 + expected_payload,
                bytes.len()
            )));
        }

        Ok(Self {
            count,
            data_slice: &bytes[8..8 + expected_payload],
        })
    }

    /// Returns the number of node records stored in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the node block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Retrieves a `NodeRecord` by direct index in $O(1)$ time.
    pub fn get_by_index(&self, index: usize) -> Option<NodeRecord> {
        if index >= self.count {
            return None;
        }

        let start = index * NODE_RECORD_SIZE;
        let end = start + NODE_RECORD_SIZE;
        let slice = &self.data_slice[start..end];
        let bytes: [u8; NODE_RECORD_SIZE] = slice.try_into().ok()?;

        Some(NodeRecord::from_bytes(bytes))
    }

    /// Retrieves a `NodeRecord` matching the given `NodeId` via direct indexing or binary search.
    pub fn get_by_id(&self, id: NodeId) -> Option<NodeRecord> {
        let idx = id.as_u32() as usize;
        if let Some(record) = self.get_by_index(idx) {
            if record.id == id && record.is_active() {
                return Some(record);
            }
        }

        // Fallback linear scan if IDs are not contiguous
        for i in 0..self.count {
            if let Some(record) = self.get_by_index(i) {
                if record.id == id && record.is_active() {
                    return Some(record);
                }
            }
        }

        None
    }
}

/// Deserializes a binary node block slice into an owned vector of `NodeRecord`s.
pub fn deserialize_node_block(bytes: &[u8]) -> Result<Vec<NodeRecord>> {
    let viewer = ZeroCopyNodeBlock::from_bytes(bytes)?;
    let mut nodes = Vec::with_capacity(viewer.len());

    for i in 0..viewer.len() {
        if let Some(node) = viewer.get_by_index(i) {
            nodes.push(node);
        }
    }

    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::StringId;
    use crate::record::NO_VECTOR_OFFSET;

    #[test]
    fn test_node_block_serialization_and_zero_copy_lookup() {
        let node0 = NodeRecord::new(
            NodeId::new(0),
            StringId::new(10),
            StringId::new(1),
            StringId::INVALID,
            NO_VECTOR_OFFSET,
        );
        let node1 = NodeRecord::new(
            NodeId::new(1),
            StringId::new(20),
            StringId::new(1),
            StringId::INVALID,
            128,
        );

        let nodes = vec![node0, node1];
        let serialized = serialize_node_block(&nodes);

        let viewer = ZeroCopyNodeBlock::from_bytes(&serialized).unwrap();
        assert_eq!(viewer.len(), 2);

        let r0 = viewer.get_by_index(0).unwrap();
        assert_eq!(r0.id, NodeId::new(0));
        assert_eq!(r0.name_id, StringId::new(10));

        let r1 = viewer.get_by_id(NodeId::new(1)).unwrap();
        assert_eq!(r1.id, NodeId::new(1));
        assert_eq!(r1.vector_offset, 128);

        assert!(viewer.get_by_id(NodeId::new(99)).is_none());
    }

    #[test]
    fn test_node_block_deserialize_roundtrip() {
        let nodes = vec![
            NodeRecord::new(
                NodeId::new(0),
                StringId::new(1),
                StringId::new(2),
                StringId::INVALID,
                0,
            ),
            NodeRecord::new(
                NodeId::new(1),
                StringId::new(3),
                StringId::new(4),
                StringId::INVALID,
                384,
            ),
        ];

        let serialized = serialize_node_block(&nodes);
        let restored = deserialize_node_block(&serialized).unwrap();

        assert_eq!(nodes, restored);
    }
}
