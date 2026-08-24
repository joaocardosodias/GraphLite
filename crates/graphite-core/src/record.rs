use crate::id::{EdgeId, NodeId, StringId};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Flag indicating that a record is active (not soft-deleted).
pub const FLAG_ACTIVE: u32 = 1 << 0;

/// Flag indicating that an edge is directed (source -> target).
pub const FLAG_DIRECTED: u32 = 1 << 1;

/// Sentinel value indicating no vector embedding is associated with a node.
pub const NO_VECTOR_OFFSET: u64 = u64::MAX;

/// Binary size of a NodeRecord in bytes.
pub const NODE_RECORD_SIZE: usize = 32;

/// Binary size of an EdgeRecord in bytes.
pub const EDGE_RECORD_SIZE: usize = 32;

/// Fixed-size binary representation of a Node in the Graphite engine.
///
/// Designed with a strict 32-byte memory layout (`#[repr(C)]`) for zero-copy
/// serialization and direct memory-mapped file access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct NodeRecord {
    /// Unique identifier for this node.
    pub id: NodeId,
    /// Identifier in the String Table for the node's name/label.
    pub name_id: StringId,
    /// Identifier in the String Table for the entity type (e.g. "Person", "Project").
    pub type_id: StringId,
    /// Identifier in the String Table for the text description/summary.
    pub description_id: StringId,
    /// Byte offset in the binary file where the node's vector embedding is stored.
    pub vector_offset: u64,
    /// Bitflags controlling record state (e.g. `FLAG_ACTIVE`).
    pub flags: u32,
    /// Reserved padding to ensure strict 8-byte alignment and future extensibility.
    pub _reserved: u32,
}

impl NodeRecord {
    /// Size of the binary representation on disk (exactly 32 bytes).
    pub const BINARY_SIZE: usize = std::mem::size_of::<Self>();

    /// Creates a new active `NodeRecord`.
    pub fn new(
        id: NodeId,
        name_id: StringId,
        type_id: StringId,
        description_id: StringId,
        vector_offset: u64,
    ) -> Self {
        Self {
            id,
            name_id,
            type_id,
            description_id,
            vector_offset,
            flags: FLAG_ACTIVE,
            _reserved: 0,
        }
    }

    /// Returns `true` if the node is marked as active.
    #[inline]
    pub fn is_active(&self) -> bool {
        (self.flags & FLAG_ACTIVE) != 0
    }

    /// Marks the node as active or soft-deleted.
    #[inline]
    pub fn set_active(&mut self, active: bool) {
        if active {
            self.flags |= FLAG_ACTIVE;
        } else {
            self.flags &= !FLAG_ACTIVE;
        }
    }

    /// Returns `true` if this node has an associated vector embedding.
    #[inline]
    pub fn has_vector(&self) -> bool {
        self.vector_offset != NO_VECTOR_OFFSET
    }

    /// Serializes this `NodeRecord` into a fixed 32-byte array.
    pub fn to_bytes(&self) -> [u8; NODE_RECORD_SIZE] {
        let mut buf = [0u8; NODE_RECORD_SIZE];
        buf[0..4].copy_from_slice(&self.id.as_u32().to_le_bytes());
        buf[4..8].copy_from_slice(&self.name_id.as_u32().to_le_bytes());
        buf[8..12].copy_from_slice(&self.type_id.as_u32().to_le_bytes());
        buf[12..16].copy_from_slice(&self.description_id.as_u32().to_le_bytes());
        buf[16..24].copy_from_slice(&self.vector_offset.to_le_bytes());
        buf[24..28].copy_from_slice(&self.flags.to_le_bytes());
        buf[28..32].copy_from_slice(&self._reserved.to_le_bytes());
        buf
    }

    /// Deserializes a `NodeRecord` from a 32-byte array.
    pub fn from_bytes(buf: [u8; NODE_RECORD_SIZE]) -> Self {
        Self {
            id: NodeId::new(u32::from_le_bytes(buf[0..4].try_into().unwrap())),
            name_id: StringId::new(u32::from_le_bytes(buf[4..8].try_into().unwrap())),
            type_id: StringId::new(u32::from_le_bytes(buf[8..12].try_into().unwrap())),
            description_id: StringId::new(u32::from_le_bytes(buf[12..16].try_into().unwrap())),
            vector_offset: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            flags: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
            _reserved: u32::from_le_bytes(buf[28..32].try_into().unwrap()),
        }
    }
}

/// Fixed-size binary representation of an Edge (relationship) connecting two nodes.
///
/// Designed with a strict 32-byte memory layout (`#[repr(C)]`) for zero-copy
/// serialization and direct memory-mapped file access.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct EdgeRecord {
    /// Unique identifier for this edge.
    pub id: EdgeId,
    /// Origin node identifier.
    pub source: NodeId,
    /// Destination node identifier.
    pub target: NodeId,
    /// Identifier in the String Table for the relation label (e.g. "LEADS", "USES").
    pub relation_id: StringId,
    /// Semantic weight or confidence of the connection (typically 0.0 to 1.0).
    pub weight: f32,
    /// Bitflags controlling edge state (e.g. `FLAG_ACTIVE`, `FLAG_DIRECTED`).
    pub flags: u32,
    /// Reserved padding to ensure strict 8-byte alignment and future extensibility.
    pub _reserved: u64,
}

impl EdgeRecord {
    /// Size of the binary representation on disk (exactly 32 bytes).
    pub const BINARY_SIZE: usize = std::mem::size_of::<Self>();

    /// Creates a new active directed `EdgeRecord` with default weight 1.0.
    pub fn new(id: EdgeId, source: NodeId, target: NodeId, relation_id: StringId) -> Self {
        Self {
            id,
            source,
            target,
            relation_id,
            weight: 1.0,
            flags: FLAG_ACTIVE | FLAG_DIRECTED,
            _reserved: 0,
        }
    }

    /// Builder method to specify the edge weight.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Builder method to toggle whether the edge is directed.
    #[inline]
    pub fn with_directed(mut self, directed: bool) -> Self {
        if directed {
            self.flags |= FLAG_DIRECTED;
        } else {
            self.flags &= !FLAG_DIRECTED;
        }
        self
    }

    /// Returns `true` if the edge is marked as active.
    #[inline]
    pub fn is_active(&self) -> bool {
        (self.flags & FLAG_ACTIVE) != 0
    }

    /// Marks the edge as active or soft-deleted.
    #[inline]
    pub fn set_active(&mut self, active: bool) {
        if active {
            self.flags |= FLAG_ACTIVE;
        } else {
            self.flags &= !FLAG_ACTIVE;
        }
    }

    /// Serializes this `EdgeRecord` into a fixed 32-byte array.
    pub fn to_bytes(&self) -> [u8; EDGE_RECORD_SIZE] {
        let mut buf = [0u8; EDGE_RECORD_SIZE];
        buf[0..4].copy_from_slice(&self.id.as_u32().to_le_bytes());
        buf[4..8].copy_from_slice(&self.source.as_u32().to_le_bytes());
        buf[8..12].copy_from_slice(&self.target.as_u32().to_le_bytes());
        buf[12..16].copy_from_slice(&self.relation_id.as_u32().to_le_bytes());
        buf[16..20].copy_from_slice(&self.weight.to_le_bytes());
        buf[20..24].copy_from_slice(&self.flags.to_le_bytes());
        buf[24..32].copy_from_slice(&self._reserved.to_le_bytes());
        buf
    }

    /// Deserializes an `EdgeRecord` from a 32-byte array.
    pub fn from_bytes(buf: [u8; EDGE_RECORD_SIZE]) -> Self {
        Self {
            id: EdgeId::new(u32::from_le_bytes(buf[0..4].try_into().unwrap())),
            source: NodeId::new(u32::from_le_bytes(buf[4..8].try_into().unwrap())),
            target: NodeId::new(u32::from_le_bytes(buf[8..12].try_into().unwrap())),
            relation_id: StringId::new(u32::from_le_bytes(buf[12..16].try_into().unwrap())),
            weight: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
            flags: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            _reserved: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        }
    }

    /// Returns `true` if the edge is directed from `source` to `target`.
    #[inline]
    pub fn is_directed(&self) -> bool {
        (self.flags & FLAG_DIRECTED) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_record_memory_layout_and_size() {
        assert_eq!(NodeRecord::BINARY_SIZE, 32);
        assert_eq!(std::mem::size_of::<NodeRecord>(), 32);

        let mut node = NodeRecord::new(
            NodeId::new(1),
            StringId::new(10),
            StringId::new(20),
            StringId::new(30),
            1024,
        );

        assert!(node.is_active());
        assert!(node.has_vector());
        assert_eq!(node.vector_offset, 1024);

        node.set_active(false);
        assert!(!node.is_active());

        let node_no_vec = NodeRecord::new(
            NodeId::new(2),
            StringId::new(11),
            StringId::new(21),
            StringId::new(31),
            NO_VECTOR_OFFSET,
        );
        assert!(!node_no_vec.has_vector());
    }

    #[test]
    fn test_edge_record_memory_layout_and_size() {
        assert_eq!(EdgeRecord::BINARY_SIZE, 32);
        assert_eq!(std::mem::size_of::<EdgeRecord>(), 32);

        let edge = EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(10),
            NodeId::new(20),
            StringId::new(5),
        )
        .with_weight(0.85)
        .with_directed(true);

        assert!(edge.is_active());
        assert!(edge.is_directed());
        assert_eq!(edge.weight, 0.85);

        let undirected = edge.with_directed(false);
        assert!(!undirected.is_directed());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_records_serde() {
        let node = NodeRecord::new(
            NodeId::new(1),
            StringId::new(2),
            StringId::new(3),
            StringId::new(4),
            500,
        );
        let serialized = serde_json::to_string(&node).unwrap();
        let deserialized: NodeRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(node, deserialized);
    }
}
