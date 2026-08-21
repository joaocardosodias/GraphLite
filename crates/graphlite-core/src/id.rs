use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Strongly-typed identifier for a Node in the graph.
///
/// Uses a compact 32-bit unsigned integer (4 bytes), supporting up to 4.29 billion nodes
/// with minimal memory footprint and zero pointer overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Sentinel constant representing an invalid / null node id.
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != Self::INVALID.0
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

/// Strongly-typed identifier for an Edge in the graph.
///
/// Uses a compact 32-bit unsigned integer (4 bytes), supporting up to 4.29 billion edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct EdgeId(pub u32);

impl EdgeId {
    /// Sentinel constant representing an invalid / null edge id.
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != Self::INVALID.0
    }
}

impl From<u32> for EdgeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for EdgeId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeId({})", self.0)
    }
}

/// Strongly-typed identifier for an interned string in the String Table.
///
/// Reduces memory consumption by replacing repeated string instances with 4-byte IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct StringId(pub u32);

impl StringId {
    /// Sentinel constant representing an invalid / null string id.
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != Self::INVALID.0
    }
}

impl From<u32> for StringId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for StringId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl fmt::Display for StringId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StringId({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_conversions_and_validity() {
        let node_id = NodeId::new(42);
        assert_eq!(node_id.as_u32(), 42);
        assert_eq!(node_id.as_usize(), 42);
        assert!(node_id.is_valid());

        let from_u32: NodeId = 100u32.into();
        assert_eq!(from_u32, NodeId(100));

        let from_usize: NodeId = 200usize.into();
        assert_eq!(from_usize, NodeId(200));

        let invalid = NodeId::INVALID;
        assert!(!invalid.is_valid());
        assert_eq!(invalid.as_u32(), u32::MAX);

        assert_eq!(format!("{}", node_id), "NodeId(42)");
    }

    #[test]
    fn test_edge_id_conversions_and_validity() {
        let edge_id = EdgeId::new(7);
        assert_eq!(edge_id.as_u32(), 7);
        assert_eq!(edge_id.as_usize(), 7);
        assert!(edge_id.is_valid());

        let from_u32: EdgeId = 77u32.into();
        assert_eq!(from_u32, EdgeId(77));

        let invalid = EdgeId::INVALID;
        assert!(!invalid.is_valid());

        assert_eq!(format!("{}", edge_id), "EdgeId(7)");
    }

    #[test]
    fn test_string_id_conversions_and_validity() {
        let str_id = StringId::new(15);
        assert_eq!(str_id.as_u32(), 15);
        assert_eq!(str_id.as_usize(), 15);
        assert!(str_id.is_valid());

        let from_u32: StringId = 55u32.into();
        assert_eq!(from_u32, StringId(55));

        let invalid = StringId::INVALID;
        assert!(!invalid.is_valid());

        assert_eq!(format!("{}", str_id), "StringId(15)");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let node = NodeId(123);
        let serialized = serde_json::to_string(&node).unwrap();
        let deserialized: NodeId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(node, deserialized);
    }
}
