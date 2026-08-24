use crate::error::{GraphiteError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Magic bytes identifying a valid Graphite binary file (`.graph`).
pub const GRAPH_MAGIC: [u8; 4] = *b"GRPH";

/// Current binary specification version.
pub const GRAPH_VERSION: u16 = 1;

/// Fixed binary size of the header in bytes (64-byte aligned).
pub const HEADER_SIZE: usize = 64;

// Feature Flags in `GraphHeader::flags`
pub const FLAG_QUANTIZED_SQ8: u16 = 1 << 0;
pub const FLAG_COMPRESSED: u16 = 1 << 1;
pub const FLAG_DIRECTED: u16 = 1 << 2;

/// Fixed 64-byte header at the beginning of every `.graph` database file.
///
/// Ensures zero-copy memory mapping and instantaneous validation upon file open.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphHeader {
    /// 0..4: Magic signature bytes (`b"GRPH"`).
    pub magic: [u8; 4],
    /// 4..6: File format specification version.
    pub version: u16,
    /// 6..8: Bitflags representing storage features (quantization, compression, etc.).
    pub flags: u16,
    /// 8..10: Embedding vector dimensionality (e.g. 384, 512, 1536).
    pub vector_dim: u16,
    /// 10..11: Distance metric enum value (0: Cosine, 1: DotProduct, 2: Euclidean, 3: Manhattan).
    pub metric_type: u8,
    /// 11..12: Quantization strategy (0: Float32, 1: SQ8 Int8).
    pub quant_type: u8,
    /// 12..16: Total count of active Node records in the file.
    pub node_count: u32,
    /// 16..20: Total count of active Edge records in the file.
    pub edge_count: u32,
    /// 20..24: Size in bytes of the serialized String Pool block.
    pub string_bytes_len: u32,
    /// 24..32: Byte offset where the Node records table starts.
    pub node_section_offset: u64,
    /// 32..40: Byte offset where the Edge records table starts.
    pub edge_section_offset: u64,
    /// 40..48: Byte offset where the Vector payload data starts.
    pub vector_section_offset: u64,
    /// 48..56: Byte offset where the String Interner pool starts.
    pub string_section_offset: u64,
    /// 56..60: CRC32 checksum computed over the rest of the database file.
    pub checksum: u32,
    /// 60..64: Reserved for future extensions, padded to ensure exact 64-byte alignment.
    pub _reserved: [u8; 4],
}

impl GraphHeader {
    /// Creates a new `GraphHeader` initialized with default section offsets.
    pub fn new(vector_dim: u16, metric_type: u8, quant_type: u8) -> Self {
        Self {
            magic: GRAPH_MAGIC,
            version: GRAPH_VERSION,
            flags: 0,
            vector_dim,
            metric_type,
            quant_type,
            node_count: 0,
            edge_count: 0,
            string_bytes_len: 0,
            node_section_offset: HEADER_SIZE as u64,
            edge_section_offset: HEADER_SIZE as u64,
            vector_section_offset: HEADER_SIZE as u64,
            string_section_offset: HEADER_SIZE as u64,
            checksum: 0,
            _reserved: [0; 4],
        }
    }

    /// Returns `true` if this database stores 8-bit Scalarly Quantized vectors (SQ8).
    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.quant_type == 1 || (self.flags & FLAG_QUANTIZED_SQ8) != 0
    }

    /// Serializes this header into a fixed 64-byte array in little-endian order.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..10].copy_from_slice(&self.vector_dim.to_le_bytes());
        buf[10] = self.metric_type;
        buf[11] = self.quant_type;
        buf[12..16].copy_from_slice(&self.node_count.to_le_bytes());
        buf[16..20].copy_from_slice(&self.edge_count.to_le_bytes());
        buf[20..24].copy_from_slice(&self.string_bytes_len.to_le_bytes());
        buf[24..32].copy_from_slice(&self.node_section_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.edge_section_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.vector_section_offset.to_le_bytes());
        buf[48..56].copy_from_slice(&self.string_section_offset.to_le_bytes());
        buf[56..60].copy_from_slice(&self.checksum.to_le_bytes());
        buf[60..64].copy_from_slice(&self._reserved);

        buf
    }

    /// Deserializes a `GraphHeader` from a 64-byte slice and validates its integrity.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(GraphiteError::CorruptedFormat(format!(
                "Header buffer too short: expected {} bytes, got {}",
                HEADER_SIZE,
                bytes.len()
            )));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != GRAPH_MAGIC {
            return Err(GraphiteError::InvalidMagicBytes {
                expected: GRAPH_MAGIC,
                found: magic,
            });
        }

        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != GRAPH_VERSION {
            return Err(GraphiteError::UnsupportedVersion {
                expected: GRAPH_VERSION,
                found: version,
            });
        }

        let header = Self {
            magic,
            version,
            flags: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            vector_dim: u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            metric_type: bytes[10],
            quant_type: bytes[11],
            node_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            edge_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            string_bytes_len: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            node_section_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            edge_section_offset: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            vector_section_offset: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            string_section_offset: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
            _reserved: bytes[60..64].try_into().unwrap(),
        };

        header.validate()?;
        Ok(header)
    }

    /// Performs structural sanity checks on header offsets and dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.magic != GRAPH_MAGIC {
            return Err(GraphiteError::InvalidMagicBytes {
                expected: GRAPH_MAGIC,
                found: self.magic,
            });
        }

        if self.version != GRAPH_VERSION {
            return Err(GraphiteError::UnsupportedVersion {
                expected: GRAPH_VERSION,
                found: self.version,
            });
        }

        if self.node_section_offset < HEADER_SIZE as u64 {
            return Err(GraphiteError::CorruptedFormat(
                "Node section offset cannot be smaller than header size".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_exact_size_and_roundtrip() {
        assert_eq!(std::mem::size_of::<GraphHeader>(), 64);
        assert_eq!(HEADER_SIZE, 64);

        let mut header = GraphHeader::new(384, 0, 1);
        header.flags = FLAG_QUANTIZED_SQ8 | FLAG_DIRECTED;
        header.node_count = 1000;
        header.edge_count = 5000;
        header.string_bytes_len = 16384;
        header.node_section_offset = 64;
        header.edge_section_offset = 32064;
        header.vector_section_offset = 192064;
        header.string_section_offset = 576064;
        header.checksum = 0xDEADBEEF;

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 64);

        let deserialized = GraphHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, deserialized);
    }

    #[test]
    fn test_invalid_magic_bytes_detection() {
        let header = GraphHeader::new(384, 0, 0);
        let mut bytes = header.to_bytes();
        bytes[0] = b'X'; // Corrupt magic bytes

        let err = GraphHeader::from_bytes(&bytes).unwrap_err();
        match err {
            GraphiteError::InvalidMagicBytes { expected, found } => {
                assert_eq!(expected, GRAPH_MAGIC);
                assert_eq!(found, [b'X', b'R', b'P', b'H']);
            }
            _ => panic!("Expected InvalidMagicBytes error"),
        }
    }

    #[test]
    fn test_unsupported_version_detection() {
        let mut header = GraphHeader::new(384, 0, 0);
        header.version = 99; // Future unsupported version
        let bytes = header.to_bytes();

        let err = GraphHeader::from_bytes(&bytes).unwrap_err();
        match err {
            GraphiteError::UnsupportedVersion { expected, found } => {
                assert_eq!(expected, GRAPH_VERSION);
                assert_eq!(found, 99);
            }
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }
}
