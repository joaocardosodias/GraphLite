use std::str;

use crate::error::{GraphiteError, Result};
use crate::id::StringId;
use crate::interner::StringInterner;

/// Serializes an in-memory `StringInterner` into a contiguous zero-copy binary format.
///
/// Binary Layout:
/// - `4 bytes` : `count` (u32, number of strings)
/// - `(count + 1) * 4 bytes` : `offsets` array (u32 each, pointing into the data block)
/// - Remaining bytes : Contiguous raw UTF-8 string payload
pub fn serialize_string_table(interner: &StringInterner) -> Vec<u8> {
    let count = interner.len() as u32;
    let mut total_string_bytes = 0usize;

    for i in 0..count {
        if let Some(s) = interner.resolve(StringId::new(i)) {
            total_string_bytes += s.len();
        }
    }

    // Allocate exact buffer size
    let header_and_offsets_size = 4 + ((count as usize) + 1) * 4;
    let mut buffer = Vec::with_capacity(header_and_offsets_size + total_string_bytes);

    // 1. Write string count
    buffer.extend_from_slice(&count.to_le_bytes());

    // 2. Compute and write offsets array
    let mut current_offset = 0u32;
    buffer.extend_from_slice(&current_offset.to_le_bytes()); // offset[0] = 0

    for i in 0..count {
        if let Some(s) = interner.resolve(StringId::new(i)) {
            current_offset += s.len() as u32;
        }
        buffer.extend_from_slice(&current_offset.to_le_bytes());
    }

    // 3. Write UTF-8 string bytes contiguously
    for i in 0..count {
        if let Some(s) = interner.resolve(StringId::new(i)) {
            buffer.extend_from_slice(s.as_bytes());
        }
    }

    buffer
}

/// A zero-copy string table viewer over a memory-mapped byte slice.
///
/// Resolves `StringId` to `&'a str` in $O(1)$ time with **zero memory allocation**.
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyStringTable<'a> {
    count: usize,
    offsets_slice: &'a [u8],
    data_slice: &'a [u8],
}

impl<'a> ZeroCopyStringTable<'a> {
    /// Creates a zero-copy reader from a binary slice.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self {
                count: 0,
                offsets_slice: &[],
                data_slice: &[],
            });
        }

        if bytes.len() < 4 {
            return Err(GraphiteError::CorruptedFormat(
                "String table too short for count header".to_string(),
            ));
        }

        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let offsets_byte_len = (count + 1) * 4;

        if bytes.len() < 4 + offsets_byte_len {
            return Err(GraphiteError::CorruptedFormat(
                "String table too short for offsets array".to_string(),
            ));
        }

        let offsets_slice = &bytes[4..4 + offsets_byte_len];
        let data_slice = &bytes[4 + offsets_byte_len..];

        // Validate final offset matches data_slice length
        let last_offset_bytes = &offsets_slice[count * 4..(count + 1) * 4];
        let total_data_len = u32::from_le_bytes(last_offset_bytes.try_into().unwrap()) as usize;

        if data_slice.len() < total_data_len {
            return Err(GraphiteError::CorruptedFormat(format!(
                "String table payload truncated: expected {} bytes, got {}",
                total_data_len,
                data_slice.len()
            )));
        }

        Ok(Self {
            count,
            offsets_slice,
            data_slice,
        })
    }

    /// Number of strings stored in this table.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the string table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Resolves a `StringId` to a zero-copy string slice in $O(1)$ time.
    pub fn get(&self, id: StringId) -> Option<&'a str> {
        let index = id.as_u32() as usize;
        if index >= self.count {
            return None;
        }

        let start_bytes = &self.offsets_slice[index * 4..(index + 1) * 4];
        let end_bytes = &self.offsets_slice[(index + 1) * 4..(index + 2) * 4];

        let start = u32::from_le_bytes(start_bytes.try_into().unwrap()) as usize;
        let end = u32::from_le_bytes(end_bytes.try_into().unwrap()) as usize;

        if start > end || end > self.data_slice.len() {
            return None;
        }

        let raw_bytes = &self.data_slice[start..end];
        str::from_utf8(raw_bytes).ok()
    }

    /// Converts this zero-copy string table into an owned in-memory `StringInterner`.
    pub fn to_interner(&self) -> StringInterner {
        let mut interner = StringInterner::with_capacity(self.count);
        for i in 0..self.count {
            if let Some(s) = self.get(StringId::new(i as u32)) {
                interner.intern(s);
            }
        }
        interner
    }
}

/// Deserializes a binary string table slice into a new in-memory `StringInterner`.
pub fn deserialize_string_table(bytes: &[u8]) -> Result<StringInterner> {
    let viewer = ZeroCopyStringTable::from_bytes(bytes)?;
    let mut interner = StringInterner::with_capacity(viewer.len());

    for i in 0..viewer.len() {
        if let Some(s) = viewer.get(StringId::new(i as u32)) {
            interner.intern(s);
        }
    }

    Ok(interner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_table_serialization_and_zero_copy_lookup() {
        let mut interner = StringInterner::new();
        let id0 = interner.intern("Projeto Titan");
        let id1 = interner.intern("Ana Silva");
        let id2 = interner.intern("Linguagem Rust 🚀");
        let id3 = interner.intern("São Paulo, Brasil");

        let serialized = serialize_string_table(&interner);
        assert!(!serialized.is_empty());

        let viewer = ZeroCopyStringTable::from_bytes(&serialized).unwrap();
        assert_eq!(viewer.len(), 4);

        assert_eq!(viewer.get(id0), Some("Projeto Titan"));
        assert_eq!(viewer.get(id1), Some("Ana Silva"));
        assert_eq!(viewer.get(id2), Some("Linguagem Rust 🚀"));
        assert_eq!(viewer.get(id3), Some("São Paulo, Brasil"));
        assert_eq!(viewer.get(StringId::new(99)), None);
    }

    #[test]
    fn test_deserialize_to_interner_roundtrip() {
        let mut interner = StringInterner::new();
        interner.intern("Entidade_A");
        interner.intern("Entidade_B");
        interner.intern("Entidade_C");

        let serialized = serialize_string_table(&interner);
        let restored = deserialize_string_table(&serialized).unwrap();

        assert_eq!(restored.len(), 3);
        assert_eq!(restored.resolve(StringId::new(0)), Some("Entidade_A"));
        assert_eq!(restored.resolve(StringId::new(1)), Some("Entidade_B"));
        assert_eq!(restored.resolve(StringId::new(2)), Some("Entidade_C"));
    }

    #[test]
    fn test_empty_string_table() {
        let interner = StringInterner::new();
        let serialized = serialize_string_table(&interner);

        let viewer = ZeroCopyStringTable::from_bytes(&serialized).unwrap();
        assert_eq!(viewer.len(), 0);
        assert!(viewer.is_empty());
        assert_eq!(viewer.get(StringId::new(0)), None);
    }
}
