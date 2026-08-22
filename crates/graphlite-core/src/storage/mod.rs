//! Binary file layout, header specification, zero-copy memory mapping, and persistence for `.graph` databases.

pub mod checksum;
pub mod header;
pub mod string_table;

pub use checksum::{compute_file_checksum, crc32, crc32_update, verify_file_integrity};
pub use header::{
    GraphHeader, FLAG_COMPRESSED, FLAG_DIRECTED, FLAG_QUANTIZED_SQ8, GRAPH_MAGIC, GRAPH_VERSION,
    HEADER_SIZE,
};
pub use string_table::{
    deserialize_string_table, serialize_string_table, ZeroCopyStringTable,
};
