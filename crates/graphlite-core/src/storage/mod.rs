//! Binary file layout, header specification, zero-copy memory mapping, and persistence for `.graph` databases.

pub mod atomic_writer;
pub mod checksum;
pub mod csr_block;
pub mod header;
pub mod mmap_reader;
pub mod node_block;
pub mod string_table;
pub mod vector_block;

pub use atomic_writer::{serialize_database, write_database_atomic};
pub use checksum::{compute_file_checksum, crc32, crc32_update, verify_file_integrity};
pub use csr_block::{deserialize_csr_block, serialize_csr_block, ZeroCopyCsrBlock};
pub use header::{
    GraphHeader, FLAG_COMPRESSED, FLAG_DIRECTED, FLAG_QUANTIZED_SQ8, GRAPH_MAGIC, GRAPH_VERSION,
    HEADER_SIZE,
};
pub use mmap_reader::MmapGraphReader;
pub use node_block::{
    deserialize_node_block, serialize_node_block, ZeroCopyNodeBlock,
};
pub use string_table::{
    deserialize_string_table, serialize_string_table, ZeroCopyStringTable,
};
pub use vector_block::{
    deserialize_quantized_vector_block, serialize_quantized_vector_block, ZeroCopyVectorBlock,
};
