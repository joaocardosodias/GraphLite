//! Binary file layout, header specification, zero-copy memory mapping, and persistence for `.graph` databases.

pub mod header;

pub use header::{
    GraphHeader, FLAG_COMPRESSED, FLAG_DIRECTED, FLAG_QUANTIZED_SQ8, GRAPH_MAGIC, GRAPH_VERSION,
    HEADER_SIZE,
};
