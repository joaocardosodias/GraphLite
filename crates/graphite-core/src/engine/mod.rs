//! High-level Graphite database engine, concurrency control, and lifecycle manager.

pub mod config;
pub mod entity_resolution;
pub mod instance;
pub mod mutation;
pub mod query;

pub use config::GraphiteConfig;
pub use entity_resolution::{ResolutionConfig, ResolutionResult};
pub use instance::GraphiteEngine;
pub use query::{QueryOptions, QueryResult};
