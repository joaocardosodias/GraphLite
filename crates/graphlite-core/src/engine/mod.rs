//! High-level GraphLite database engine, concurrency control, and lifecycle manager.

pub mod config;
pub mod instance;

pub use config::GraphLiteConfig;
pub use instance::GraphLiteEngine;
