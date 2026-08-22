//! # Autonomous AI Agent Long-Term Memory Example
//!
//! Demonstrates how an AI agent uses GraphLite to maintain a persistent,
//! disk-backed knowledge graph across multiple conversation sessions.
//!
//! To run this example:
//! ```bash
//! cargo run --example ai_agent_memory
//! ```

use tempfile::tempdir;

use graphlite_core::engine::config::GraphLiteConfig;
use graphlite_core::engine::entity_resolution::ResolutionConfig;
use graphlite_core::engine::instance::GraphLiteEngine;
use graphlite_core::engine::query::QueryOptions;
use graphlite_core::prompt::markdown::MarkdownStyle;
use graphlite_core::vector::distance::Metric;
use graphlite_core::vector::quantization::Quantization;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let memory_db_path = dir.path().join("agent_memory.graph");

    println!("============================================================");
    println!("       GraphLite: AI Agent Long-Term Memory Lifecycle       ");
    println!("============================================================\n");

    let config = GraphLiteConfig::new()
        .with_dim(4)
        .with_metric(Metric::Cosine)
        .with_quantization(Quantization::ScalarInt8)
        .with_max_tokens(400)
        .with_auto_flush(true);

    // =========================================================================
    // Session 1: Initial Knowledge Ingestion (User Preferences & Architecture)
    // =========================================================================
    println!("[Session 1] Ingesting user preferences and system architecture...");
    {
        let db = GraphLiteEngine::open_or_create(&memory_db_path, config.clone())?;

        // Semantic embedding vectors (normalized 4D simulation)
        let v_alice = [1.0, 0.0, 0.0, 0.0];
        let v_apollo = [0.95, 0.05, 0.0, 0.0];
        let v_axum = [0.85, 0.15, 0.0, 0.0];
        let v_error_rule = [0.1, 0.9, 0.0, 0.0];

        let id_alice = db.upsert_node(
            "Alice",
            "User",
            "Principal Engineer; prefers Rust code without unwrap in production",
            Some(&v_alice),
        )?;

        let id_apollo = db.upsert_node(
            "Project Apollo",
            "Project",
            "High-throughput microservices backend",
            Some(&v_apollo),
        )?;

        let id_axum = db.upsert_node(
            "Axum Framework",
            "Technology",
            "Async web framework built on top of Tokio and Tower",
            Some(&v_axum),
        )?;

        let id_rule = db.upsert_node(
            "Error Handling Policy",
            "Rule",
            "All endpoints must return structured Result<T, AppError> with custom error types",
            Some(&v_error_rule),
        )?;

        // Connect relational dependencies
        db.add_edge(id_alice, id_apollo, "LEADS", 0.95, true)?;
        db.add_edge(id_apollo, id_axum, "BUILT_WITH", 0.90, true)?;
        db.add_edge(id_apollo, id_rule, "MUST_COMPLY_WITH", 0.99, true)?;

        println!("  - Ingested {} nodes and {} edges to disk.\n", db.node_count(), db.edge_count());
    }

    // =========================================================================
    // Session 2: Entity Resolution (Merging Synonyms in Real Time)
    // =========================================================================
    println!("[Session 2] Agent learns new information with synonym phrasing...");
    {
        let db = GraphLiteEngine::open_or_create(&memory_db_path, config.clone())?;

        // Vector highly similar to "Project Apollo" (cosine similarity ~ 0.99)
        let v_apollo_synonym = [0.96, 0.04, 0.0, 0.0];

        let res = db.upsert_node_resolved(
            "Apollo Microservice Engine",
            "Project",
            "Deployed on Kubernetes with automatic horizontal pod scaling",
            &v_apollo_synonym,
            Some(ResolutionConfig {
                similarity_threshold: 0.90,
                require_matching_type: true,
                merge_descriptions: true,
            }),
        )?;

        println!("  - Entity Resolution Triggered: is_merged = {}", res.is_merged);
        println!("  - Node Count remained at {} (No duplicate nodes created!)\n", db.node_count());
    }

    // =========================================================================
    // Session 3: Token-Budgeted Retrieval for LLM System Prompt
    // =========================================================================
    println!("[Session 3] User asks a question in a new conversation session:");
    println!("  > User: 'How should I implement the new endpoint for Alice's service?'\n");
    {
        let db = GraphLiteEngine::open_or_create(&memory_db_path, config)?;

        // Query vector targeting Alice + Project Apollo
        let query_vector = [0.97, 0.03, 0.0, 0.0];

        let result = db.retrieve_context(
            &query_vector,
            Some(QueryOptions {
                top_k_seeds: 2,
                max_tokens: Some(300),
                markdown_style: MarkdownStyle::Hierarchical,
                max_depth: Some(2),
                min_score_threshold: Some(0.1),
                alpha: Some(0.6),
            }),
        )?;

        println!("=== Context Retrieved by GraphLite ({} tokens) ===", result.token_count);
        println!("{}\n", result.markdown);

        println!("=== Formatted LLM Prompt ===");
        println!("System: You are an autonomous coding assistant. Use the verified knowledge below:");
        println!("```markdown\n{}\n```", result.markdown.trim());
        println!("User: How should I implement the new endpoint for Alice's service?\n");
    }

    println!("============================================================");
    println!("            AI Agent Memory Workflow Completed!             ");
    println!("============================================================");

    Ok(())
}
