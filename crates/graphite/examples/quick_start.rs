//! Quickstart Example for the high-level `graphite` crate.
//!
//! Run with:
//! ```bash
//! cargo run --example quick_start -p graphite
//! ```

use graphite::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("                 Graphite: Quickstart Demo                  ");
    println!("============================================================\n");

    // 1. Initialize an in-memory database with 4-dimensional vectors
    let config = GraphiteConfig::new().with_dim(4).with_max_tokens(400);

    let db = Graphite::in_memory(config)?;

    // 2. Ingest entities with dense vectors
    let v_auth = [1.0, 0.0, 0.0, 0.0];
    let v_jwt = [0.95, 0.05, 0.0, 0.0];
    let v_db = [0.1, 0.9, 0.0, 0.0];

    let id_auth = db.upsert_node(
        "AuthService",
        "Module",
        "Handles user authentication and JWT sessions",
        Some(&v_auth),
    )?;
    let id_jwt = db.upsert_node(
        "JwtValidator",
        "Component",
        "Validates RS256 JWT tokens and claims",
        Some(&v_jwt),
    )?;
    let id_db = db.upsert_node(
        "UsersDB",
        "Database",
        "PostgreSQL database storing user profiles",
        Some(&v_db),
    )?;

    // 3. Connect entities with relationships
    db.add_edge(id_auth, id_jwt, "USES", 0.95, true)?;
    db.add_edge(id_auth, id_db, "QUERIES", 0.85, true)?;

    // 4. Query knowledge context
    let query_vector = [0.98, 0.02, 0.0, 0.0];
    let options = QueryOptions::default().with_max_tokens(300);
    let result = db.retrieve_context(&query_vector, Some(options))?;

    println!("Retrieved tokens: {}", result.token_count);
    println!("\nGenerated Markdown Prompt Context:\n{}", result.markdown);

    Ok(())
}
