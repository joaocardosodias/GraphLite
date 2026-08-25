//! Quickstart Example for the high-level `graphite` crate.
//!
//! Run with:
//! ```bash
//! cargo run --example quick_start -p graphite
//! ```

use graphite::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Graphite in-memory engine...");
    let db = Graphite::in_memory()?;

    println!("Ingesting entities...");
    db.insert_node("AuthService", "Module", "Handles user authentication and JWT tokens", None)?;
    db.insert_node("UsersDB", "Database", "PostgreSQL database storing user profiles", None)?;
    db.connect("AuthService", "UsersDB", "CONNECTS_TO", 0.95)?;

    println!("Querying GraphRAG knowledge context...");
    let result = db.query("How does authentication connect to the database?")?;

    println!("Retrieved tokens: {}", result.token_count);
    println!("\nGenerated Markdown Prompt:\n{}", result.markdown);

    Ok(())
}
