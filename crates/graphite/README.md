<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-white.png">
    <img src="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-black.png" alt="Graphite Logo" width="120" />
  </picture>
</p>

<h1 align="center">Graphite DB</h1>

<p align="center">
  <strong>Embedded Single-File GraphRAG Engine in Pure Rust</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/graphite-db"><img src="https://img.shields.io/crates/v/graphite-db.svg?color=black" alt="Crates.io" /></a>
  <a href="https://docs.rs/graphite-db"><img src="https://img.shields.io/docsrs/graphite-db?color=black" alt="Documentation" /></a>
  <a href="https://github.com/joaocardosodias/Graphite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-black.svg" alt="License" /></a>
</p>

Graphite DB is an embedded database engine for GraphRAG and AI agent memory. It enables relational knowledge graph storage, vector search, and BM25 full-text indexing directly within your Rust application using a single `.graph` file.

---

## Installation

Add `graphite-db` to your `Cargo.toml`:

```toml
[dependencies]
graphite-db = "0.1"
```

Or via Cargo CLI:
```bash
cargo add graphite-db
```

---

## Quickstart Example

```rust
use graphite::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize an in-memory or disk-backed database
    let config = GraphiteConfig::new()
        .with_dim(384)
        .with_threshold(0.75);

    let db = Graphite::open_or_create("knowledge.graph", config)?;

    // 2. Insert entities with dense embedding vectors
    let v_auth = vec![0.05f32; 384];
    let v_users = vec![0.048f32; 384];

    let id_auth = db.upsert_node(
        "AuthService",
        "Module",
        "Validates JSON Web Tokens and user sessions",
        Some(&v_auth),
    )?;

    let id_users = db.upsert_node(
        "UsersRepository",
        "Database",
        "PostgreSQL cluster storing credential hashes and user profiles",
        Some(&v_users),
    )?;

    // 3. Connect entities with weighted relationships
    db.add_edge(id_auth, id_users, "CONNECTS_TO", 0.95, true)?;
    db.flush()?;

    // 4. Retrieve GraphRAG prompt context with Auto-K cutoff
    let query_vector = vec![0.05f32; 384];
    let options = QueryOptions::default()
        .with_threshold(0.75)
        .with_auto_k(0.85);

    let result = db.retrieve_context(&query_vector, Some(options))?;

    println!("Generated Markdown Prompt Context:\n{}", result.markdown);
    Ok(())
}
```

---

## Local Embeddings (Optional)

Generate dense vectors locally on CPU with zero external API calls:

```rust
use graphite::prelude::*;
use graphite::LocalEmbedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize local embedder
    let embedder = LocalEmbedder::new_minilm()?;
    let query_vector = embedder.embed_one("How does authentication work?")?;

    // 2. Query database
    let db = Graphite::open_or_create("knowledge.graph", GraphiteConfig::default())?;
    let options = QueryOptions::default().with_threshold(0.75);
    let result = db.retrieve_context(&query_vector, Some(options))?;

    println!("{}", result.markdown);
    Ok(())
}
```

---

## License

Dual-licensed under either [MIT](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-APACHE) at your option.
