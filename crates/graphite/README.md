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

> **Graphite DB** is a high-performance, embedded database engine for GraphRAG and AI agent memory. It combines Compressed Sparse Row (CSR) graph topology, SIMD AVX2 vector distance kernels, BM25 inverted lexical indexing, and token-budgeted prompt synthesis into a single, zero-dependency `.graph` binary file with zero-copy memory mapping (`mmap`).

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
        .with_max_tokens(400);

    let db = Graphite::open_or_create("knowledge.graph", config)?;

    // 2. Ingest entities with dense embedding vectors
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

    // 4. Retrieve token-budgeted GraphRAG prompt context
    let query_vector = vec![0.05f32; 384];
    let options = QueryOptions::default().with_max_tokens(300);
    let result = db.retrieve_context(&query_vector, Some(options))?;

    println!("Retrieved tokens: {}", result.token_count);
    println!("\nGenerated Markdown Prompt Context:\n{}", result.markdown);

    Ok(())
}
```

---

## Local Embeddings with FastEmbed (Optional)

Generate 384-dimensional dense vectors locally on CPU with zero external API calls:

```rust
use graphite::prelude::*;
use graphite::LocalEmbedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize local ONNX embedder (MiniLM-L6-v2)
    let embedder = LocalEmbedder::new_minilm()?;
    let query_vector = embedder.embed_one("How does authentication work?")?;

    // 2. Query database
    let db = Graphite::open_or_create("knowledge.graph", GraphiteConfig::default())?;
    let result = db.retrieve_context(&query_vector, None)?;

    println!("{}", result.markdown);
    Ok(())
}
```

---

## Technical Highlights

- **Single-File Zero-Copy Storage:** Graph topology, Int8 scalar-quantized vectors (SQ8), BM25 inverted indices, and interned string tables packed into a single `.graph` file mapped directly via `memmap2`.
- **Sub-Millisecond Hybrid Retrieval:** Fuses SIMD AVX2 cosine distance with inverted BM25 lexical ranking via Reciprocal Rank Fusion (RRF with $k=60$).
- **Multi-Hop Relational Traversal:** High-speed BFS traversal over CSR buffers with $O(1)$-cycle bitset deduplication.
- **Token Budgeting & MMR:** Enforces exact token limits with BPE Tiktoken while pruning semantic redundancy via Maximal Marginal Relevance.

---

## Benchmarks

| Benchmark Metric | Scale | Latency / Performance | Speedup vs Baseline |
| :--- | :--- | :--- | :--- |
| **SIMD AVX2 Cosine (384-Dim)** | `MiniLM-L6` | **42 ns** (23.8M dist/s) | **7.02x faster** |
| **SQ8 Quantization Memory** | 100,000 Vectors | **37.4 MB RAM** | **-74.5% memory** |
| **CSR BFS Graph Traversal** | 5k Nodes / 15k Edges (2 Hops) | **4.8 µs** | 0 heap allocations |
| **End-to-End GraphRAG Pipeline** | Vector + BM25 + CSR + MMR | **280 µs** (0.28 ms) | Sub-millisecond |

---

## License

Dual-licensed under either [MIT](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-APACHE) at your option.
