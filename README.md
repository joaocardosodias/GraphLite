# GraphLite

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-102%20passed-success.svg)]()

> **GraphLite** is an embedded, single-file Graph + Vector database engine written in pure Rust. 
> Designed for local-first GraphRAG, autonomous AI agent memory, and low-latency architectural knowledge retrieval.

---

## Key Highlights

- **Single-File Binary Storage (`.graph`):** Fully self-contained database with crash-resilient atomic writes, CRC32 checksums, and zero external database dependencies.
- **Zero-Copy Memory-Mapped Access (`mmap`):** Direct memory layout mapping with near-zero deserialization overhead and sub-millisecond retrieval latency.
- **SIMD-Accelerated Vector Search:** Hardware-accelerated dot product and cosine similarity (AVX2, FMA, NEON) across dimensions (384, 768, 1536).
- **Scalar Quantization (Int8 / SQ8):** 4x memory reduction with asymmetric distance calculations and minimal recall degradation.
- **CSR Graph Topology:** High-throughput multi-hop BFS graph traversal using Compressed Sparse Row representation.
- **Token-Budgeted Retrieval:** Built-in semantic prompt pruner (Tiktoken `o200k_base` / `cl100k_base` + heuristic) enforcing strict LLM context token limits.
- **Real-Time Entity Resolution:** Automatic semantic deduplication and entity merging for synonyms with similarity thresholds > 0.92.
- **Native Model Context Protocol (MCP) Server:** Seamless integration with AI assistants (Claude Code, Antigravity, Cursor, Windsurf) over stdio JSON-RPC 2.0.

---

## Architectural Workflow

```
[User / Agent Query] 
        │
        ▼
[1. SIMD Vector Search] ──────► Find Top-K Seed Entities (Cosine / Euclidean)
        │
        ▼
[2. CSR Multi-Hop BFS]  ──────► Traverse Connected Subgraph (Weight & Depth Filters)
        │
        ▼
[3. Hybrid Scoring]     ──────► Combine Vector Similarity + Graph Proximity (Alpha Decay)
        │
        ▼
[4. Token Budget Prune] ──────► Prune Entities & Edges to Fit Exact Token Limits
        │
        ▼
[5. Formatted Markdown] ──────► Ready for Direct Injection into LLM System Prompt
```

---

## Binary File Layout (`.graph`)

```text
┌────────────────────────────────────────────────────────┐
│ Header (64 Bytes)                                      │
│  - Magic: "GRPH" | Version: 1 | Flags | Vector Dim     │
│  - Metric Type | CRC32 Checksum | Section Offsets      │
├────────────────────────────────────────────────────────┤
│ NodeBlock (Fixed-Size 32 Bytes per Node)               │
│  - ID, Name StringId, Type StringId, Vector Offset     │
├────────────────────────────────────────────────────────┤
│ CSR Edge Block (Compressed Sparse Row)                 │
│  - Row Offsets Table | Target Node IDs | Edge Records  │
├────────────────────────────────────────────────────────┤
│ Vector Block (Quantized SQ8 / Float32)                 │
│  - Scale Factors (f32) | Vector Magnitudes | i8 Arrays │
├────────────────────────────────────────────────────────┤
│ String Table Pool (Interned UTF-8 Strings)             │
│  - String Count | Offset Array | UTF-8 Bytes Payload   │
└────────────────────────────────────────────────────────┘
```

---

## Quickstart (Rust API)

Add GraphLite to your `Cargo.toml`:

```toml
[dependencies]
graphlite-core = { git = "https://github.com/joaocardosodias/GraphLite" }
```

### Basic Usage Example

```rust
use graphlite_core::engine::{GraphLiteConfig, GraphLiteEngine, QueryOptions};

fn main() -> anyhow::Result<()> {
    // 1. Initialize an in-memory or disk-backed database
    let config = GraphLiteConfig::new()
        .with_dim(384)
        .with_max_tokens(500);

    let db = GraphLiteEngine::open_or_create("knowledge.graph", config)?;

    // 2. Ingest entities with embeddings
    let embedding_titan = vec![0.1f32; 384];
    let embedding_ana = vec![0.12f32; 384];

    let id_titan = db.upsert_node(
        "Project Titan", 
        "Project", 
        "Core Generative AI Engine", 
        Some(&embedding_titan)
    )?;

    let id_ana = db.upsert_node(
        "Ana Silva", 
        "Person", 
        "Lead AI Architect", 
        Some(&embedding_ana)
    )?;

    // 3. Connect entities with relationships
    db.add_edge(id_ana, id_titan, "LEADS", 0.95, true)?;
    db.flush()?;

    // 4. Retrieve token-budgeted prompt context
    let query_vector = vec![0.11f32; 384];
    let result = db.retrieve_context(&query_vector, Some(QueryOptions {
        top_k_seeds: 2,
        max_tokens: Some(300),
        ..Default::default()
    }))?;

    println!("Retrieved Markdown Prompt ({} tokens):\n{}", result.token_count, result.markdown);
    Ok(())
}
```

---

## Command-Line Interface (`graphlite-cli`)

Install the CLI tool:

```bash
cargo install --path crates/graphlite-cli
```

### Common Commands

```bash
# 1. Initialize a new database
graphlite -d db.graph init -D 384 -m cosine -q scalar-int8 -f

# 2. Insert entities and relationships
graphlite -d db.graph insert-node -n "AuthService" -t "Struct" -D "JWT validation service" -V "0.1, 0.2, 0.3, 0.4"
graphlite -d db.graph insert-node -n "RedisCache" -t "Cache" -D "In-memory session store" -V "0.15, 0.22, 0.31, 0.41"
graphlite -d db.graph insert-edge -s "AuthService" -t "RedisCache" -r "STORES_SESSIONS" -w 0.95

# 3. Query the knowledge graph with token budgeting
graphlite -d db.graph query -V "0.12, 0.21, 0.32, 0.40" -k 2 -t 400 -f markdown

# 4. Inspect disk layout and CRC32 verification
graphlite -d db.graph inspect

# 5. Export entire graph to JSON
graphlite -d db.graph dump -f json
```

---

## Model Context Protocol (MCP) Integration

GraphLite includes a native MCP stdio server (`graphlite-mcp`) allowing AI assistants (Claude Code, Antigravity, Cursor) to autonomously read and write to the knowledge graph.

### Global Installation:

```bash
cargo install --path crates/graphlite-mcp
```

### Configuration (`~/.claude.json` or `~/.gemini/config/mcp_config.json`):

```json
{
  "mcpServers": {
    "graphlite": {
      "command": "graphlite-mcp",
      "args": [
        "--db",
        "/path/to/your/knowledge.graph"
      ]
    }
  }
}
```

### Exposed MCP Tools:

- `graphlite_retrieve(query: string, max_tokens?: number)`: Queries the graph with automatic background embedding generation.
- `graphlite_remember(name: string, type: string, description: string)`: Persists entities with real-time semantic deduplication.
- `graphlite_connect(source: string, target: string, relation: string, weight?: number)`: Connects relational dependencies.

---

## Benchmarks

Run the Criterion benchmark suite:

```bash
cargo bench
```

| Benchmark Operation | Scale | Latency / Throughput |
| :--- | :--- | :--- |
| **SIMD Cosine Similarity** | 384 Dimensions | ~45 ns / op |
| **SIMD Cosine Similarity** | 1536 Dimensions | ~180 ns / op |
| **SQ8 Quantized Dot Product** | 384 Dimensions | ~32 ns / op |
| **CSR Multi-Hop BFS** | 5,000 Nodes / 15k Edges (Depth 2) | ~8.4 µs |
| **End-to-End `retrieve_context`** | 200 Nodes (Mmap + Vector + BFS + Pruner) | ~1.4 ms |

---

## Workspace Structure

- [`crates/graphlite-core`](crates/graphlite-core/): Core embedded storage engine and graph traversal algorithms.
- [`crates/graphlite-cli`](crates/graphlite-cli/): Command-line interface (`graphlite`).
- [`crates/graphlite-mcp`](crates/graphlite-mcp/): Model Context Protocol server for AI coding assistants.

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
