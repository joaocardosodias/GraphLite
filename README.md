<p align="center">
  <img src="assets/logo.png" alt="Graphite Logo" width="120" />
</p>

<h1 align="center">Graphite</h1>

<p align="center">
  <strong>Embedded Single-File GraphRAG & AI Agent Memory Database in Pure Rust</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg" alt="License" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Rust-1.80%2B-orange.svg?logo=rust" alt="Rust" /></a>
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status" />
  <img src="https://img.shields.io/badge/tests-99%20passed-success.svg" alt="Tests" />
</p>

> **Graphite** is an embedded, single-file Graph + Vector database engine written in pure Rust. 
> Designed as the **"SQLite of GraphRAG"** for autonomous AI agents, chatbots, and local-first document knowledge bases.

---

## ⚡ Key Highlights

- **Single-File Embedded Database (`.graphite`):** Fully self-contained single file with crash-resilient atomic writes, CRC32 checksums, and zero external daemon or database dependencies.
- **Universal Document Ingestion (`graphite ingest`):** Ingests Markdown, PDF, Plain Text, JSON, YAML, and CSV with structure-aware semantic chunking and automated cross-document relational linking.
- **Agent Long-Term Memory (`graphite remember`):** Real-time recording of facts, preferences, business rules, and tasks with automatic local embeddings.
- **Hybrid Retrieval (Vector + BM25 + Graph):** Fuses SIMD vector similarity with BM25 lexical ranking via Reciprocal Rank Fusion (RRF) and multi-hop BFS graph traversal.
- **MMR Semantic Deduplication & Diversity:** Built-in Maximal Marginal Relevance (MMR) and Jaccard token overlap pruning to suppress redundant text chunks and maximize information density.
- **Token-Budgeted Retrieval:** Built-in semantic prompt pruner (Tiktoken `cl100k_base` / `o200k_base` + heuristic) enforcing strict LLM context token limits.
- **SIMD-Accelerated Vector Search:** Hardware-accelerated dot product and cosine similarity (AVX2, FMA, NEON) with optional Int8 scalar quantization (SQ8) for a 4x memory reduction.
- **Native Model Context Protocol (MCP) Server:** Seamless integration with Claude Desktop, Cursor, Antigravity, and Windsurf via stdio JSON-RPC 2.0.

---

## 🏗️ GraphRAG Retrieval Pipeline

```
[User / Agent Query] 
        │
        ▼
[1. Hybrid Search (RRF)]  ──► Top-K Seeds via SIMD Cosine + BM25 Tokenizer
        │
        ▼
[2. CSR Multi-Hop BFS]    ──► Walk Knowledge Graph (Topology & Entity Relations)
        │
        ▼
[3. Hybrid Scoring]       ──► Combine Vector Similarity + Graph Proximity (Alpha Decay)
        │
        ▼
[4. MMR Deduplication]    ──► Eliminate Redundant Snippets / Maximize Marginal Diversity
        │
        ▼
[5. Token Budget Pruner]  ──► Prune Graph to Fit Exact Context Window (e.g. 500 tokens)
        │
        ▼
[6. Formatted Markdown]   ──► Clean, Structured Context Ready for LLM Injection
```

---

## 📂 Binary File Layout (`.graphite`)

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

## 🚀 Quickstart (CLI)

### 1. Installation

```bash
cargo install --path crates/graphite-cli
```

### 2. Ingest Documents into Knowledge Graph

```bash
# Ingest an entire folder of documentation, manuals, or policies
graphite -d knowledge.graphite ingest ./docs/

# Custom chunk size and overlap
graphite -d knowledge.graphite ingest ./docs/ --chunk-size 400 --chunk-overlap 50
```

### 3. Record Agent Memories & Preferences

```bash
# Store user preference
graphite -d knowledge.graphite remember \
  "User prefers concise answers in Portuguese and is an Enterprise customer" \
  --category "UserPreference"

# Store business fact or rule
graphite -d knowledge.graphite remember \
  "Refund requests are fully refundable within 7 days of subscription" \
  --category "BusinessRule"
```

### 4. Query with Token-Budgeted Context

```bash
# General query
graphite -d knowledge.graphite query -T "qual o prazo para reembolso?"

# Query filtered by entity type
graphite -d knowledge.graphite query -T "preferências do usuário" --type UserPreference
```

### 5. Launch REST API Server (Python / TypeScript / Web Clients)

```bash
# Start embedded HTTP server on port 8000
graphite -d knowledge.graphite serve --port 8000
```

```bash
# Query from any language via cURL:
curl -s -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "qual o prazo para estorno via PIX?", "max_tokens": 300}' | jq .
```

---

## 🦀 Rust API Example

Add Graphite to your `Cargo.toml`:

```toml
[dependencies]
graphite-core = { git = "https://github.com/joaocardosodias/Graphite" }
```

### Basic GraphRAG Retrieval

```rust
use graphite_core::engine::{GraphiteConfig, GraphiteEngine, QueryOptions};

fn main() -> anyhow::Result<()> {
    // 1. Initialize or open an existing database
    let config = GraphiteConfig::new()
        .with_dim(384)
        .with_max_tokens(500);

    let db = GraphiteEngine::open_or_create("knowledge.graphite", config)?;

    // 2. Ingest knowledge nodes with vectors
    let embedding_policy = vec![0.1f32; 384];
    let embedding_faq = vec![0.12f32; 384];

    let id_policy = db.upsert_node(
        "Refund Policy", 
        "Policy", 
        "Full refunds are granted within 7 days of purchase.", 
        Some(&embedding_policy)
    )?;

    let id_faq = db.upsert_node(
        "FAQ: Cancellations", 
        "FAQ", 
        "Subscriptions can be canceled anytime from the dashboard.", 
        Some(&embedding_faq)
    )?;

    // 3. Connect entities with relational edges
    db.add_edge(id_faq, id_policy, "RELATES_TO", 0.90, true)?;
    db.flush()?;

    // 4. Retrieve token-budgeted prompt context
    let query_vector = vec![0.11f32; 384];
    let result = db.retrieve_context(&query_vector, Some(QueryOptions {
        top_k_seeds: 2,
        max_tokens: Some(300),
        redundancy_threshold: Some(0.82),
        ..Default::default()
    }))?;

    println!("Context ({} tokens):\n{}", result.token_count, result.markdown);
    Ok(())
}
```

---

## 🤖 Model Context Protocol (MCP) Server

Graphite includes a native stdio MCP server (`graphite-mcp`) allowing AI assistants (Claude Desktop, Cursor, Antigravity) to retrieve and store knowledge autonomously.

### Install MCP Server:

```bash
cargo install --path crates/graphite-mcp
```

### Configuration (`~/.claude.json` or `~/.gemini/antigravity-cli/mcp/`):

```json
{
  "mcpServers": {
    "graphite": {
      "command": "graphite-mcp",
      "args": [
        "--db",
        "/path/to/your/knowledge.graphite"
      ]
    }
  }
}
```

### Available MCP Tools:

- `graphite_retrieve(query: string, max_tokens?: number, entity_type?: string)`: Retrieves structured context from the knowledge graph.
- `graphite_remember(name: string, type: string, description: string)`: Stores a fact or note with automatic semantic deduplication.
- `graphite_connect(source: string, target: string, relation: string, weight?: number)`: Connects relational dependencies between entities.

---

## 📊 Benchmarks

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
| **End-to-End `retrieve_context`** | Mmap + SIMD + BM25 + BFS + MMR + Pruner | ~1.4 ms |

---

## 📦 Workspace Structure

- [`crates/graphite-core`](crates/graphite-core/): Core embedded storage engine, SIMD vector store, CSR graph traversal, BM25, MMR, and prompt pruner.
- [`crates/graphite-cli`](crates/graphite-cli/): Document ingestion engine (`ingest`), agent memory (`remember`), and CLI interface (`graphite`).
- [`crates/graphite-mcp`](crates/graphite-mcp/): Model Context Protocol server for AI assistants and chatbots.

---

## 📄 License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
