<p align="center">
  <img src="assets/logo.png" alt="Graphite Logo" width="120" />
</p>

<h1 align="center">Graphite</h1>

<p align="center">
  <strong>The Embedded Single-File GraphRAG Engine in Pure Rust</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-black.svg" alt="License" /></a>
  <a href="https://crates.io/crates/graphite-core"><img src="https://img.shields.io/crates/v/graphite-core.svg?color=black" alt="Crates.io" /></a>
  <a href="https://github.com/joaocardosodias/Graphite/actions"><img src="https://img.shields.io/badge/CI-passing-black.svg" alt="CI Status" /></a>
  <a href="https://joaocardosodias.github.io/Graphite"><img src="https://img.shields.io/badge/docs-fumadocs-black.svg" alt="Documentation" /></a>
</p>

> **Graphite** combines Knowledge Graphs (CSR), SIMD Vector Search (AVX2), BM25 Lexical Indexing, and Cross-Encoder Reranking into a single, zero-dependency `.graphite` binary file with zero-copy memory-mapped virtual memory (`mmap`).

---

## ⚡ Quick Install

### Linux & macOS (One-line installer):
```bash
curl -fsSL https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.sh | bash
```

### Windows (PowerShell):
```powershell
irm https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.ps1 | iex
```

### Via Cargo (Rust):
```bash
cargo install graphite-cli
```

---

## 🌟 Key Highlights

- **Single-File Zero-Copy Storage (`.graphite`):** Compressed Sparse Row (CSR) graph topology, scalar-quantized vectors (SQ8), inverted lexical index (BM25), and string tables packed into a single binary file with CRC32 integrity verification and atomic safe rename commits.
- **Sub-Millisecond Hybrid Retrieval (RRF):** Fuses SIMD-accelerated 256-bit AVX2 cosine distance with inverted BM25 lexical ranking via Reciprocal Rank Fusion (RRF with $k=60$).
- **Multi-Hop Relational Graph Traversal:** Instantaneous BFS traversal over contiguous memory-mapped CSR buffers with $O(1)$-cycle visited node deduplication (`DenseNodeBitSet`).
- **Token-Budgeted Context Synthesis (MMR):** Enforces strict prompt token limits (via BPE Tiktoken `cl100k_base` and `o200k_base`) while eliminating redundant chunks via Maximal Marginal Relevance ($\lambda = 0.75$).
- **Universal Hierarchical Ingestion (`graphite ingest`):** Ingests Markdown, PDF, Plain Text, JSON, and CSV into structured `Document` $\to$ `Section` $\to$ `Chunk` hierarchies with automatic relationship synthesis.
- **Embedded REST API (`graphite serve`):** Zero-dependency HTTP server with CORS providing endpoints for Python, TypeScript, Go, and web clients.

---

## 🏗️ GraphRAG Architecture

```text
[User / Query Text]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 1: Candidate Generation (Hybrid Search & RRF)       │
│  ├─ SIMD AVX2 256-bit Cosine Vector Scan (Top-K Seeds)    │
│  ├─ Inverted Index BM25 Lexical Scoring (Top-K Seeds)     │
│  ├─ Seed Fusion via Reciprocal Rank Fusion (RRF k=60)     │
│  └─ Multi-Hop BFS Relational Traversal over CSR Graph     │
└─────────────────────────────┬─────────────────────────────┘
                              │
                    Top Candidates Subgraph
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 2: Prompt Synthesis & Token Budgeting               │
│  ├─ Maximal Marginal Relevance (MMR) Redundancy Pruning   │
│  ├─ Exact Token Budget Enforcement (BPE Tiktoken / O200k) │
│  └─ Structured Markdown Context Generation for LLMs       │
└───────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart CLI Guide

### 1. Initialize a Database
```bash
graphite init -d knowledge.graphite
```

### 2. Ingest Document Acquis
```bash
graphite ingest ./docs -d knowledge.graphite --chunk-size 350 --chunk-overlap 40
```

### 3. Query GraphRAG Context
```bash
graphite query "How does authentication work?" -d knowledge.graphite --max-tokens 400 --top-k 5
```

### 4. Record Agent Facts & Rules
```bash
graphite remember "User prefers concise answers in Portuguese." -d knowledge.graphite --category "UserPreference"
```

### 5. Launch Embedded REST API
```bash
graphite serve -d knowledge.graphite --port 8080 --host 0.0.0.0
```

---

## 🦀 Rust SDK Example

Add `graphite-core` to your `Cargo.toml`:

```toml
[dependencies]
graphite-core = "0.1"
```

```rust
use graphite_core::engine::{GraphiteConfig, GraphiteEngine, QueryOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize engine
    let config = GraphiteConfig::new()
        .with_dim(384)
        .with_max_tokens(500);

    let db = GraphiteEngine::open_or_create("knowledge.graphite", config)?;

    // 2. Ingest nodes with dense embeddings
    let v_titan = vec![0.05f32; 384];
    let v_ana = vec![0.048f32; 384];

    let id_titan = db.upsert_node("Project Titan", "Project", "Generative AI Core", Some(&v_titan))?;
    let id_ana = db.upsert_node("Ana Silva", "Person", "Lead Architect", Some(&v_ana))?;

    // 3. Create relational edge
    db.add_edge(id_ana, id_titan, "LEADS", 0.95, true)?;
    db.flush()?;

    // 4. Retrieve token-budgeted GraphRAG prompt context
    let query_vector = vec![0.05f32; 384];
    let result = db.retrieve_context(&query_vector, Some(QueryOptions::default().with_max_tokens(400)))?;

    println!("Context ({} tokens):\n{}", result.token_count, result.markdown);
    Ok(())
}
```

---

## 📊 Benchmarks

| Benchmark Metric | Scale | Latency / Performance | Speedup vs Baseline |
| :--- | :--- | :--- | :--- |
| **SIMD AVX2 Cosine (384-Dim)** | `MiniLM-L6` | **42 ns** (23.8M dist/s) | **7.02x faster** |
| **SQ8 Quantization Memory** | 100,000 Vectors | **37.4 MB RAM** | **-74.5% memory** |
| **CSR BFS Graph Traversal** | 5k Nodes / 15k Edges (2 Hops) | **4.8 µs** | 0 heap allocations |
| **End-to-End GraphRAG Pipeline** | Vector + BM25 + CSR + MMR | **280 µs** (0.28 ms) | Sub-millisecond |

---

## 📦 Workspace Layout

- [`crates/graphite-core`](crates/graphite-core/): Pure Rust embedded storage engine, zero-copy mmap reader, SIMD AVX2 kernels, SQ8 quantization, CSR graph topology, BM25, and MMR token budgeting.
- [`crates/graphite-cli`](crates/graphite-cli/): The `graphite` command-line executable, multi-format document chunker, and embedded REST API server.

---

## 📄 License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
