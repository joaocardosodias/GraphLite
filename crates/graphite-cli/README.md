<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-white.png">
    <img src="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-black.png" alt="Graphite Logo" width="120" />
  </picture>
</p>

<h1 align="center">Graphite CLI</h1>

<p align="center">
  <strong>Command-Line Tool & Local HTTP Server for Graphite GraphRAG</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/graphite-db-cli"><img src="https://img.shields.io/crates/v/graphite-db-cli.svg?color=black" alt="Crates.io" /></a>
  <a href="https://github.com/joaocardosodias/Graphite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-black.svg" alt="License" /></a>
  <a href="https://joaocardosodias.github.io/Graphite"><img src="https://img.shields.io/badge/docs-fumadocs-black.svg" alt="Documentation Site" /></a>
</p>

> **Graphite CLI** is the standalone command-line interface and embedded HTTP server for the Graphite GraphRAG engine. It allows you to ingest documents, execute hybrid retrieval queries, manage knowledge graphs, and serve local REST APIs directly from your terminal.

---

## Installation

### Via One-Line Installer (Linux / macOS):
```bash
curl -fsSL https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.sh | bash
```

### Via Windows PowerShell:
```powershell
irm https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.ps1 | iex
```

### Via Cargo (Rust):
```bash
cargo install graphite-db-cli
```

---

## Command Reference

### 1. Ingest Documents
Automatically parse, chunk, embed, and index directories containing Markdown, PDF, Plain Text, JSON, and CSV files:

```bash
graphite ingest ./docs -d knowledge.graphite --chunk-size 350 --chunk-overlap 40
```

### 2. Query Knowledge Context
Execute sub-millisecond GraphRAG retrieval with token budgeting:

```bash
graphite query "How does authentication work?" -d knowledge.graphite --max-tokens 400 --top-k 5
```

### 3. Record Facts & Agent Rules
Insert single entities, rules, or user preferences with real-time semantic deduplication:

```bash
graphite remember "User prefers concise answers in Portuguese." -d knowledge.graphite --category "UserPreference"
```

### 4. Inspect Database Health & Integrity
Verify CRC32 checksums, view block counts, and check memory allocation:

```bash
graphite inspect knowledge.graphite
```

### 5. Launch Embedded REST API Server
Start a lightweight, zero-dependency local HTTP server with CORS for Python, TypeScript, and web integrations:

```bash
graphite serve -d knowledge.graphite --port 8080 --host 0.0.0.0
```

#### REST API Endpoints:
- `POST /v1/query` — Execute GraphRAG context retrieval
- `POST /v1/insert` — Upsert entity nodes and embeddings
- `GET /health` — Check server status

---

## License

Dual-licensed under either [MIT](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-APACHE) at your option.
