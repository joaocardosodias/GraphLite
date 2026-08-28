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

### 1. Ingest Documents or Direct Text
Automatically parse, chunk, embed, and index directories or raw text strings:

```bash
# Ingest files or folders
graphite ingest ./docs -d knowledge.graph --chunk-size 350 --chunk-overlap 40

# Ingest direct text
graphite ingest -t "User prefers concise answers in Portuguese." -d knowledge.graph --title "UserPreference"
```

### 2. Query Knowledge Context
Execute GraphRAG retrieval with automatic embeddings, reranking, and dynamic Auto-K cutoff:

```bash
graphite query "How does authentication work?" -d knowledge.graph --threshold 0.80 --drop-off 0.85 -k 5
```

### 3. Verify System & GPU Acceleration
Run diagnostics on CUDA acceleration, ONNX runtime, and models:

```bash
graphite doctor
```

### 4. Launch Embedded REST API Server
Start a lightweight, zero-dependency local HTTP server with CORS for Python, TypeScript, and web integrations:

```bash
graphite serve -d knowledge.graph --port 8080 --host 0.0.0.0
```

#### REST API Endpoints:
- `POST /v1/query` — Execute GraphRAG context retrieval
- `POST /v1/ingest` — Ingest document files, directories, or direct raw text
- `GET /health` — Check server status

---

## License

Dual-licensed under either [MIT](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-APACHE) at your option.
