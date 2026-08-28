<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-white.png">
    <img src="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-black.png" alt="Graphite Logo" width="120" />
  </picture>
</p>

<h1 align="center">Graphite</h1>

<p align="center">
  <strong>Embedded Single-File GraphRAG & AI Memory Engine</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-black.svg" alt="License" /></a>
  <a href="https://pypi.org/project/graphite-database/"><img src="https://img.shields.io/pypi/v/graphite-database.svg?color=black" alt="PyPI" /></a>
  <a href="https://crates.io/crates/graphite-db"><img src="https://img.shields.io/crates/v/graphite-db.svg?color=black" alt="Crates.io" /></a>
  <a href="https://github.com/joaocardosodias/Graphite/actions"><img src="https://img.shields.io/badge/CI-passing-black.svg" alt="CI Status" /></a>
</p>

Graphite is an embedded, single-file GraphRAG and agent memory engine. It combines relational knowledge graphs, dense vector search, and full-text keyword search into a single portable `.graph` database file.

---

## Installation

### Python SDK
```bash
pip install graphite-database
```

### CLI Binary

#### Linux & macOS (One-line installer)
```bash
curl -fsSL https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.sh | bash
```

#### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/joaocardosodias/Graphite/main/install.ps1 | iex
```

#### Via Cargo (Rust CLI)
```bash
cargo install graphite-db-cli
```

### Rust Library
```bash
cargo add graphite-db
```

---

## Quickstart

### Python

```python
import graphite_db as graphite

# 1. Open or create database
db = graphite.open("knowledge.graph")

# 2. Ingest Markdown, PDF, text, or structured files
db.ingest("./docs/manual.md")

# 3. Query knowledge context (Auto-K and threshold enabled)
result = db.query(
    "How does authentication work?",
    threshold=0.80,
    top_k=5
)

print(result.markdown)

# 4. Ingest direct text strings
db.ingest(
    text="User prefers concise answers in Portuguese",
    title="UserPreferences"
)
```

---

### Command-Line Interface (CLI)

#### 1. Ingest Documents or Direct Text
Ingest single files, entire directories, or direct text strings into a `.graph` file:
```bash
# Ingest directory/files
graphite ingest ./docs -d knowledge.graph

# Ingest direct text
graphite ingest -t "System requires JWT RS256 authentication" -d knowledge.graph --title "SecurityPolicy"
```

#### 2. Query Knowledge
Query the database with automatic local embeddings, reranking, and dynamic Auto-K cutoff:
```bash
graphite query "How does authentication work?" -d knowledge.graph --threshold 0.80
```

#### 3. Start Local REST Server
Launch the embedded HTTP server:
```bash
graphite serve -d knowledge.graph --port 8080 --host 0.0.0.0
```

---

### REST API

When running `graphite serve`, interact via standard HTTP requests:

#### Query Context
```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "threshold": 0.80,
    "top_k_seeds": 5
  }'
```

#### Ingest Direct Text or Documents
```bash
# Ingest raw text
curl -X POST http://localhost:8080/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User prefers concise answers in Portuguese",
    "title": "UserPreferences"
  }'

# Ingest file/folder path
curl -X POST http://localhost:8080/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./docs"
  }'
```
```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./docs"
  }'
```

#### Health Check
```bash
curl http://localhost:8080/health
```

---

### Rust Library

```rust
use graphite::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize database
    let config = GraphiteConfig::new()
        .with_dim(384)
        .with_threshold(0.75);

    let db = Graphite::open_or_create("knowledge.graph", config)?;

    // 2. Insert entities with optional embeddings
    let v_auth = vec![0.05f32; 384];
    let v_db = vec![0.048f32; 384];

    let id_auth = db.upsert_node("AuthService", "Module", "Handles user authentication", Some(&v_auth))?;
    let id_db = db.upsert_node("UsersDB", "Database", "Stores user profiles and credentials", Some(&v_db))?;

    // 3. Connect entities with relationships
    db.add_edge(id_auth, id_db, "QUERIES", 0.95, true)?;
    db.flush()?;

    // 4. Retrieve context
    let query_vector = vec![0.05f32; 384];
    let options = QueryOptions::default().with_threshold(0.75).with_auto_k(0.85);
    let result = db.retrieve_context(&query_vector, Some(options))?;

    println!("Context:\n{}", result.markdown);
    Ok(())
}
```

---

## CLI Command Reference

| Command | Description | Example |
| :--- | :--- | :--- |
| `graphite init [path]` | Initialize a new `.graph` database with interactive setup | `graphite init knowledge.graph` |
| `graphite ingest [path] [-t <text>]` | Ingest Markdown, PDF, text, JSON, CSV, or direct text string | `graphite ingest ./docs -d knowledge.graph`<br>`graphite ingest -t "Raw text..." -d knowledge.graph` |
| `graphite query <text>` | Query knowledge context with reranking and Auto-K | `graphite query "auth flow" -d knowledge.graph --threshold 0.80` |
| `graphite serve` | Start the embedded HTTP REST API server | `graphite serve -d knowledge.graph --port 8080` |
| `graphite dump` | Export database nodes, edges, and statistics | `graphite dump -d knowledge.graph` |
| `graphite doctor` | Verify system dependencies, GPU acceleration, and environment | `graphite doctor` |

---

## Python API Reference

### Database Initialization
- `graphite.open(path="knowledge.graph", dim=384, metric="cosine", quantization="sq8", device="auto")`: Opens or creates a database file.
- `graphite.in_memory(dim=384, metric="cosine", quantization="sq8", device="auto")`: Creates an ephemeral in-memory database.

### Methods
- `db.ingest(source: str = None, text: str = None, title: str = "DirectInput") -> int`: Ingests files, directories, or direct text strings.
- `db.query(text: str, top_k: int = 10, threshold: float = None, relative_drop_off: float = 0.85, max_depth: int = None) -> QueryResult`: Queries the knowledge graph using plain text.
- `db.upsert_node(name: str, node_type: str, description: str, embedding: list[float] = None) -> int`: Manually inserts or updates a node.
- `db.add_edge(source_id: int, target_id: int, relation: str, weight: float = 1.0, bidirectional: bool = True)`: Adds a relationship between nodes.
- `db.retrieve_context(vector: list[float], query_text: str = None, top_k: int = 10, threshold: float = None, relative_drop_off: float = 0.85) -> QueryResult`: Low-level retrieval with raw query vectors.
- `db.flush()`: Persists all pending in-memory mutations to disk.
- `db.close()`: Flushes and releases all database locks.

---

## Query & Retrieval Tuning

| Parameter | Purpose | Typical Values |
| :--- | :--- | :--- |
| `threshold` / `min_score` | Minimum relevance score required for an entity to be included. | `0.80` - `0.90` (Strict/Exact)<br>`0.65` - `0.79` (Broad) |
| `relative_drop_off` / `auto_k` | Auto-K cutoff ratio. Discards candidates that drop below $\text{top\_score} \times \text{drop\_off}$. | `0.85` (Default: within 15% of top match)<br>`0.90` (Strict top group) |
| `top_k` | Upper bound ceiling of candidates evaluated before Auto-K filtering. | `5` - `20` |
| `max_depth` | Multi-hop graph exploration depth in relations (BFS hops). | `1` - `2` |
| `device` | Hardware acceleration backend for local embedding and reranking. | `auto`, `cuda`, `cpu` |

---

## Docker Deployment

### Run Container
```bash
docker run -d \
  --name graphite-server \
  -p 8080:8080 \
  -v graphite-data:/data \
  ghcr.io/joaocardosodias/graphite:latest
```

### Docker Compose
```yaml
services:
  graphite:
    image: ghcr.io/joaocardosodias/graphite:latest
    container_name: graphite-server
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./data:/data
    environment:
      - GRAPHITE_DB_PATH=/data/knowledge.graph
      - GRAPHITE_PORT=8080
      - GRAPHITE_HOST=0.0.0.0
```

---

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
