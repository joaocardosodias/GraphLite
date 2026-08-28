<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-white.png">
    <img src="https://raw.githubusercontent.com/joaocardosodias/Graphite/main/assets/logo-black.png" alt="Graphite Logo" width="120" />
  </picture>
</p>

<h1 align="center">Graphite Python SDK</h1>

<p align="center">
  <strong>Native Python bindings for the Graphite embedded GraphRAG engine</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/graphite-database"><img src="https://img.shields.io/pypi/v/graphite-db.svg?color=black" alt="PyPI" /></a>
  <a href="https://github.com/joaocardosodias/Graphite/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-black.svg" alt="License" /></a>
</p>

> **Graphite DB** combines Knowledge Graphs, SIMD AVX2 Vector Search, BM25 Lexical Indexing, and Token-Budgeted Context Synthesis into a single-file `.graph` database with zero-copy memory mapping.

---

## Installation

```bash
pip install graphite-database
```

---

## Quickstart

```python
import graphite_db as graphite

# 1. Open or create an embedded database
db = graphite.open("knowledge.graph", dim=384, max_tokens=400)

# 2. Insert entities and create connections
id_auth = db.insert("AuthService", entity_type="Module", description="Validates JWT tokens")
id_db = db.insert("UsersDB", entity_type="Database", description="PostgreSQL primary cluster")
db.connect("AuthService", "UsersDB", relation="CONNECTS_TO", weight=0.95)
db.flush()

# 3. Query GraphRAG context with automatic local FastEmbed embedding
result = db.query("How does authentication connect to the database?")

print(f"Retrieved {result.token_count} tokens:")
print(result.markdown)
```

---

## In-Memory Ephemeral Storage

```python
import graphite_db as graphite

with graphite.in_memory(dim=384) as db:
    db.remember("User prefers concise Portuguese responses.", category="UserPreference")
    result = db.query("What language does the user prefer?")
    print(result.markdown)
```

---

## Direct Vector Operations & Custom Embeddings

```python
import graphite_db as graphite

# Generate 384-dimensional vector locally on CPU
vector = graphite.embed("How does authentication work?")

db = graphite.in_memory(dim=384)
result = db.retrieve_context(vector, query_text="authentication", max_tokens=300)
print(result.markdown)
```

---

## License

Dual-licensed under [MIT](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/joaocardosodias/Graphite/blob/main/LICENSE-APACHE).
