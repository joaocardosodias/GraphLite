"""
Graphite DB — Embedded Single-File GraphRAG Engine in Pure Rust.

Official Python SDK.
"""

try:
    from ._graphite_db import (
        Graphite,
        QueryResult,
        embed,
        embed_batch,
        __version__,
    )
except ImportError as e:
    raise ImportError(
        "Failed to load compiled Graphite native extension (_graphite_db). "
        "Please install graphite-db using pip or rebuild using maturin develop."
    ) from e

def open(
    path: str = "knowledge.graphite",
    dim: int = 384,
    max_tokens: int = 400,
    metric: str = "cosine",
    quantization: str = "sq8",
) -> Graphite:
    """
    Opens an existing or creates a new `.graphite` database file.

    Args:
        path: Path to the `.graphite` database file.
        dim: Embedding vector dimensionality (default: 384 for MiniLM-L6).
        max_tokens: Default token budget for prompt context synthesis (default: 400).
        metric: Distance metric ('cosine', 'euclidean', 'dot', 'manhattan').
        quantization: Vector quantization mode ('sq8', 'none').

    Returns:
        Graphite database instance.
    """
    return Graphite.open(
        path=path,
        dim=dim,
        max_tokens=max_tokens,
        metric=metric,
        quantization=quantization,
    )

def in_memory(
    dim: int = 384,
    max_tokens: int = 400,
    metric: str = "cosine",
    quantization: str = "sq8",
) -> Graphite:
    """
    Creates an ephemeral, in-memory Graphite database instance.

    Args:
        dim: Embedding vector dimensionality (default: 384).
        max_tokens: Default token budget for prompt context synthesis (default: 400).
        metric: Distance metric ('cosine', 'euclidean', 'dot', 'manhattan').
        quantization: Vector quantization mode ('sq8', 'none').

    Returns:
        Graphite database instance.
    """
    return Graphite.in_memory(
        dim=dim,
        max_tokens=max_tokens,
        metric=metric,
        quantization=quantization,
    )

__all__ = [
    "Graphite",
    "QueryResult",
    "open",
    "in_memory",
    "embed",
    "embed_batch",
    "__version__",
]
