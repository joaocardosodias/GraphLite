from typing import Optional, List, Dict, Any, Union

__version__: str

class QueryResult:
    markdown: str
    token_count: int
    entities_count: int
    edges_count: int

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...

class Graphite:
    def __init__(
        self,
        path: Optional[str] = ...,
        dim: int = ...,
        max_tokens: int = ...,
        metric: str = ...,
        quantization: str = ...,
    ) -> None: ...
    
    @staticmethod
    def open(
        path: str,
        dim: int = ...,
        max_tokens: int = ...,
        metric: str = ...,
        quantization: str = ...,
    ) -> Graphite: ...
    
    @staticmethod
    def in_memory(
        dim: int = ...,
        max_tokens: int = ...,
        metric: str = ...,
        quantization: str = ...,
    ) -> Graphite: ...

    def query(
        self,
        text: str,
        top_k: int = ...,
        max_tokens: Optional[int] = ...,
        max_depth: Optional[int] = ...,
        alpha: Optional[float] = ...,
    ) -> QueryResult: ...

    def retrieve_context(
        self,
        vector: List[float],
        query_text: Optional[str] = ...,
        top_k: int = ...,
        max_tokens: Optional[int] = ...,
        max_depth: Optional[int] = ...,
        alpha: Optional[float] = ...,
    ) -> QueryResult: ...

    def insert(
        self,
        name: str,
        entity_type: str = ...,
        description: str = ...,
        vector: Optional[List[float]] = ...,
    ) -> int: ...

    def ingest(
        self,
        source: Optional[str] = ...,
        text: Optional[str] = ...,
        title: str = ...,
        chunk_size: int = ...,
        overlap: int = ...,
        batch_size: int = ...,
    ) -> int: ...

    def connect(
        self,
        source_name: str,
        target_name: str,
        relation: str = ...,
        weight: float = ...,
    ) -> None: ...

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        relation: str = ...,
        weight: float = ...,
        directed: bool = ...,
    ) -> None: ...

    def flush(self) -> None: ...
    def inspect(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Graphite: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

def open(
    path: str = ...,
    dim: int = ...,
    metric: str = ...,
    quantization: str = ...,
    device: str = ...,
) -> Graphite: ...

def in_memory(
    dim: int = ...,
    metric: str = ...,
    quantization: str = ...,
    device: str = ...,
) -> Graphite: ...

def embed(text: str) -> List[float]: ...
def embed_batch(texts: List[str]) -> List[List[float]]: ...
