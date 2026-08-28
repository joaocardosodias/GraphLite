"""
Graphite DB — Embedded Single-File GraphRAG Engine in Pure Rust.

Official Python SDK.
"""

import os
import re
from typing import Optional, List, Dict, Any, Union

try:
    from ._graphite_db import (
        Graphite as _NativeGraphite,
        QueryResult,
        embed,
        embed_batch,
        __version__,
    )
except ImportError as e:
    raise ImportError(
        "Failed to load compiled Graphite native extension (_graphite_db). "
        "Please install graphite-database using pip or rebuild using maturin develop."
    ) from e


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Divide texto e documentos estruturados em chunks semanticos completos.
    """
    chunks = []
    
    # Divide por cabecalhos markdown (#, ##, ###)
    sections = re.split(r'(?m)^(#{1,6}\s+.+)$', text)
    if len(sections) > 1:
        i = 1
        prefix_context = ""
        while i < len(sections):
            header = sections[i].strip().lstrip("#").strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ""
            
            # Se for um cabecalho curto sem corpo de artigo, guarda como contexto para o proximo
            if len(content.split()) < 10 and not re.search(r'Art\.?\s*\d+', header, re.IGNORECASE):
                prefix_context = f"{header} - {content}".strip().strip("-").strip()
                i += 2
                continue

            full_chunk_text = f"{prefix_context}\n\n{header}\n\n{content}".strip() if prefix_context else f"{header}\n\n{content}".strip()
            chunk_title = f"{header} ({prefix_context})"[:80] if prefix_context else header[:80]
            prefix_context = ""

            if full_chunk_text:
                chunks.append({
                    "title": chunk_title,
                    "content": full_chunk_text
                })
            i += 2
        if chunks:
            return chunks

    # Fallback para divisao por paragrafos
    palavras = text.split()
    for i in range(0, len(palavras), max(chunk_size - overlap, 50)):
        bloco = " ".join(palavras[i:i + chunk_size])
        if bloco.strip():
            chunks.append({
                "title": f"Trecho {i // max(chunk_size - overlap, 50) + 1}",
                "content": bloco
            })

    return chunks


def ingest(self, path: str, chunk_size: int = 400, overlap: int = 50, batch_size: int = 64) -> int:
    """
    Ingesta um arquivo (.md, .txt, .pdf) ou diretorio diretamente no banco.
    Gera embeddings locais e constroi as conexoes no grafo automaticamente.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo ou diretorio nao encontrado: {path}")

    files_to_process = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".md", ".txt", ".pdf", ".json", ".csv")):
                    files_to_process.append(os.path.join(root, f))
    else:
        files_to_process.append(path)

    total_ingested = 0

    for fpath in files_to_process:
        text = ""
        if fpath.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(fpath)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except ImportError:
                raise ImportError("Instale pypdf para ingerir arquivos PDF: pip install pypdf")
        else:
            import builtins
            with builtins.open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            textos = [c["content"] for c in batch]
            vetores = embed_batch(textos)

            prev_id = None
            for c, v in zip(batch, vetores):
                node_id = self.insert(
                    name=c["title"],
                    entity_type="Artigo",
                    description=c["content"],
                    vector=v
                )
                if prev_id is not None:
                    self.add_edge(prev_id, node_id, "SEQUENCIA_DE", 0.9, True)
                prev_id = node_id
                total_ingested += 1

    self.flush()
    return total_ingested

_NativeGraphite.ingest = ingest
Graphite = _NativeGraphite


def open(
    path: str = "knowledge.graph",
    dim: int = 384,
    metric: str = "cosine",
    quantization: str = "sq8",
    device: str = "auto",
) -> Graphite:
    """
    Abre ou cria um arquivo de banco de dados `.graph`.
    """
    return Graphite.open(
        path=path,
        dim=dim,
        metric=metric,
        quantization=quantization,
        device=device,
    )


def in_memory(
    dim: int = 384,
    metric: str = "cosine",
    quantization: str = "sq8",
    device: str = "auto",
) -> Graphite:
    """
    Cria um banco de dados efêmero em memória.
    """
    return Graphite.in_memory(
        dim=dim,
        metric=metric,
        quantization=quantization,
        device=device,
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
