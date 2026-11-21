"""Local RAG MCP server, using persistent Chroma collections."""
import hashlib
import math
import os
import re
from pathlib import Path

import chromadb
import httpx
import trafilatura
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

class HashEmbedding:
    """Deterministic offline token-hash vectors; no model download required."""
    def __call__(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % 384] += 1.0 if digest[4] % 2 else -1.0
            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])
        return vectors

mcp = FastMCP("vectorvault-mcp")
_client = None
_embedding = HashEmbedding()


def client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault"))
    return _client


def collection(name):
    return client().get_collection(name)


def ingest_text(collection_name, text, source, metadata=None):
    if not text.strip():
        raise ValueError("Document has no extractable text")
    metadata = metadata or {}
    if any(not isinstance(v, (str, int, float, bool)) for v in metadata.values()):
        raise ValueError("Metadata values must be strings, numbers, or booleans")
    if any(k in metadata for k in ("document_id", "source", "ordinal")):
        raise ValueError("document_id, source, and ordinal are reserved metadata keys")
    doc_id = hashlib.sha256(source.encode()).hexdigest()
    col = collection(collection_name)
    chunks = [text[i:i + 1500] for i in range(0, len(text), 1200)]
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    # Remove stale chunks when a document becomes shorter.
    old = col.get(where={"document_id": doc_id}, include=[])["ids"]
    col.upsert(ids=ids, documents=chunks, embeddings=_embedding(chunks), metadatas=[{**metadata, "document_id": doc_id, "source": source, "ordinal": i} for i in range(len(chunks))])
    stale = list(set(old) - set(ids))
    if stale:
        col.delete(ids=stale)
    return {"document_id": doc_id, "chunks": len(chunks), "source": source}


@mcp.tool()
def create_collection(name: str) -> dict:
    """Create a named collection. Existing names are rejected."""
    client().create_collection(name, metadata={"hnsw:space": "cosine"})
    return {"name": name}


@mcp.tool()
def list_collections() -> list[str]:
    """List available collections."""
    return [c.name for c in client().list_collections()]


@mcp.tool()
def delete_collection(name: str) -> dict:
    """Delete a collection and all its documents permanently."""
    client().delete_collection(name)
    return {"deleted": name}


@mcp.tool()
def ingest_document(collection_name: str, path: str, metadata: dict | None = None) -> dict:
    """Ingest a local UTF-8 text, Markdown, or PDF file."""
    file = Path(path).resolve(strict=True)
    if file.suffix.lower() == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(file)).pages)
    elif file.suffix.lower() in (".txt", ".md", ".markdown"):
        text = file.read_text(encoding="utf-8")
    else:
        raise ValueError("Supported file types: .txt, .md, .markdown, .pdf")
    return ingest_text(collection_name, text, str(file), metadata)


@mcp.tool()
def ingest_folder(collection_name: str, path: str, metadata: dict | None = None) -> dict:
    """Recursively ingest supported files, reporting individual failures."""
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("path must be a directory")
    results, errors = [], []
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.suffix.lower() in (".txt", ".md", ".markdown", ".pdf"):
            try:
                results.append(ingest_document(collection_name, str(file), metadata))
            except Exception as exc:
                errors.append({"path": str(file), "error": str(exc)})
    return {"documents": results, "errors": errors}


@mcp.tool()
def ingest_url(collection_name: str, url: str, metadata: dict | None = None) -> dict:
    """Fetch and ingest an HTTP(S) webpage or plain text URL."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("URL must use HTTP or HTTPS")
    response = httpx.get(url, follow_redirects=True, timeout=20)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        text = trafilatura.extract(response.text) or ""
    elif "text/plain" in content_type:
        text = response.text
    else:
        raise ValueError("URL must return HTML or plain text")
    return ingest_text(collection_name, text, url, metadata)


@mcp.tool()
def get_document(collection_name: str, document_id: str) -> dict:
    """Retrieve a document's chunks and metadata in order."""
    result = collection(collection_name).get(where={"document_id": document_id}, include=["documents", "metadatas"])
    if not result["ids"]:
        raise ValueError("Document not found")
    rows = sorted(zip(result["documents"], result["metadatas"]), key=lambda row: row[1]["ordinal"])
    return {"document_id": document_id, "chunks": [row[0] for row in rows], "metadata": rows[0][1]}


@mcp.tool()
def update_metadata(collection_name: str, document_id: str, metadata: dict) -> dict:
    """Merge scalar metadata into every chunk of a document."""
    if any(k in metadata for k in ("document_id", "source", "ordinal")) or any(not isinstance(v, (str, int, float, bool)) for v in metadata.values()):
        raise ValueError("Metadata must contain scalar values and no reserved keys")
    col = collection(collection_name)
    result = col.get(where={"document_id": document_id}, include=["metadatas"])
    if not result["ids"]:
        raise ValueError("Document not found")
    col.update(ids=result["ids"], metadatas=[{**m, **metadata} for m in result["metadatas"]])
    return {"document_id": document_id, "updated_chunks": len(result["ids"])}


@mcp.tool()
def search(collection_name: str, query: str, limit: int = 6, mode: str = "semantic", filters: dict | None = None) -> list[dict]:
    """Search chunks with semantic or hybrid ranking and exact metadata filters."""
    if not query.strip() or not 1 <= limit <= 100 or mode not in ("semantic", "hybrid"):
        raise ValueError("Provide a query, limit between 1 and 100, and mode semantic or hybrid")
    col = collection(collection_name)
    where = None
    if filters:
        clauses = [{k: {"$eq": v}} for k, v in filters.items()]
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    # Hybrid needs the full filtered corpus to avoid excluding lexical matches.
    if mode == "hybrid":
        data = col.get(where=where, include=["documents", "metadatas", "embeddings"])
        q = _embedding([query])[0]
        terms = set(re.findall(r"\w+", query.lower()))
        rows = []
        for ident, doc, meta, vec in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
            semantic = sum(a*b for a,b in zip(q, vec))
            lexical = len(terms & set(re.findall(r"\w+", doc.lower()))) / len(terms) if terms else 0
            rows.append({"id": ident, "text": doc, "metadata": meta, "score": (semantic + lexical) / 2})
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    if not col.count():
        return []
    data = col.query(query_embeddings=_embedding([query]), n_results=min(limit, col.count()), where=where, include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": d, "metadata": m, "score": 1 - dist} for i,d,m,dist in zip(data["ids"][0], data["documents"][0], data["metadatas"][0], data["distances"][0])]


if __name__ == "__main__":
    mcp.run(transport="stdio")
