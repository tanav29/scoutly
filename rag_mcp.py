"""Local RAG MCP server, using persistent Chroma collections."""
import hashlib
import os
import re
from pathlib import Path

import chromadb
import httpx
import trafilatura
from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def embed(texts: list[str]) -> list[list[float]]:
    """Embed text with OpenAI; OPENAI_API_KEY must be configured."""
    result = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]


mcp = FastMCP("vectorvault-mcp")
_client = None


def client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault"))
    return _client


def collection(name):
    col = client().get_collection(name)
    if (col.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Collection uses an incompatible or unknown embedding model. Create a new collection and re-ingest its documents.")
    return col


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
    col.upsert(ids=ids, documents=chunks, embeddings=embed(chunks), metadatas=[{**metadata, "document_id": doc_id, "source": source, "ordinal": i} for i in range(len(chunks))])
    stale = list(set(old) - set(ids))
    if stale:
        col.delete(ids=stale)
    return {"document_id": doc_id, "chunks": len(chunks), "source": source}


@mcp.tool()
def create_collection(name: str) -> dict:
    """Create a named collection. Existing names are rejected."""
    client().create_collection(name, metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL})
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
def ingest_content(collection_name: str, content: str, source: str, metadata: dict | None = None) -> dict:
    """Ingest text supplied directly by an agent, using a stable source identifier."""
    if not source.strip():
        raise ValueError("source must not be empty")
    return ingest_text(collection_name, content, source, metadata)


@mcp.tool()
def list_documents(collection_name: str, limit: int = 100, offset: int = 0) -> dict:
    """List documents in a collection, ordered by document ID, with pagination."""
    if not 1 <= limit <= 1000 or offset < 0:
        raise ValueError("limit must be 1-1000 and offset must be nonnegative")
    data = collection(collection_name).get(include=["metadatas"])
    docs = {}
    for meta in data["metadatas"]:
        docs.setdefault(meta["document_id"], {k: v for k, v in meta.items() if k != "ordinal"})
    ordered = [docs[key] for key in sorted(docs)]
    return {"documents": ordered[offset:offset + limit], "total": len(ordered), "offset": offset}


@mcp.tool()
def delete_document(collection_name: str, document_id: str) -> dict:
    """Delete a document and all of its chunks from a collection."""
    col = collection(collection_name)
    ids = col.get(where={"document_id": document_id}, include=[])["ids"]
    if not ids:
        raise ValueError("Document not found")
    col.delete(ids=ids)
    return {"document_id": document_id, "deleted_chunks": len(ids)}


@mcp.tool()
def collection_info(collection_name: str) -> dict:
    """Inspect the collection's embedding model and stored chunk count."""
    col = collection(collection_name)
    return {"name": collection_name, "chunks": col.count(), "embedding_model": EMBEDDING_MODEL}


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
        q = embed([query])[0]
        terms = set(re.findall(r"\w+", query.lower()))
        rows = []
        for ident, doc, meta, vec in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
            qnorm = sum(a*a for a in q) ** 0.5 or 1.0
            vnorm = sum(a*a for a in vec) ** 0.5 or 1.0
            semantic = sum(a*b for a,b in zip(q, vec)) / (qnorm * vnorm)
            lexical = len(terms & set(re.findall(r"\w+", doc.lower()))) / len(terms) if terms else 0
            rows.append({"id": ident, "text": doc, "metadata": meta, "score": (semantic + lexical) / 2})
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]
    if not col.count():
        return []
    data = col.query(query_embeddings=embed([query]), n_results=min(limit, col.count()), where=where, include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": d, "metadata": m, "score": 1 - dist} for i,d,m,dist in zip(data["ids"][0], data["documents"][0], data["metadatas"][0], data["distances"][0])]


if __name__ == "__main__":
    mcp.run(transport="stdio")
