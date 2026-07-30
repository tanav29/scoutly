"""Persistent Chroma vector index for Scoutly chunks."""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import chromadb

INDEX_PATH = Path(os.getenv("SCOUTLY_CHROMA_DIR", "data/chroma"))
INDEX_PATH.mkdir(parents=True, exist_ok=True)
_client = chromadb.PersistentClient(path=str(INDEX_PATH))
_collection = _client.get_or_create_collection("scoutly_chunks", metadata={"hnsw:space": "cosine"})


class HashEmbedding:
    """Offline deterministic embeddings for development and tests.

    Production can replace this class with an OpenAI embedding function without
    changing the collection or retrieval contract.
    """

    dimension = 384

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vector = [0.0] * self.dimension
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])
        return vectors


# Chroma uses this callable for both indexing and querying.
_embedding = HashEmbedding()


def upsert(source_id: str, version_id: str, chunks: list[tuple[str, str, int]]) -> None:
    if not chunks:
        return
    _collection.upsert(
        ids=[chunk_id for chunk_id, _, _ in chunks],
        documents=[text for _, text, _ in chunks],
        metadatas=[{"source_id": source_id, "source_version_id": version_id, "ordinal": ordinal} for _, _, ordinal in chunks],
        embeddings=_embedding([text for _, text, _ in chunks]),
    )


def query(text: str, source_ids: list[str], limit: int = 6) -> list[str]:
    if not source_ids:
        return []
    result = _collection.query(
        query_embeddings=_embedding([text]),
        n_results=limit,
        where={"source_id": {"$in": source_ids}},
    )
    return result.get("ids", [[]])[0]


def count_for_source(source_id: str) -> int:
    """Return the number of embedded chunks persisted for one source."""
    result = _collection.get(where={"source_id": source_id}, include=[])
    return len(result.get("ids", []))
