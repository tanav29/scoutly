"""Read-only local dashboard for VectorVault."""
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Route

import activity
import rag_mcp


async def index(request):
    return FileResponse(Path(__file__).parent / "dashboard" / "index.html")


async def overview(request):
    collections = []
    for col in rag_mcp.client().list_collections():
        collections.append({"name": col.name, "chunks": col.count(), "metadata": col.metadata or {}})
    return JSONResponse({"collections": sorted(collections, key=lambda c: c["name"]), "data_dir": str(Path(os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault")).resolve()), "embedding_model": rag_mcp.EMBEDDING_MODEL, "api_key_configured": bool(os.getenv("OPENAI_API_KEY")), "logs": activity.recent()})


async def documents(request):
    # Read legacy collections too, including those with incompatible embeddings.
    col = rag_mcp.client().get_collection(request.path_params["name"])
    data = col.get(include=["documents", "metadatas"])
    docs = {}
    for text, meta in zip(data["documents"], data["metadatas"]):
        meta = meta or {}
        doc_id = meta.get("document_id", "unknown")
        doc = docs.setdefault(doc_id, {"id": doc_id, "source": meta.get("source", "Unknown"), "metadata": {k: v for k, v in meta.items() if k != "ordinal"}, "chunks": []})
        doc["chunks"].append({"ordinal": meta.get("ordinal", 0), "text": text})
    for doc in docs.values():
        doc["chunks"].sort(key=lambda chunk: chunk["ordinal"])
    return JSONResponse({"documents": sorted(docs.values(), key=lambda doc: doc["source"])})


app = Starlette(middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])], routes=[Route("/", index), Route("/api/overview", overview), Route("/api/collections/{name}/documents", documents)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
