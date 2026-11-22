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
import settings


async def index(request):
    return FileResponse(Path(__file__).parent / "dashboard" / "index.html")


async def overview(request):
    collections = []
    for col in rag_mcp.client().list_collections():
        collections.append({"name": col.name, "chunks": col.count(), "metadata": col.metadata or {}})
    config = settings.load()
    return JSONResponse({"collections": sorted(collections, key=lambda c: c["name"]), "data_dir": str(Path(os.getenv("VECTORVAULT_DATA_DIR", "data/vectorvault")).resolve()), "provider": config["provider"], "embedding_model": config["model"], "llm_provider": config["llm_provider"], "llm_model": config["llm_model"], "api_key_configured": bool(settings.key(config)), "logs": activity.recent()})


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


async def action(request):
    # Browser requests must originate from this dashboard; no CORS is enabled.
    origin = request.headers.get("origin")
    if origin != str(request.base_url).rstrip("/") or request.headers.get("content-type", "").split(";")[0] != "application/json":
        return JSONResponse({"error": "Request must originate from the dashboard"}, status_code=403)
    try:
        payload = await request.json()
        operation = request.path_params["operation"]
        if operation == "settings":
            config = settings.save(payload.get("provider"), payload.get("model"), payload.get("api_key"))
            result = {"provider": config["provider"], "model": config["model"]}
        elif operation == "llm-settings":
            config = settings.save_llm(payload.get("provider"), payload.get("model"))
            result = {"provider": config["llm_provider"], "model": config["llm_model"]}
        elif operation == "answer":
            result = rag_mcp.answer_question(payload.get("collection", ""), payload.get("question", ""))
        elif operation == "create":
            result = rag_mcp.create_collection(payload.get("name", ""))
        elif operation == "ingest":
            kind, collection_name = payload.get("kind"), payload.get("collection", "")
            if kind == "text":
                result = rag_mcp.ingest_content(collection_name, payload.get("content", ""), payload.get("source", ""))
            elif kind == "file":
                result = rag_mcp.ingest_document(collection_name, payload.get("source", ""))
            elif kind == "folder":
                result = rag_mcp.ingest_folder(collection_name, payload.get("source", ""))
            elif kind == "url":
                result = rag_mcp.ingest_url(collection_name, payload.get("source", ""))
            else:
                raise ValueError("Select a valid source type")
        elif operation == "search":
            result = rag_mcp.search(payload.get("collection", ""), payload.get("query", ""), mode=payload.get("mode", "hybrid"))
        else:
            return JSONResponse({"error": "Unknown action"}, status_code=404)
        return JSONResponse({"result": result})
    except Exception as exc:
        # Avoid returning provider errors, which can contain sensitive request details.
        if isinstance(exc, ValueError):
            message = str(exc)
        else:
            message = f"{type(exc).__name__}: operation failed; check your model, credentials, and connection."
        return JSONResponse({"error": message}, status_code=400)


app = Starlette(middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])], routes=[Route("/", index), Route("/api/overview", overview), Route("/api/collections/{name}/documents", documents), Route("/api/{operation}", action, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
