"""Scoutly company-brain API.

The development profile uses SQLite and the local object store so the product
can be started with one command.  The schema deliberately keeps workspace IDs,
source versions, jobs and citations explicit; production deployments can swap
the repository/object-store adapters for PostgreSQL/pgvector and S3.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from utils.providers import grounded_completion
from utils.vector_store import upsert as vector_upsert, query as vector_query, count_for_source as vector_count

ROOT = Path(os.getenv("SCOUTLY_DATA_DIR", "data"))
ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("SCOUTLY_DB_PATH", str(ROOT / "scoutly.sqlite3")))
OBJECTS = ROOT / "objects"
OBJECTS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.getenv("SCOUTLY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
WORKSPACE_ID = os.getenv("SCOUTLY_WORKSPACE_ID", "bootstrap")

app = FastAPI(title="Scoutly Company Brain API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SCOUTLY_WEB_ORIGIN", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, source_limit INTEGER NOT NULL DEFAULT 1000, token_limit_daily INTEGER NOT NULL DEFAULT 1000000);
CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', color TEXT NOT NULL DEFAULT '#7c6cf2', archived_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(workspace_id,name));
CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, title TEXT NOT NULL, source_type TEXT NOT NULL, canonical_url TEXT, mime_type TEXT, original_key TEXT, extracted_key TEXT, source_metadata TEXT NOT NULL DEFAULT '{}', content_hash TEXT, state TEXT NOT NULL, error_code TEXT, error_message TEXT, retry_count INTEGER NOT NULL DEFAULT 0, archived_at TEXT, deleted_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS collection_sources (collection_id TEXT NOT NULL, source_id TEXT NOT NULL, PRIMARY KEY(collection_id,source_id));
CREATE TABLE IF NOT EXISTS source_versions (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, content_hash TEXT NOT NULL, parser_version TEXT NOT NULL, indexed_at TEXT, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source_version_id TEXT NOT NULL, source_id TEXT NOT NULL, ordinal INTEGER NOT NULL, location TEXT NOT NULL DEFAULT '{}', text TEXT NOT NULL, token_count INTEGER NOT NULL DEFAULT 0, search_text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingestion_jobs (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, job_type TEXT NOT NULL, state TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, attempt_count INTEGER NOT NULL DEFAULT 0, error TEXT, started_at TEXT, ended_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, title TEXT NOT NULL, visibility TEXT NOT NULL DEFAULT 'private', scope_collections TEXT NOT NULL DEFAULT '[]', scope_sources TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, provider_metadata TEXT NOT NULL DEFAULT '{}', completion_state TEXT NOT NULL DEFAULT 'complete', token_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS message_citations (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, source_version_id TEXT NOT NULL, chunk_id TEXT NOT NULL, excerpt TEXT NOT NULL, relevance_score REAL NOT NULL, citation_order INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, event_type TEXT NOT NULL, source_id TEXT, chat_id TEXT, metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage_daily (day TEXT NOT NULL, workspace_id TEXT NOT NULL, source_bytes INTEGER NOT NULL DEFAULT 0, indexed_chunks INTEGER NOT NULL DEFAULT 0, ingestion_jobs INTEGER NOT NULL DEFAULT 0, chat_requests INTEGER NOT NULL DEFAULT 0, token_total INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(day,workspace_id));
"""


def init_db() -> None:
    connection = db()
    connection.executescript(SCHEMA)
    connection.execute("INSERT OR IGNORE INTO workspaces(id,name,slug,created_at) VALUES (?,?,?,?)", (WORKSPACE_ID, os.getenv("SCOUTLY_WORKSPACE_NAME", "Scoutly workspace"), "bootstrap", now()))
    connection.commit()
    connection.close()


init_db()
jobs: dict[str, asyncio.Task[Any]] = {}


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def require_source(source_id: str) -> dict[str, Any]:
    connection = db()
    row = connection.execute("SELECT * FROM sources WHERE id=? AND workspace_id=?", (source_id, WORKSPACE_ID)).fetchone()
    connection.close()
    if not row:
        raise HTTPException(404, "Source not found")
    result = dict(row)
    result["source_metadata"] = json.loads(result["source_metadata"])
    return result


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    color: str = "#7c6cf2"


class ScopeIn(BaseModel):
    collection_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ChatIn(BaseModel):
    title: str = "New chat"
    scope: ScopeIn = ScopeIn()


class ChatMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class WebpageIn(BaseModel):
    url: str
    collection_ids: list[str] = []


class YouTubeIn(BaseModel):
    url: str
    language_preference: str | None = None
    collection_ids: list[str] = []


def validate_scope(scope: ScopeIn) -> None:
    connection = db()
    if scope.source_ids:
        marks = ",".join("?" * len(scope.source_ids))
        count = connection.execute(f"SELECT COUNT(*) FROM sources WHERE workspace_id=? AND deleted_at IS NULL AND id IN ({marks})", [WORKSPACE_ID, *scope.source_ids]).fetchone()[0]
        if count != len(set(scope.source_ids)):
            connection.close(); raise HTTPException(422, "One or more sources are not available")
    if scope.collection_ids:
        marks = ",".join("?" * len(scope.collection_ids))
        count = connection.execute(f"SELECT COUNT(*) FROM collections WHERE workspace_id=? AND archived_at IS NULL AND id IN ({marks})", [WORKSPACE_ID, *scope.collection_ids]).fetchone()[0]
        if count != len(set(scope.collection_ids)):
            connection.close(); raise HTTPException(422, "One or more collections are not available")
    connection.close()


def validate_public_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(422, "Only public http(s) URLs are supported")
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local") or host.startswith(("10.", "192.168.", "172.16.")):
        raise HTTPException(422, "Private and local network URLs are not allowed")
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def object_path(key: str) -> Path:
    target = (OBJECTS / key).resolve()
    if OBJECTS.resolve() not in target.parents:
        raise HTTPException(400, "Invalid object key")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.parts: list[str] = []; self.skip = 0
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}: self.skip += 1
    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip: self.skip -= 1
    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip(): self.parts.append(data.strip())


def extract_text(raw: bytes, mime: str, filename: str = "") -> str:
    suffix = Path(filename).suffix.lower()
    if mime == "application/pdf" or suffix == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(__import__("io").BytesIO(raw)).pages)
        except Exception as exc: raise ValueError(f"pdf_extract_failed: {exc}") from exc
    if mime == "text/html" or suffix in {".html", ".htm"}:
        parser = TextExtractor(); parser.feed(raw.decode("utf-8", "ignore")); return "\n".join(parser.parts)
    if suffix in {".docx", ".pptx", ".xlsx"}:
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(raw)) as archive:
                xml = "\n".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))
            return re.sub(r"<[^>]+>", " ", html.unescape(xml))
        except Exception as exc: raise ValueError(f"office_extract_failed: {exc}") from exc
    return raw.decode("utf-8", "ignore")


def chunks_for(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean: return []
    result: list[str] = []; start = 0
    while start < len(clean):
        end = min(len(clean), start + size); result.append(clean[start:end])
        if end == len(clean): break
        start = end - overlap
    return result


async def ingest(source_id: str) -> None:
    connection = db(); source = connection.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    if not source: connection.close(); return
    connection.execute("UPDATE sources SET state='processing',updated_at=? WHERE id=?", (now(), source_id))
    job_id = uuid.uuid4().hex
    connection.execute("INSERT INTO ingestion_jobs VALUES (?,?,?,?,?,?,?,?,?,?)", (job_id, source_id, "ingest", "processing", 10, 1, None, now(), None, now())); connection.commit(); connection.close()
    try:
        raw = object_path(source["original_key"]).read_bytes()
        text = extract_text(raw, source["mime_type"] or "", source["title"])
        pieces = chunks_for(text)
        if not pieces: raise ValueError("empty_content")
        digest = hashlib.sha256(text.encode()).hexdigest()
        connection = db()
        existing = connection.execute("SELECT id FROM source_versions WHERE source_id=? AND content_hash=?", (source_id, digest)).fetchone()
        if not existing:
            version_id = uuid.uuid4().hex
            connection.execute("UPDATE source_versions SET active=0 WHERE source_id=?", (source_id,))
            connection.execute("INSERT INTO source_versions VALUES (?,?,?,?,?,?,?)", (version_id, source_id, digest, "scoutly-1", now(), 1, now()))
            vector_chunks = []
            for ordinal, piece in enumerate(pieces):
                chunk_id = uuid.uuid4().hex
                connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)", (chunk_id, version_id, source_id, ordinal, json.dumps({"ordinal": ordinal}), piece, len(piece.split()), piece.lower()))
                vector_chunks.append((chunk_id, piece, ordinal))
            vector_upsert(source_id, version_id, vector_chunks)
        else: version_id = existing["id"]
        connection.execute("UPDATE sources SET state='ready',content_hash=?,error_code=NULL,error_message=NULL,updated_at=? WHERE id=?", (digest, now(), source_id))
        connection.execute("UPDATE ingestion_jobs SET state='ready',progress=100,ended_at=? WHERE id=?", (now(), job_id))
        connection.execute("INSERT INTO audit_events VALUES (?,?,?,?,?,?,?)", (uuid.uuid4().hex, WORKSPACE_ID, "source.ready", source_id, None, "{}", now()))
        connection.commit(); connection.close()
    except Exception as exc:
        connection = db(); connection.execute("UPDATE sources SET state='failed',error_code=?,error_message=?,updated_at=? WHERE id=?", (str(exc).split(":")[0], str(exc)[:500], now(), source_id)); connection.execute("UPDATE ingestion_jobs SET state='failed',error=?,ended_at=? WHERE id=?", (str(exc)[:500], now(), job_id)); connection.commit(); connection.close()


def queue_ingest(source_id: str) -> None:
    jobs[source_id] = asyncio.create_task(ingest(source_id))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    connection = db(); connection.execute("SELECT 1"); connection.close(); return {"status": "ok", "database": "ok", "worker": "ok"}


@app.get("/api/workspace")
async def workspace() -> dict[str, Any]:
    connection = db(); row = connection.execute("SELECT * FROM workspaces WHERE id=?", (WORKSPACE_ID,)).fetchone(); connection.close(); return row_dict(row) or {}


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    connection = db()
    source_counts = {row["state"]: row["count"] for row in connection.execute("SELECT state,COUNT(*) AS count FROM sources WHERE workspace_id=? AND deleted_at IS NULL GROUP BY state", (WORKSPACE_ID,)).fetchall()}
    source_keys = [row["original_key"] for row in connection.execute("SELECT original_key FROM sources WHERE workspace_id=? AND deleted_at IS NULL", (WORKSPACE_ID,)).fetchall()]
    source_bytes = sum(object_path(key).stat().st_size for key in source_keys if key and object_path(key).exists())
    chats_count = connection.execute("SELECT COUNT(*) FROM chats WHERE workspace_id=? AND deleted_at IS NULL", (WORKSPACE_ID,)).fetchone()[0]
    messages_count = connection.execute("SELECT COUNT(*) FROM messages m JOIN chats c ON c.id=m.chat_id WHERE c.workspace_id=? AND c.deleted_at IS NULL", (WORKSPACE_ID,)).fetchone()[0]
    collections_count = connection.execute("SELECT COUNT(*) FROM collections WHERE workspace_id=? AND archived_at IS NULL", (WORKSPACE_ID,)).fetchone()[0]
    connection.close()
    return {"sources": sum(source_counts.values()), "ready_sources": source_counts.get("ready", 0), "processing_sources": source_counts.get("processing", 0) + source_counts.get("queued", 0), "failed_sources": source_counts.get("failed", 0), "chats": chats_count, "messages": messages_count, "collections": collections_count, "storage_bytes": source_bytes}


@app.get("/api/collections")
async def collections() -> list[dict[str, Any]]:
    connection = db(); rows = connection.execute("SELECT c.*,COUNT(cs.source_id) AS source_count FROM collections c LEFT JOIN collection_sources cs ON cs.collection_id=c.id WHERE c.workspace_id=? GROUP BY c.id ORDER BY c.name", (WORKSPACE_ID,)).fetchall(); connection.close(); return [dict(row) for row in rows]


@app.post("/api/collections")
async def create_collection(payload: CollectionIn) -> dict[str, Any]:
    item = (uuid.uuid4().hex, WORKSPACE_ID, payload.name, payload.description, payload.color, None, now(), now()); connection = db()
    try: connection.execute("INSERT INTO collections VALUES (?,?,?,?,?,?,?,?)", item); connection.commit()
    except sqlite3.IntegrityError: raise HTTPException(409, "Collection already exists")
    finally: connection.close()
    return {"id": item[0], "name": payload.name, "description": payload.description, "color": payload.color, "source_count": 0}


@app.patch("/api/collections/{collection_id}")
async def update_collection(collection_id: str, payload: CollectionIn) -> dict[str, Any]:
    connection = db(); result = connection.execute("UPDATE collections SET name=?,description=?,color=?,updated_at=? WHERE id=? AND workspace_id=?", (payload.name,payload.description,payload.color,now(),collection_id,WORKSPACE_ID)); connection.commit(); connection.close()
    if not result.rowcount: raise HTTPException(404, "Collection not found")
    return {"id": collection_id, **payload.model_dump()}


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str) -> dict[str, bool]:
    connection = db(); result = connection.execute("UPDATE collections SET archived_at=? WHERE id=? AND workspace_id=?", (now(),collection_id,WORKSPACE_ID)); connection.commit(); connection.close()
    if not result.rowcount: raise HTTPException(404, "Collection not found")
    return {"deleted": True}


def create_source(title: str, source_type: str, raw: bytes, mime: str, canonical_url: str | None, metadata: dict[str, Any], collection_ids: list[str]) -> dict[str, Any]:
    source_id = uuid.uuid4().hex; key = f"originals/{source_id}/{secrets.token_hex(8)}-{Path(title).name or 'source'}"; object_path(key).write_bytes(raw); timestamp = now(); connection = db()
    connection.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (source_id, WORKSPACE_ID, title, source_type, canonical_url, mime, key, None, json.dumps(metadata), None, "queued", None, None, 0, None, None, timestamp, timestamp))
    for collection_id in collection_ids: connection.execute("INSERT OR IGNORE INTO collection_sources VALUES (?,?)", (collection_id, source_id))
    connection.execute("INSERT INTO audit_events VALUES (?,?,?,?,?,?,?)", (uuid.uuid4().hex, WORKSPACE_ID, "source.created", source_id, None, json.dumps({"source_type": source_type}), timestamp)); connection.commit(); connection.close(); queue_ingest(source_id); return require_source(source_id)


@app.get("/api/sources")
async def sources(search: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
    connection = db(); query = "SELECT s.*, (SELECT COUNT(*) FROM chunks ch WHERE ch.source_id=s.id) AS chunk_count, GROUP_CONCAT(c.name) AS collection_names FROM sources s LEFT JOIN collection_sources cs ON cs.source_id=s.id LEFT JOIN collections c ON c.id=cs.collection_id WHERE s.workspace_id=? AND s.deleted_at IS NULL"; params: list[Any] = [WORKSPACE_ID]
    if search: query += " AND (s.title LIKE ? OR s.canonical_url LIKE ?)"; params += [f"%{search}%", f"%{search}%"]
    if state: query += " AND s.state=?"; params.append(state)
    query += " GROUP BY s.id ORDER BY s.created_at DESC"; rows = connection.execute(query, params).fetchall(); connection.close()
    result = []
    for row in rows:
        item = dict(row)
        chunks = item.pop("chunk_count", None)
        item["chunk_count"] = chunks if chunks is not None else 0
        item["vector_count"] = vector_count(item["id"]) if item["state"] == "ready" else 0
        item["indexed"] = item["state"] == "ready" and item["vector_count"] > 0
        result.append(item)
    return result


@app.post("/api/sources/upload")
async def upload_source(file: UploadFile = File(...), collection_ids: str = "[]") -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD: raise HTTPException(413, "File exceeds the workspace upload limit")
    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".markdown", ".txt"}
    title = file.filename or "upload.txt"
    if Path(title).suffix.lower() not in supported: raise HTTPException(415, "Unsupported file type")
    try: collections_value = json.loads(collection_ids)
    except json.JSONDecodeError: collections_value = []
    return create_source(title, "upload", raw, file.content_type or mimetypes.guess_type(title)[0] or "text/plain", None, {"filename": title}, collections_value)


async def fetched(url: str) -> tuple[bytes, str]:
    def fetch() -> tuple[bytes, str]:
        request = urllib.request.Request(url, headers={"User-Agent": "Scoutly/2.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(MAX_UPLOAD + 1); content_type = response.headers.get_content_type()
        if len(data) > MAX_UPLOAD: raise ValueError("response_too_large")
        if content_type not in {"text/html", "text/plain", "application/pdf"}: raise ValueError("unsupported_content_type")
        return data, content_type
    return await asyncio.to_thread(fetch)


@app.post("/api/sources/webpage")
async def webpage(payload: WebpageIn) -> dict[str, Any]:
    url = validate_public_url(payload.url)
    try: raw, mime = await fetched(url)
    except Exception as exc: raise HTTPException(422, f"Could not fetch webpage: {exc}")
    return create_source(urllib.parse.urlparse(url).netloc, "webpage", raw, mime, url, {"url": url}, payload.collection_ids)


@app.post("/api/sources/youtube")
async def youtube(payload: YouTubeIn) -> dict[str, Any]:
    url = validate_public_url(payload.url); parsed = urllib.parse.urlparse(url); video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0] or parsed.path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id): raise HTTPException(422, "Invalid YouTube URL")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[payload.language_preference] if payload.language_preference else ["en"])
        text = "\n".join(f"[{item['start']:.1f}s] {item['text']}" for item in transcript)
    except Exception: raise HTTPException(422, "transcript_unavailable")
    return create_source(f"YouTube {video_id}", "youtube", text.encode(), "text/plain", url, {"video_id": video_id, "language": payload.language_preference or "en", "transcript_provenance": "captions"}, payload.collection_ids)


@app.get("/api/sources/{source_id}")
async def source(source_id: str) -> dict[str, Any]: return require_source(source_id)


@app.post("/api/sources/{source_id}/retry")
async def retry(source_id: str) -> dict[str, Any]:
    source_value = require_source(source_id); connection = db(); connection.execute("UPDATE sources SET state='queued',retry_count=retry_count+1,updated_at=? WHERE id=?", (now(),source_id)); connection.commit(); connection.close(); queue_ingest(source_id); return require_source(source_id)


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str) -> dict[str, bool]:
    require_source(source_id); connection = db(); connection.execute("UPDATE sources SET deleted_at=?,state='archived',updated_at=? WHERE id=?", (now(),now(),source_id)); connection.commit(); connection.close(); return {"deleted": True}


@app.post("/api/collections/{collection_id}/sources/{source_id}")
async def add_to_collection(collection_id: str, source_id: str) -> dict[str, bool]:
    require_source(source_id); connection = db(); collection = connection.execute("SELECT id FROM collections WHERE id=? AND workspace_id=? AND archived_at IS NULL", (collection_id, WORKSPACE_ID)).fetchone()
    if not collection: connection.close(); raise HTTPException(404, "Collection not found")
    connection.execute("INSERT OR IGNORE INTO collection_sources VALUES (?,?)", (collection_id,source_id)); connection.commit(); connection.close(); return {"added": True}


@app.delete("/api/collections/{collection_id}/sources/{source_id}")
async def remove_from_collection(collection_id: str, source_id: str) -> dict[str, bool]:
    connection = db(); connection.execute("DELETE FROM collection_sources WHERE collection_id=? AND source_id=?", (collection_id,source_id)); connection.commit(); connection.close(); return {"removed": True}


@app.get("/api/chats")
async def chats() -> list[dict[str, Any]]:
    connection = db(); rows = connection.execute("SELECT id,title,created_at,updated_at FROM chats WHERE workspace_id=? AND deleted_at IS NULL ORDER BY updated_at DESC", (WORKSPACE_ID,)).fetchall(); connection.close()
    return [chat_row(row["id"]) for row in rows]


@app.post("/api/chats")
async def create_chat(payload: ChatIn) -> dict[str, Any]:
    validate_scope(payload.scope)
    chat_id = uuid.uuid4().hex; timestamp = now(); connection = db(); connection.execute("INSERT INTO chats VALUES (?,?,?,?,?,?,?,?,?)", (chat_id,WORKSPACE_ID,payload.title,"private",json.dumps(payload.scope.collection_ids),json.dumps(payload.scope.source_ids),timestamp,timestamp,None)); connection.commit(); connection.close(); return {"id": chat_id, "title": payload.title, "scope": payload.scope.model_dump()}


def chat_row(chat_id: str) -> dict[str, Any]:
    connection = db(); row = connection.execute("SELECT * FROM chats WHERE id=? AND workspace_id=? AND deleted_at IS NULL", (chat_id,WORKSPACE_ID)).fetchone(); connection.close()
    if not row: raise HTTPException(404,"Chat not found")
    item = dict(row); item["scope"] = {"collection_ids": json.loads(item.pop("scope_collections")), "source_ids": json.loads(item.pop("scope_sources"))}; return item


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, Any]: return chat_row(chat_id)


class ChatPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)


@app.patch("/api/chats/{chat_id}")
async def update_chat(chat_id: str, payload: ChatPatch) -> dict[str, Any]:
    chat_row(chat_id)
    if payload.title is not None:
        connection = db(); connection.execute("UPDATE chats SET title=?,updated_at=? WHERE id=?", (payload.title, now(), chat_id)); connection.commit(); connection.close()
    return chat_row(chat_id)


@app.patch("/api/chats/{chat_id}/scope")
async def update_scope(chat_id: str, payload: ScopeIn) -> dict[str, Any]:
    chat_row(chat_id); validate_scope(payload); connection = db(); connection.execute("UPDATE chats SET scope_collections=?,scope_sources=?,updated_at=? WHERE id=? AND workspace_id=?", (json.dumps(payload.collection_ids),json.dumps(payload.source_ids),now(),chat_id,WORKSPACE_ID)); connection.commit(); connection.close(); return chat_row(chat_id)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, bool]:
    chat_row(chat_id); connection = db(); connection.execute("UPDATE chats SET deleted_at=?,updated_at=? WHERE id=?", (now(),now(),chat_id)); connection.commit(); connection.close(); return {"deleted": True}


@app.get("/api/chats/{chat_id}/messages")
async def messages(chat_id: str) -> list[dict[str, Any]]:
    chat_row(chat_id); connection = db(); rows = connection.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)).fetchall(); connection.close(); return [dict(row) for row in rows]


def retrieve(scope: dict[str, Any], query: str) -> list[sqlite3.Row]:
    connection = db(); ids = set(scope["source_ids"]); collections_ids = scope["collection_ids"]
    if collections_ids:
        marks = ",".join("?"*len(collections_ids)); ids.update(row[0] for row in connection.execute(f"SELECT source_id FROM collection_sources WHERE collection_id IN ({marks})", collections_ids))
    if not ids: rows = []
    else:
        vector_ids = vector_query(query, list(ids), 6)
        marks = ",".join("?"*len(ids)); rows = connection.execute(f"SELECT c.*,s.title,s.source_type,s.canonical_url FROM chunks c JOIN source_versions v ON v.id=c.source_version_id AND v.active=1 JOIN sources s ON s.id=c.source_id WHERE c.source_id IN ({marks}) AND s.state='ready' AND s.deleted_at IS NULL", list(ids)).fetchall()
        rank = {chunk_id: index for index, chunk_id in enumerate(vector_ids)}
        rows = sorted(rows, key=lambda row: rank.get(row["id"], 9999))[:6]
    connection.close(); return rows


async def answer(chat_id: str, payload: ChatMessageIn) -> tuple[str, list[dict[str, Any]]]:
    chat = chat_row(chat_id); rows = retrieve(chat["scope"], payload.content)
    if not rows: return "I couldn’t find enough support in the selected sources.", []
    citations = [{"id": row["id"], "source_version_id": row["source_version_id"], "title": row["title"], "source_type": row["source_type"], "url": row["canonical_url"], "location": json.loads(row["location"]), "excerpt": row["text"][:300], "score": 1.0/(index+1)} for index,row in enumerate(rows)]
    content = grounded_completion(payload.content, [f"[{i+1}] {row['text']}" for i,row in enumerate(rows)])
    return content, citations


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, payload: ChatMessageIn) -> StreamingResponse:
    chat_row(chat_id); message_id = uuid.uuid4().hex; timestamp = now(); connection = db(); connection.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (uuid.uuid4().hex,chat_id,"user",payload.content,"{}","complete",len(payload.content.split()),timestamp)); connection.commit(); connection.close(); content, citations = await answer(chat_id, payload); connection = db(); connection.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (message_id,chat_id,"assistant",content,json.dumps({"provider":"openai-or-extractive"}),"complete",len(content.split()),now())); [connection.execute("INSERT INTO message_citations VALUES (?,?,?,?,?,?,?)", (uuid.uuid4().hex,message_id, c["source_version_id"], c["id"], c["excerpt"], c["score"], i+1)) for i,c in enumerate(citations)]; connection.commit(); connection.close()
    async def events():
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
        for part in re.findall(r".{1,100}(?:\s+|$)", content): yield f"event: token\ndata: {json.dumps({'text': part})}\n\n"
        yield f"event: done\ndata: {json.dumps({'message_id': message_id})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/search")
async def search(q: str = "") -> list[dict[str, Any]]: return await sources(search=q)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
