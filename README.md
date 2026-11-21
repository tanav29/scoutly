# VectorVault MCP

A local stdio MCP server for retrieval over named collections. Start it with `uv run --locked python rag_mcp.py`. Configure an MCP client with command `uv` and arguments `run`, `--directory`, the absolute project path, `--locked`, `python`, `rag_mcp.py`.

Tools: `create_collection`, `list_collections`, `delete_collection`, `ingest_document`, `ingest_folder`, `ingest_url`, `search`, `get_document`, and `update_metadata`. Ingestion supports UTF-8 text, Markdown, PDF, and HTML/plain-text URLs. Search supports `semantic` and `hybrid` modes and exact scalar metadata filters. Documents are identified by a hash of their source path or URL; re-ingestion replaces their chunks. Metadata keys `document_id`, `source`, and `ordinal` are reserved. Deleting a collection is permanent.

Data persists in `data/vectorvault` by default; set `VECTORVAULT_DATA_DIR` to change it. Embeddings are deterministic offline token hashes, so semantic similarity is approximate and does not use a trained language model. Hybrid search scans the filtered collection in memory and is intended for small datasets. The server can read local files and fetch URLs accessible to its process; only connect trusted MCP clients.

Run the lifecycle checks with `uv run --locked python -m unittest test_rag_mcp -v`.

## Legacy application

# ðŸ§  Scoutly - Research Agent

A smart research assistant that:
- ðŸ” Searches DuckDuckGo for **5 keywords**
- ðŸŒ Scrapes **top 7 results** per keyword
- ðŸ“š Builds a **RAG pipeline** for intelligent Q&A
- âš¡ Delivers **context-rich answers** with sources

## ðŸŽ¯ Features

| Feature | Description |
|---------|-------------|
| **Multi-Keyword Search** | Searches 5 keywords simultaneously |
| **Smart Scraping** | Extracts top 7 results from DuckDuckGo |
| **RAG Pipeline** | Vector database + LLM for accurate answers |
| **Source Attribution** | Every answer cites original sources |
| **Async Processing** | Fast parallel scraping and processing |

## ðŸ› ï¸ Tech Stack

| Component | Technology |
|-----------|------------|
| **Search** | DuckDuckGo |
| **Scraping** | Selectolax |
| **Vector DB** | Chroma |
| **Embeddings** | Embedding Gemma |
| **LLM** | Gemma3 |
| **Framework** | LangChain / Docling |

## ðŸš€ Quick Start

```bash
# Clone and install
git clone https://github.com/tanav29/scoutly.git
cd scoutly
uv sync --locked

# Run with your keywords
uv run streamlit run app.py
```

## Web app

Run the FastAPI backend and Next.js UI in separate terminals:

```bash
uv run --locked uvicorn api:app --reload --port 8000
cd web
bun install --frozen-lockfile
bun run dev
```

Open `http://localhost:3000`. The web app provides a single bootstrap company
workspace with durable sources, background ingestion, persistent private chats,
saved source scope, webpage/YouTube ingestion, and citation-linked grounded
answers. Local development stores application data in `data/` and originals in
`data/objects/`; set `SCOUTLY_DB_PATH`, `SCOUTLY_DATA_DIR`, and
`OPENAI_API_KEY` as needed. Without an OpenAI key, the safe extractive provider
is used for local development. API documentation is available at
`http://localhost:8000/docs`.

The current Phase 1â€“3 API includes `/api/workspace`, collections, source upload
and URL endpoints, source status/retry/delete, persistent chats and scopes,
library search, and SSE chat events (`token`, `citations`, `done`). The schema
already carries `workspace_id`, source versions, ingestion jobs, audit events,
usage counters, and citation provenance for the next production adapters.

