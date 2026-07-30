# 🧠 Scoutly - Research Agent

A smart research assistant that:
- 🔍 Searches DuckDuckGo for **5 keywords**
- 🌐 Scrapes **top 7 results** per keyword
- 📚 Builds a **RAG pipeline** for intelligent Q&A
- ⚡ Delivers **context-rich answers** with sources

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Multi-Keyword Search** | Searches 5 keywords simultaneously |
| **Smart Scraping** | Extracts top 7 results from DuckDuckGo |
| **RAG Pipeline** | Vector database + LLM for accurate answers |
| **Source Attribution** | Every answer cites original sources |
| **Async Processing** | Fast parallel scraping and processing |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Search** | DuckDuckGo |
| **Scraping** | Selectolax |
| **Vector DB** | Chroma |
| **Embeddings** | Embedding Gemma |
| **LLM** | Gemma3 |
| **Framework** | LangChain / Docling |

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/research-agent
cd research-agent
pip install -r requirements.txt

# Run with your keywords
streamlit run app.py
```

## Web app

Run the FastAPI backend and Next.js UI in separate terminals:

```bash
uv run uvicorn api:app --reload --port 8000
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The web app provides a single bootstrap company
workspace with durable sources, background ingestion, persistent private chats,
saved source scope, webpage/YouTube ingestion, and citation-linked grounded
answers. Local development stores application data in `data/` and originals in
`data/objects/`; set `SCOUTLY_DB_PATH`, `SCOUTLY_DATA_DIR`, and
`OPENAI_API_KEY` as needed. Without an OpenAI key, the safe extractive provider
is used for local development. API documentation is available at
`http://localhost:8000/docs`.

The current Phase 1–3 API includes `/api/workspace`, collections, source upload
and URL endpoints, source status/retry/delete, persistent chats and scopes,
library search, and SSE chat events (`token`, `citations`, `done`). The schema
already carries `workspace_id`, source versions, ingestion jobs, audit events,
usage counters, and citation provenance for the next production adapters.
