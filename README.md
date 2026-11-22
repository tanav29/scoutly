# VectorVault MCP

A local stdio MCP server for document retrieval. Install with `uv sync --locked`, set `OPENAI_API_KEY`, then start with `uv run --locked python rag_mcp.py`. The server sends document chunks and search queries to OpenAI for embeddings using `text-embedding-3-small`; API usage incurs charges.

Configure an MCP client with `uv` as the command and `run`, `--directory`, the absolute project path, `--locked`, `python`, `rag_mcp.py` as arguments. Supply `OPENAI_API_KEY` in the client's environment.

Tools include collection creation, listing, inspection and deletion; file, folder, URL and direct text ingestion; document listing, retrieval, deletion and metadata updates; and semantic or hybrid search with exact scalar metadata filters. Supported files are UTF-8 text, Markdown and PDF; URLs must return HTML or plain text. Documents use a hash of their source path or URL as their ID, and re-ingestion replaces existing chunks. Metadata keys `document_id`, `source`, and `ordinal` are reserved.

Data persists in `data/vectorvault` by default; set `VECTORVAULT_DATA_DIR` to change it. Collections created with the earlier hash embeddings are incompatible: create a new collection and re-ingest the source documents. Hybrid search scans the filtered collection in memory and is intended for small datasets. The server can read local files and fetch URLs accessible to its process, so connect only trusted MCP clients. Collection and document deletion are permanent.

Run tests with `uv run --locked python -m unittest test_rag_mcp -v`. Tests mock the embedding API and do not require an API key.

## Dashboard

Run `uv run --locked python dashboard.py` and open http://127.0.0.1:8765. The dashboard is read-only and shows collections, document chunks, storage settings, and the latest 200 MCP tool outcomes. It uses the same `VECTORVAULT_DATA_DIR` as the MCP server; set that variable consistently for both processes. Refresh to see new activity.

Activity logging begins when this version runs. Logs contain tool names, timestamps, outcomes and error types, not arguments or document contents. They are stored in `activity.sqlite3` inside the data directory. Dashboard access exposes document contents to anyone who can access the local server, so it binds only to localhost.
