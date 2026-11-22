# VectorVault MCP

VectorVault provides a local dashboard and an MCP server for organizing and searching documents with embeddings.

## Start the dashboard

```powershell
uv sync --locked
uv run --locked python dashboard.py
```

Open http://127.0.0.1:8765. Choose an embedding provider and model, create a collection, add sources, and test retrieval from the dashboard. Configure an answer model separately to ask questions against a collection and inspect the retrieved sources. Local files and folders must be accessible to the server. Supported files include UTF-8 text, Markdown and PDF; URLs must return HTML or plain text.

Choose **Ollama** for local embeddings or answers; install each selected model separately. The answer model uses Ollama or OpenAI independently of the collection?s embedding model. For OpenAI answers, the shared OpenAI API key is configured in embedding settings or via `OPENAI_API_KEY`; questions and retrieved excerpts are sent to OpenAI. Generated answers can be inaccurate, so review their sources. Choose **Ollama** for local embeddings. Ollama must be running at `http://127.0.0.1:11434`, and the chosen embedding model must already be installed (for example, run `ollama pull nomic-embed-text`). Choose **OpenAI** to use an OpenAI embedding model; document text and queries are sent to OpenAI and API usage may incur charges. Enter an API key in the dashboard or set `OPENAI_API_KEY` in both processes. A dashboard-saved key takes precedence over the environment variable and is stored **unencrypted** in `settings.json` inside the data directory; protect access to that directory. To remove a saved key, remove the `api_key` value from that local file.

Settings affect newly created collections. Each collection keeps its original provider and model for ingestion and search, even when the default changes. Collections from the old token-hash version have no known model and must be recreated and re-ingested. Changing a model does not migrate or re-embed documents.

## Connect an MCP client

Configure your client to run `uv` with arguments `run`, `--directory`, the absolute project path, `--locked`, `python`, `rag_mcp.py`. The MCP server and dashboard read the same settings from the data directory; if you set `VECTORVAULT_DATA_DIR`, use the same value in both processes. Without a saved key, pass `OPENAI_API_KEY` to the MCP client's environment when using OpenAI.

MCP tools provide collection management, file/folder/URL/text ingestion, document retrieval and metadata updates, semantic or hybrid search, and `answer_question` for retrieval-backed answers. Hybrid search scans the filtered collection in memory and is intended for small datasets. Deletion is permanent.

## Data and activity

The default data directory is `data/vectorvault`. The dashboard binds to localhost and exposes stored document contents and settings controls to local users. Connect only trusted MCP clients: they can read local files and fetch URLs accessible to the server. Activity records tool names, timestamps, outcomes and error types, not arguments or document contents; it begins when logging is enabled and displays the latest 200 calls. Refresh the dashboard to see new activity.

Run checks with `uv run --locked python -m unittest test_rag_mcp test_dashboard -v`. Tests mock embedding requests and do not require an API key or Ollama.
