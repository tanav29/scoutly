# VectorVault MCP

VectorVault is a local tool for building a retrieval-augmented generation (RAG) pipeline and giving an agent access to it through the Model Context Protocol (MCP). Its dashboard manages document collections and tests retrieval; its MCP server exposes those capabilities as tools to compatible agents.

The pipeline extracts text from sources, splits it into overlapping chunks, embeds and stores those chunks in local collections, then retrieves relevant excerpts for a query. An optional answer model uses those excerpts to generate an answer with sources. Agents can also manage collections and documents directly through MCP.

The server and collection storage run locally. Model processing can also be local with Ollama; selecting OpenAI sends content or queries to OpenAI, and importing a URL fetches it over the network.

## Start VectorVault

From the project directory, install dependencies once:

```powershell
uv sync --locked
```

Start the dashboard and MCP server:

```powershell
uv run --locked python dashboard.py
```

Leave the terminal running while using VectorVault. Press `Ctrl+C` to stop it.

Open http://127.0.0.1:8765. The dashboard and Streamable HTTP MCP server run together; the MCP endpoint is http://127.0.0.1:8765/mcp. Copy the agent setup prompt from the dashboard. Choose an embedding provider and model, create a collection, add sources, and test retrieval from the dashboard. Configure an answer model separately to ask questions against a collection and inspect the retrieved sources. Local files and folders must be accessible to the server. Supported local file formats are listed below; URLs must return HTML or plain text.

Choose **Ollama** for local embeddings or answers; install each selected model separately. The answer model uses Ollama or OpenAI independently of the collection's embedding model. For OpenAI answers, the shared OpenAI API key is configured in embedding settings or via `OPENAI_API_KEY`; questions and retrieved excerpts are sent to OpenAI. Generated answers can be inaccurate, so review their sources. Choose **Ollama** for local embeddings. Ollama must be running at `http://127.0.0.1:11434`, and the chosen embedding model must already be installed (for example, run `ollama pull nomic-embed-text`). Choose **OpenAI** to use an OpenAI embedding model; document text and queries are sent to OpenAI and API usage may incur charges. Enter an API key in the dashboard or set `OPENAI_API_KEY` in both processes. A dashboard-saved key takes precedence over the environment variable and is stored **unencrypted** in `settings.json` inside the data directory; protect access to that directory. To remove a saved key, remove the `api_key` value from that local file.

Settings affect newly created collections. Each collection keeps its original provider and model for ingestion and search, even when the default changes. Collections from the old token-hash version have no known model and must be recreated and re-ingested. Changing a model does not migrate or re-embed documents.

## Connect an MCP client

Local imports support PDF, PNG, JPEG, TIFF, BMP, WebP, DOCX, PPTX, XLSX, HTML, and UTF-8 text formats: TXT, Markdown, CSV, TSV, JSON, JSONL, XML, LOG, and RST. Folder imports recursively discover these extensions and report individual failures. Office files are parsed locally with lightweight libraries. Spreadsheet formulas use the cached values saved by the spreadsheet application. Legacy Office formats such as DOC and XLS are not supported; save them in the newer formats first.

For OCR, install Tesseract and its language data separately and ensure `tesseract` is on PATH, or set `VECTORVAULT_TESSERACT` to its executable path. For scanned PDF pages also run `uv sync --locked --extra ocr` to install PDF rendering support. Image files are OCRed locally; PDF pages with embedded text use that text, while pages without text that contain images are rendered and OCRed. OCR defaults to English; set `VECTORVAULT_OCR_LANG` to an installed Tesseract language code (or codes joined by `+`). Pages containing both selectable text and scanned images are not OCRed. OCR quality depends on image quality and language data.

When starting with `uv run`, include the extra so it remains installed: `uv run --locked --extra ocr python dashboard.py`. Use the same `--extra ocr` arguments for the stdio MCP command if it needs scanned PDF support.

For a client that supports Streamable HTTP, configure a server named `vectorvault` with URL `http://127.0.0.1:8765/mcp`. Keep `dashboard.py` running while the client is connected. The dashboard offers a copyable prompt to help an agent configure this connection. MCP client configuration formats vary by application. The HTTP endpoint is local and unauthenticated; only connect trusted clients and do not expose the port to a network.

For clients that only support stdio, configure the command `uv` with arguments `run`, `--directory`, the absolute project path, `--locked`, `python`, `rag_mcp.py`. Set `VECTORVAULT_DATA_DIR` consistently for separate processes and pass `OPENAI_API_KEY` to the client if needed.

MCP tools provide collection management, file/folder/URL/text ingestion, document retrieval and metadata updates, semantic or hybrid search, and `answer_question` for retrieval-backed answers. Hybrid search scans the filtered collection in memory and is intended for small datasets. Deletion is permanent.

## Data and activity

The default data directory is `data/vectorvault`. The dashboard binds to localhost and exposes stored document contents and settings controls to local users. Connect only trusted MCP clients: they can read local files and fetch URLs accessible to the server. Activity records tool names, timestamps, outcomes and error types, not arguments or document contents; it begins when logging is enabled and displays the latest 200 calls. Refresh the dashboard to see new activity.

Run checks with `uv run --locked python -m unittest test_rag_mcp test_dashboard -v`. Tests mock embedding requests and do not require an API key or Ollama.
