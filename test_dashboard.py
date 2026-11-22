import os
import tempfile
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient
import dashboard
import rag_mcp


class DashboardTest(unittest.TestCase):
    def test_dashboard_shows_collections_documents_and_activity(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}), patch.object(rag_mcp, "embed", side_effect=lambda texts, *args: [[1.0, 0.0] for _ in texts]):
            rag_mcp._client = None
            rag_mcp.create_collection("dashboard_test")
            rag_mcp.ingest_content("dashboard_test", "A stored document", "example")
            with TestClient(dashboard.app, base_url="http://localhost") as client:
                self.assertEqual(client.get("/").status_code, 200)
                overview = client.get("/api/overview").json()
                self.assertEqual(overview["collections"][0]["chunks"], 1)
                self.assertEqual(overview["logs"][0]["tool"], "ingest_content")
                self.assertNotIn("A stored document", str(overview["logs"]))
                documents = client.get("/api/collections/dashboard_test/documents").json()["documents"]
                self.assertEqual(documents[0]["chunks"][0]["text"], "A stored document")
                self.assertEqual(client.get("/api/overview", headers={"host": "example.com"}).status_code, 400)
            rag_mcp._client = None

    def test_setup_and_collection_model_isolation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}, clear=False), patch.object(rag_mcp, "embed", side_effect=lambda texts, *args: [[1.0, 0.0] for _ in texts]) as embed:
            rag_mcp._client = None
            with TestClient(dashboard.app, base_url="http://localhost") as client:
                headers = {"origin": "http://localhost"}
                self.assertEqual(client.post("/api/settings", json={"provider": "ollama", "model": "nomic-embed-text"}).status_code, 403)
                response = client.post("/api/settings", headers=headers, json={"provider": "ollama", "model": "nomic-embed-text"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(client.post("/api/create", headers=headers, json={"name": "local_model"}).status_code, 200)
                client.post("/api/settings", headers=headers, json={"provider": "openai", "model": "text-embedding-3-small", "api_key": "test-secret"})
                response = client.post("/api/ingest", headers=headers, json={"collection": "local_model", "kind": "text", "source": "note", "content": "test content"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(embed.call_args.args[1:], ("nomic-embed-text", "ollama"))
                response = client.post("/api/search", headers=headers, json={"collection": "local_model", "query": "test", "mode": "semantic"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(embed.call_args.args[1:], ("nomic-embed-text", "ollama"))
                self.assertNotIn("test-secret", client.get("/api/overview").text)
            rag_mcp._client = None

    def test_llm_settings_and_grounded_answer(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}), patch.object(rag_mcp, "embed", side_effect=lambda texts, *args: [[1.0, 0.0] for _ in texts]), patch.object(rag_mcp.httpx, "post") as post:
            rag_mcp._client = None
            post.return_value.json.return_value = {"message": {"content": "An answer [Source 1]"}}
            with TestClient(dashboard.app, base_url="http://localhost") as client:
                headers = {"origin": "http://localhost"}
                client.post("/api/create", headers=headers, json={"name": "answer_test"})
                client.post("/api/ingest", headers=headers, json={"collection": "answer_test", "kind": "text", "source": "notes", "content": "A relevant fact"})
                self.assertEqual(client.post("/api/llm-settings", json={"provider": "ollama", "model": "llama3.2"}).status_code, 403)
                self.assertEqual(client.post("/api/llm-settings", headers=headers, json={"provider": "ollama", "model": "llama3.2"}).status_code, 200)
                response = client.post("/api/answer", headers=headers, json={"collection": "answer_test", "question": "What fact?"})
                self.assertEqual(response.status_code, 200)
                result = response.json()["result"]
                self.assertEqual(result["answer"], "An answer [Source 1]")
                self.assertEqual(result["sources"][0]["metadata"]["source"], "notes")
                self.assertEqual(post.call_args.kwargs["json"]["model"], "llama3.2")
                self.assertIn("A relevant fact", post.call_args.kwargs["json"]["messages"][1]["content"])
                overview = client.get("/api/overview").json()
                self.assertEqual(overview["llm_model"], "llama3.2")
                self.assertEqual(overview["collections"][0]["metadata"]["embedding_model"], "text-embedding-3-small")
            rag_mcp._client = None

    def test_openai_answer_uses_configured_model(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}), patch.object(rag_mcp, "search", return_value=[{"text": "evidence", "metadata": {"source": "doc"}}]), patch.object(rag_mcp, "OpenAI") as openai:
            import settings
            settings.save("ollama", "nomic-embed-text", "example-key")
            settings.save_llm("openai", "custom-model")
            openai.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Result [Source 1]"))])
            self.assertEqual(rag_mcp.answer_question("collection", "question")["answer"], "Result [Source 1]")
            self.assertEqual(openai.return_value.chat.completions.create.call_args.kwargs["model"], "custom-model")

    def test_chunk_management_and_server_defaults(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}), patch.object(rag_mcp, "embed", side_effect=lambda texts, *args: [[1.0, 0.0] for _ in texts]):
            rag_mcp._client = None

            with TestClient(dashboard.app, base_url="http://localhost") as client:
                headers = {"origin": "http://localhost"}
                client.post("/api/create", headers=headers, json={"name": "chunk_test"})
                client.post("/api/ingest", headers=headers, json={"collection": "chunk_test", "kind": "text", "source": "notes", "content": "Original text"})
                chunk = client.get("/api/collections/chunk_test/documents").json()["documents"][0]["chunks"][0]
                self.assertIn("id", chunk)
                response = client.post("/api/update-chunk", headers=headers, json={"collection": "chunk_test", "chunk_id": chunk["id"], "text": "Updated text"})
                self.assertEqual(response.status_code, 200)
                documents = client.get("/api/collections/chunk_test/documents").json()["documents"]
                self.assertEqual(documents[0]["chunks"][0]["text"], "Updated text")
                self.assertEqual(client.post("/api/update-chunk", headers=headers, json={"collection": "chunk_test", "chunk_id": chunk["id"], "text": "  "}).status_code, 400)
                self.assertEqual(client.post("/api/server-settings", headers=headers, json={"mode": "hybrid", "limit": 3}).status_code, 200)
                self.assertEqual(client.post("/api/server-settings", headers=headers, json={"mode": "invalid", "limit": 3}).status_code, 400)
                overview = client.get("/api/overview").json()
                self.assertEqual(overview["search_mode"], "hybrid")
                self.assertEqual(overview["search_limit"], 3)
                self.assertTrue(rag_mcp.search("chunk_test", "updated"))
                self.assertEqual(client.post("/api/delete-chunk", headers=headers, json={"collection": "chunk_test", "chunk_id": chunk["id"]}).status_code, 200)
                self.assertEqual(client.get("/api/collections/chunk_test/documents").json()["documents"], [])
                client.post("/api/ingest", headers=headers, json={"collection": "chunk_test", "kind": "text", "source": "second", "content": "Second document"})
                document_id = client.get("/api/collections/chunk_test/documents").json()["documents"][0]["id"]
                self.assertEqual(client.post("/api/delete-document", headers=headers, json={"collection": "chunk_test", "document_id": document_id}).status_code, 200)
                self.assertEqual(client.get("/api/collections/chunk_test/documents").json()["documents"], [])
                self.assertEqual(client.post("/api/delete-collection", headers=headers, json={"name": "chunk_test"}).status_code, 200)
                self.assertEqual(client.get("/api/overview").json()["collections"], [])
            rag_mcp._client = None

    def test_search_forwards_requested_limit_and_defaults(self):
        with TestClient(dashboard.app, base_url="http://localhost") as client, patch.object(rag_mcp, "search", return_value=[]) as search:
            headers = {"origin": "http://localhost"}
            response = client.post("/api/search", headers=headers, json={"collection": "notes", "query": "hello", "mode": "hybrid", "limit": 3})
            self.assertEqual(response.status_code, 200)
            search.assert_called_with("notes", "hello", limit=3, mode="hybrid")
            client.post("/api/search", headers=headers, json={"collection": "notes", "query": "hello"})
            search.assert_called_with("notes", "hello", limit=None, mode=None)

    def test_mcp_http_initialization_and_tool_discovery(self):
        import json
        with TestClient(dashboard.app, base_url="http://localhost:8765") as client:
            headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
            response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
            self.assertEqual(response.status_code, 200)
            session = response.headers["mcp-session-id"]
            headers["mcp-session-id"] = session
            client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            self.assertEqual(response.status_code, 200)
            event = next(line[6:] for line in response.text.splitlines() if line.startswith("data: "))
            names = [tool["name"] for tool in json.loads(event)["result"]["tools"]]
            self.assertIn("search", names)
            self.assertIn("answer_question", names)
            denied = client.post("/mcp", headers={**headers, "origin": "https://evil.example"}, json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
            self.assertEqual(denied.status_code, 403)


if __name__ == "__main__":
    unittest.main()
