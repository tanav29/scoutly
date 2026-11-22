import os
import tempfile
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient
import dashboard
import rag_mcp


class DashboardTest(unittest.TestCase):
    def test_dashboard_shows_collections_documents_and_activity(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory, patch.dict(os.environ, {"VECTORVAULT_DATA_DIR": directory}), patch.object(rag_mcp, "embed", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]):
            rag_mcp._client = None
            rag_mcp.create_collection("dashboard_test")
            rag_mcp.ingest_content("dashboard_test", "A stored document", "example")
            with TestClient(dashboard.app) as client:
                self.assertEqual(client.get("/").status_code, 200)
                overview = client.get("/api/overview").json()
                self.assertEqual(overview["collections"][0]["chunks"], 1)
                self.assertEqual(overview["logs"][0]["tool"], "ingest_content")
                self.assertNotIn("A stored document", str(overview["logs"]))
                documents = client.get("/api/collections/dashboard_test/documents").json()["documents"]
                self.assertEqual(documents[0]["chunks"][0]["text"], "A stored document")
                self.assertEqual(client.get("/api/overview", headers={"host": "example.com"}).status_code, 400)
            rag_mcp._client = None


if __name__ == "__main__":
    unittest.main()
