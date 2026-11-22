import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import rag_mcp as server


class RagServerTest(unittest.TestCase):
    @patch.object(server, "embed", side_effect=lambda texts, *args: [[1.0, 0.0] for _ in texts])
    def test_document_lifecycle(self, mock_embed):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            os.environ['VECTORVAULT_DATA_DIR'] = directory
            server._client = None
            server.create_collection('test_collection')
            path = Path(directory) / 'doc.txt'
            path.write_text('apples and oranges ' * 200, encoding='utf-8')
            doc = server.ingest_document('test_collection', str(path), {'kind': 'fruit'})
            self.assertGreater(doc['chunks'], 1)
            self.assertTrue(server.search('test_collection', 'apples', filters={'kind': 'fruit'}))
            self.assertTrue(server.search('test_collection', 'apples', mode='hybrid'))
            server.update_metadata('test_collection', doc['document_id'], {'kind': 'food'})
            self.assertFalse(server.search('test_collection', 'apples', filters={'kind': 'fruit'}))
            path.write_text('short replacement', encoding='utf-8')
            server.ingest_document('test_collection', str(path))
            self.assertEqual(server.get_document('test_collection', doc['document_id'])['chunks'], ['short replacement'])
            self.assertEqual(server.collection_info('test_collection')['chunks'], 1)
            self.assertEqual(server.list_documents('test_collection')['total'], 1)
            server.delete_document('test_collection', doc['document_id'])
            self.assertEqual(server.list_documents('test_collection')['total'], 0)
            server.ingest_content('test_collection', 'agent supplied content', 'agent:note')
            self.assertEqual(server.list_documents('test_collection')['total'], 1)
            self.assertTrue(mock_embed.called)
            server.delete_collection('test_collection')
            self.assertNotIn('test_collection', server.list_collections())
            server._client = None


if __name__ == '__main__':
    unittest.main()

