import os
import tempfile
import unittest
from pathlib import Path

import rag_mcp as server


class RagServerTest(unittest.TestCase):
    def test_document_lifecycle(self):
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
            server.delete_collection('test_collection')
            self.assertNotIn('test_collection', server.list_collections())
            server._client = None


if __name__ == '__main__':
    unittest.main()

