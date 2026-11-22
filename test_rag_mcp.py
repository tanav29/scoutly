import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import rag_mcp as server


class RagServerTest(unittest.TestCase):
    def test_image_ocr_and_missing_tesseract(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'scan.png'
            image.write_bytes(b'image')
            with patch.object(server.subprocess, 'run') as run:
                run.return_value.stdout = b'recognized text'
                self.assertEqual(server.extract_file(image), 'recognized text')
                self.assertEqual(run.call_args.args[0][1], str(image))
            with patch.object(server.subprocess, 'run', side_effect=FileNotFoundError):
                with self.assertRaisesRegex(ValueError, 'Tesseract'):
                    server.extract_file(image)

    def test_scanned_pdf_page_uses_ocr(self):
        try:
            import pypdfium2  # noqa: F401
        except ImportError:
            self.skipTest('PDF OCR extra is not installed')
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'scan.pdf'
            image = Image.new('RGB', (100, 100), 'white')
            try:
                image.save(file, 'PDF')
            finally:
                image.close()
            with patch.object(server, 'ocr_image', return_value='scanned text') as ocr:
                self.assertEqual(server.extract_file(file), 'scanned text')
                ocr.assert_called_once()

    def test_file_formats_and_folder_errors(self):
        from docx import Document
        from openpyxl import Workbook
        from pptx import Presentation
        from pptx.util import Inches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = Document()
            doc.add_paragraph('word content')
            doc.add_table(rows=1, cols=1).cell(0, 0).text = 'table content'
            doc.save(root / 'a.docx')
            book = Workbook()
            book.active['A1'] = 'spreadsheet content'
            book.save(root / 'b.xlsx')
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])
            slide.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text = 'slide content'
            presentation.save(root / 'c.pptx')
            (root / 'd.html').write_text('<p>visible</p><script>hidden</script>', encoding='utf-8')
            (root / 'e.csv').write_text('name,value\napple,2', encoding='utf-8')
            (root / 'bad.docx').write_text('broken', encoding='utf-8')
            self.assertIn('table content', server.extract_file(root / 'a.docx'))
            self.assertIn('spreadsheet content', server.extract_file(root / 'b.xlsx'))
            self.assertIn('slide content', server.extract_file(root / 'c.pptx'))
            self.assertNotIn('hidden', server.extract_file(root / 'd.html'))
            self.assertIn('apple,2', server.extract_file(root / 'e.csv'))
            with patch.object(server, 'ingest_text', side_effect=lambda collection, text, source, metadata: {'source': source}):
                result = server.ingest_folder('test', str(root))
            self.assertEqual(len(result['documents']), 5)
            self.assertEqual(len(result['errors']), 1)

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

