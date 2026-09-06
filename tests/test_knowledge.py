import tempfile
import unittest
from pathlib import Path

from dit_autoresearch.knowledge import build_index, search_index


class KnowledgeTests(unittest.TestCase):
    def test_knowledge_index(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            k = tmp_path / "knowledge"
            k.mkdir()
            (k / "a.md").write_text("# Attention on Ampere\nFlashAttention and SDPA matter for attention.")
            (k / "b.md").write_text("# Memory\nOffload and VAE tiling reduce memory.")
            db = tmp_path / "index.sqlite3"
            self.assertEqual(build_index(k, db), 2)
            rows = search_index(db, "Ampere attention", limit=2)
            self.assertTrue(rows)
            self.assertEqual(rows[0]["path"], "a.md")


if __name__ == "__main__":
    unittest.main()
