import json
from pathlib import Path
import tempfile
import unittest

from autotrim import find_existing_content


class ExistingContentTests(unittest.TestCase):
    def test_substack_only_output_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.txt"
            content_dir = Path(tmp) / "content"
            content_dir.mkdir()
            (content_dir / "substack_post.txt").write_text("article", encoding="utf-8")
            (content_dir / "repurposed.json").write_text(
                json.dumps({"substack_title": "Title", "substack_content": "Body"}),
                encoding="utf-8",
            )

            self.assertEqual(find_existing_content(transcript), content_dir)

    def test_legacy_social_output_is_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.txt"
            content_dir = Path(tmp) / "content"
            content_dir.mkdir()
            (content_dir / "substack_post.txt").write_text("article", encoding="utf-8")
            (content_dir / "repurposed.json").write_text(
                json.dumps({
                    "x_thread": ["tweet"],
                    "reddit_title": "Title",
                    "reddit_body": "Body",
                    "substack_title": "Title",
                    "substack_content": "Body",
                }),
                encoding="utf-8",
            )

            self.assertIsNone(find_existing_content(transcript))


if __name__ == "__main__":
    unittest.main()
