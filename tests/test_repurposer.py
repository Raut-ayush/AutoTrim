import json
from pathlib import Path
import tempfile
import unittest

from repurposer import (
    MAX_OUTPUT_TOKENS,
    RepurposedContent,
    build_generation_config,
    build_prompt,
    save_outputs,
    target_word_range,
    validate_content,
)


class RepurposerTests(unittest.TestCase):
    def test_article_word_targets_scale_with_transcript(self):
        self.assertEqual(target_word_range(200), (250, 400))
        self.assertEqual(target_word_range(1_000), (500, 1_000))
        self.assertEqual(target_word_range(5_000), (900, 1_800))
        self.assertEqual(target_word_range(9_237), (1_500, 2_500))

    def test_prompt_includes_full_transcript_and_substack_target(self):
        transcript = " ".join(["distinctive transcript ending marker"] * 2_500)
        prompt = build_prompt(transcript)

        self.assertIn("FULL TRANSCRIPT", prompt)
        self.assertTrue(prompt.endswith(transcript))
        self.assertIn("1,500–2,500", prompt)
        self.assertNotIn("X thread", prompt)
        self.assertNotIn("Reddit post", prompt)

    def test_generation_config_disables_unused_afc(self):
        config = build_generation_config(0.3)

        self.assertTrue(config.automatic_function_calling.disable)
        self.assertEqual(config.max_output_tokens, MAX_OUTPUT_TOKENS)
        self.assertEqual(config.response_mime_type, "application/json")

    def test_validation_warns_when_article_is_below_target(self):
        data = RepurposedContent(
            substack_title="A title",
            substack_content="short article",
        )

        issues = validate_content(data, transcript_word_count=9_237)

        self.assertTrue(any("target is 1,500–2,500" in issue for issue in issues))

    def test_saves_only_substack_files_and_removes_old_social_outputs(self):
        data = RepurposedContent(
            substack_title="A title",
            substack_content="A detailed article.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for name in ("x_thread.txt", "reddit_post.txt", "x_thread.md", "reddit_post.md"):
                (output_dir / name).write_text("legacy", encoding="utf-8")

            text_path, json_path = save_outputs(data, output_dir)

            self.assertEqual(text_path.name, "substack_post.txt")
            self.assertEqual(json_path.name, "repurposed.json")
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"substack_post.txt", "repurposed.json"},
            )
            self.assertIn("# A title", text_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(json.loads(json_path.read_text(encoding="utf-8"))),
                {"substack_title", "substack_content"},
            )


if __name__ == "__main__":
    unittest.main()
