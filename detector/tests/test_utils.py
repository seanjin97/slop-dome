import importlib.util
from pathlib import Path
import unittest


# Import the utility directly so these tests do not load the detector model.
utils_path = Path(__file__).resolve().parents[1] / "src" / "detector" / "utils.py"
spec = importlib.util.spec_from_file_location("detector_utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class TextSignalsTests(unittest.TestCase):
    def test_combined_flags(self):
        text = "Certainly! Here’s **why** — not just to explain but to delve into it."
        self.assertEqual(
            utils.common_identifiers(text),
            {
                "em_dash": True,
                "markdown": True,
                "ai_phrase": True,
                "contrast_phrase": True,
                "ai_opening": True,
            },
        )

    def test_plain_text_has_no_flags(self):
        for text in ("", "I can meet at the cafe on Friday.\n", "Café\t東京\r\n"):
            with self.subTest(text=text):
                self.assertFalse(any(utils.common_identifiers(text).values()))

    def test_only_em_dash_sets_em_dash_flag(self):
        self.assertTrue(utils.common_identifiers("One — two")["em_dash"])
        self.assertFalse(utils.common_identifiers("One – two - three")["em_dash"])

    def test_common_markdown_formats(self):
        for text in (
            "a **bold** word",
            "__bold__",
            "Text\n## Heading",
            "> Quote",
            "```python\npass\n```",
            "- Bullet",
            "1. Item",
            "Use `code`",
            "[Link](https://example.com)",
        ):
            with self.subTest(text=text):
                self.assertTrue(utils.common_identifiers(text)["markdown"])
        for text in ("3 * 4", "snake_case", "Unclosed **bold", "[Unclosed [link"):
            with self.subTest(text=text):
                self.assertFalse(utils.common_identifiers(text)["markdown"])

    def test_phrases_are_case_insensitive_and_use_word_boundaries(self):
        for text in ("Let's DELVE INTO this.", "It’s worth noting.", "A tapestry of ideas."):
            with self.subTest(text=text):
                self.assertTrue(utils.common_identifiers(text)["ai_phrase"])
        self.assertFalse(utils.common_identifiers("tapestrylike")["ai_phrase"])

    def test_contrast_stays_within_a_sentence(self):
        for text in ("Not just fast, but reliable.", "Not only fast but also reliable."):
            with self.subTest(text=text):
                self.assertTrue(utils.common_identifiers(text)["contrast_phrase"])
        for text in ("Not just fast. But reliable.", "Not only fast."):
            with self.subTest(text=text):
                self.assertFalse(utils.common_identifiers(text)["contrast_phrase"])

    def test_unusual_unicode_and_controls(self):
        for char in ("’", "😀", "\u200b", "\u00a0", "\x00"):
            with self.subTest(char=char):
                self.assertTrue(utils._is_unusual_unicode(char))
        self.assertFalse(utils._is_unusual_unicode("\u0301"))

    def test_opening_must_start_the_text(self):
        for text in ("  Certainly! Here's why.", "Absolutely. Here is the answer.", "Here’s a summary."):
            with self.subTest(text=text):
                self.assertTrue(utils.common_identifiers(text)["ai_opening"])
        self.assertFalse(utils.common_identifiers("She said, certainly! Here's why.")["ai_opening"])

    def test_sanitise_preserves_existing_conversion(self):
        self.assertEqual(utils.sanitise("Café — it’s open."), "Cafe - it`s open.")

    def test_is_ai_text_returns_true_for_any_flag(self):
        cases = (
            ("Plain text.", 0, False),
            ("A **word** in a sentence.", 1, True),
            ("Here's a **word** — in a sentence.", 3, True),
        )
        for text, flag_count, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(sum(utils.common_identifiers(text).values()), flag_count)
                self.assertIs(utils.is_ai_text(text), expected)
