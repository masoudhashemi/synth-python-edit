import json
import tempfile
import unittest
from pathlib import Path

from synthetic_debug.catalog import ScenarioCatalog
from synthetic_debug.pipeline import DebugConversationPipeline, GeneratedSpec, LLMGenerator, ScenarioSeed


class FakeLLM(LLMGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate_spec(self, seed: ScenarioSeed) -> GeneratedSpec:
        self.calls += 1
        assert seed.domain == "Software Engineering"
        assert seed.topic == "String utilities"
        assert seed.subtopic == "Palindrome detection"
        return GeneratedSpec(
            domain=seed.domain,
            topic=seed.topic,
            subtopic=seed.subtopic,
            summary="Detect palindromes ignoring case and non-alphanumerics.",
            module_name="string_tools",
            problem_description=(
                "Provide utilities for string analysis. The primary function should\n"
                "determine whether an input is a palindrome after normalizing case\n"
                "and removing non-alphanumeric characters."
            ),
            solution_outline=(
                "1. Normalize by filtering to str.isalnum and lowercasing.\n"
                "2. Compare the normalized string to its reverse.\n"
            ),
            correct_code='''"""String utilities for analysis."""
from typing import Iterable


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def is_palindrome(text: str) -> bool:
    cleaned = _normalize(text)
    return cleaned == cleaned[::-1]
''',
            unit_tests='''import unittest

from string_tools import is_palindrome


class PalindromeTests(unittest.TestCase):
    def test_simple(self) -> None:
        self.assertTrue(is_palindrome("racecar"))

    def test_phrase(self) -> None:
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama!"))

    def test_negative(self) -> None:
        self.assertFalse(is_palindrome("hello"))


if __name__ == "__main__":
    unittest.main()
''',
        )


class FileEditTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_llm = FakeLLM()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        catalog_path = Path(self.tmpdir.name) / "catalog.json"
        catalog_path.write_text(
            json.dumps(
                [
                    {
                        "domain": "Software Engineering",
                        "topic": "String utilities",
                        "subtopic": "Palindrome detection",
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.catalog = ScenarioCatalog(catalog_path, random_seed=42)
        self.pipeline = DebugConversationPipeline(llm=self.fake_llm, catalog=self.catalog)

    def test_generate_file_edit_task_writes_files_and_references_path(self) -> None:
        placeholder = Path(self.tmpdir.name) / "placeholder.py"
        convo = self.pipeline.generate_file_edit_task(bug_file_path=placeholder)

        # Should coerce filename to module_name.py
        expected_bug_path = Path(self.tmpdir.name) / "string_tools.py"
        self.assertEqual(Path(convo.bug_file_path), expected_bug_path)

        # Files exist on disk
        self.assertTrue(expected_bug_path.exists())
        self.assertTrue((expected_bug_path.parent / "test_string_tools.py").exists())
        self.assertTrue((expected_bug_path.parent / "run_tests.py").exists())

        # Conversation should reference paths and not inline buggy source
        payload_roles = [turn["role"] for turn in convo.conversation]
        self.assertIn("qa", payload_roles)
        combined_content = "\n".join(turn["content"] for turn in convo.conversation)
        self.assertIn(str(expected_bug_path), combined_content)
        self.assertNotIn("Injected defect version:", combined_content)

        # Failing output should indicate a failure
        self.assertIn("FAIL", convo.failing_output.upper())


if __name__ == "__main__":
    unittest.main()


