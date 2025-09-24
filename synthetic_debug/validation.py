"""Utilities for validating and materialising conversation artefacts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from .pipeline import DebugConversation

Variant = Literal["correct", "buggy"]


class ValidationError(RuntimeError):
    """Raised when a conversation payload cannot be validated."""


def load_conversation(path: Path) -> DebugConversation:
    """Load a conversation JSON file into a DebugConversation instance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    # Gracefully allow missing buggy_code for file-edit tasks
    if "buggy_code" not in payload:
        payload["buggy_code"] = ""
    try:
        return DebugConversation(**payload)
    except TypeError as exc:
        # Attempt last-ditch recovery if buggy_code is still problematic
        payload.setdefault("buggy_code", "")
        try:
            return DebugConversation(**payload)
        except Exception:
            raise ValidationError(f"Invalid conversation payload in {path}: {exc}") from exc


def extract_code_artifacts(conversation: DebugConversation, destination: Path) -> None:
    """Write the code artefacts from a conversation to a destination directory."""

    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{conversation.module_name}_correct.py").write_text(
        conversation.correct_code,
        encoding="utf-8",
    )
    if conversation.buggy_code.strip():
        (destination / f"{conversation.module_name}_buggy.py").write_text(
            conversation.buggy_code,
            encoding="utf-8",
        )
    (destination / f"test_{conversation.module_name}.py").write_text(
        conversation.unit_tests,
        encoding="utf-8",
    )
    (destination / "run_tests.py").write_text(
        conversation.runner_code,
        encoding="utf-8",
    )


def run_tests_for_variant(conversation: DebugConversation, variant: Variant) -> subprocess.CompletedProcess[str]:
    """Execute the unit tests for either the correct or buggy implementation."""

    source = conversation.correct_code if variant == "correct" else conversation.buggy_code

    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir, f"{conversation.module_name}.py")
        tests_path = Path(tmpdir, f"test_{conversation.module_name}.py")
        runner_path = Path(tmpdir, "run_tests.py")

        module_path.write_text(source, encoding="utf-8")
        tests_path.write_text(conversation.unit_tests, encoding="utf-8")
        runner_path.write_text(conversation.runner_code, encoding="utf-8")

        return subprocess.run(
            [sys.executable, "run_tests.py"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False,
        )


__all__ = [
    "ValidationError",
    "load_conversation",
    "extract_code_artifacts",
    "run_tests_for_variant",
]
