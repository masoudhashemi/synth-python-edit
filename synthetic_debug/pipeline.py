"""Dynamic synthetic debugging conversation pipeline using LiteLLM."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import ast
import random
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence
import litellm
import difflib
from .catalog import DEFAULT_CATALOG_PATH, ScenarioCatalog, ScenarioSeed
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


class LLMGenerator(Protocol):
    """Protocol for objects capable of returning debugging specs via an LLM."""

    def generate_spec(self, seed: ScenarioSeed) -> "GeneratedSpec":
        ...


@dataclass(frozen=True)
class GeneratedSpec:
    """Structured description returned by the language model."""

    domain: str
    topic: str
    subtopic: str
    summary: str
    module_name: str
    problem_description: str
    solution_outline: str
    correct_code: str
    unit_tests: str


@dataclass
class DebugConversation:
    """Final artefact produced by the pipeline."""

    domain: str
    topic: str
    subtopic: str
    summary: str
    problem_description: str
    solution_outline: str
    module_name: str
    correct_code: str
    buggy_code: str
    unit_tests: str
    runner_code: str
    failing_output: str
    conversation: List[Dict[str, str]]
    # Optional: when the buggy code is materialised to a real file for a file-edit task
    bug_file_path: Optional[str] = None
    # Optional: unified diff between correct and buggy code, included in JSON payloads
    code_diff: Optional[str] = None

    def to_json(self) -> str:
        correct_lines = self.correct_code.splitlines(keepends=True)
        buggy_lines = self.buggy_code.splitlines(keepends=True)
        diff = difflib.unified_diff(
            correct_lines,
            buggy_lines,
            fromfile="correct.py",
            tofile="buggy.py",
        )
        code_diff = "".join(diff).rstrip()

        return json.dumps(
            {
                "domain": self.domain,
                "topic": self.topic,
                "subtopic": self.subtopic,
                "summary": self.summary,
                "problem_description": self.problem_description,
                "solution_outline": self.solution_outline,
                "module_name": self.module_name,
                "correct_code": self.correct_code,
                "buggy_code": self.buggy_code,
                "unit_tests": self.unit_tests,
                "runner_code": self.runner_code,
                "failing_output": self.failing_output,
                "conversation": self.conversation,
                "code_diff": code_diff,
                "bug_file_path": self.bug_file_path,
            },
            indent=2,
            sort_keys=True,
        )


class LiteLLMGenerator:
    """Default generator that delegates content creation to LiteLLM."""

    SYSTEM_PROMPT = (
        "You are a senior Scientist very familiar with Python and implementing scientific software. "
        "Always respond with a single YAML document matching the requested schema."
    )

    USER_PROMPT = """
Create a unique STEM-focused debugging challenge. Use the domain, topic, and subtopic specified in the scenario constraints above. Respond with YAML matching this schema:
---
domain: <use the domain from scenario constraints>
topic: <use the topic from scenario constraints>
subtopic: <use the subtopic from scenario constraints>
summary: <one-sentence feature description>
module_name: <valid snake_case module name>
problem_description: |
  <multi-paragraph problem statement explaining the scenario and requirements>
solution_outline: |
  1. <first reasoning step>
  2. <second reasoning step>
  ...
correct_code: |
  <fully working Python module using only the standard library>
unit_tests: |
  <unittest-based test module referencing only the provided module>

Constraints:
- The problme must be solvable by a senior scientist with a PhD in the domain.
- The code must solve the problem in a way that is consistent with the domain.
- The correct code must pass the tests when saved to module_name.py.
- Use only the Python standard library and stay within 150 lines per code block.
- The module must include cooperating functions/classes with domain-relevant branching or aggregation logic.
- Provide docstrings or comments that clarify non-obvious logic.
- Tests must exercise happy path(s) and edge conditions, and clearly encode the behavioural contract.
- Do not include Markdown fences or additional commentary—only the YAML document.
"""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_attempts: int = 3,
    ) -> None:

        self._litellm = litellm
        self._model = model or os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
        self._temperature = temperature
        self._max_attempts = max_attempts

    def generate_spec(self, seed: ScenarioSeed) -> GeneratedSpec:
        last_error: Optional[Exception] = None
        for _ in range(self._max_attempts):
            try:
                content = self._invoke_llm(seed)
                data = _parse_yaml_document(content)
                return _spec_from_payload(data)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
        raise RuntimeError("Unable to obtain a valid JSON payload from LiteLLM") from last_error

    def _invoke_llm(self, seed: ScenarioSeed) -> str:
        scenario_context = (
            "You must honour the following scenario constraints:\n"
            f"- Domain: {seed.domain}\n"
            f"- Topic: {seed.topic}\n"
            f"- Subtopic: {seed.subtopic}\n"
            "Ensure the generated artefacts align with this context.\n"
        )
        response = self._litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": scenario_context + "\n" + self.USER_PROMPT,
                },
            ],
            temperature=self._temperature,
        )
        choice = response["choices"][0]
        return choice["message"]["content"]


class DebugConversationPipeline:
    """Pipeline orchestrating end-to-end debugging conversation generation."""

    def __init__(
        self,
        *,
        llm: Optional[LLMGenerator] = None,
        catalog: Optional[ScenarioCatalog] = None,
        bug_injector: Optional["BugInjector"] = None,
        num_llm_candidates: int = 1,  # Control initial LLM candidates for efficiency
    ) -> None:
        self._llm = llm or LiteLLMGenerator()
        self._catalog = catalog or ScenarioCatalog(DEFAULT_CATALOG_PATH)
        self._bug_injector = bug_injector or _default_bug_injector()
        self._num_llm_candidates = num_llm_candidates
        logger.info("Initialized DebugConversationPipeline")

    def generate(self, seed: Optional[ScenarioSeed] = None) -> DebugConversation:
        """Generate a debugging conversation with verified correct code and meaningful bug."""
        logger.info("Starting generation process")

        if seed is None:
            seed = self._catalog.acquire()
        logger.info(f"Acquired seed: {seed.domain} - {seed.topic} - {seed.subtopic}")

        try:
            max_retries = 3
            for attempt in range(max_retries):
                logger.info(f"Generating spec from LLM (attempt {attempt + 1})")
                spec = self._llm.generate_spec(seed)
                logger.info("Validating spec against seed")
                self._validate_spec_against_seed(spec, seed)
                runner_code = _build_test_runner()

                with tempfile.TemporaryDirectory() as tmpdir:
                    module_path = Path(tmpdir, f"{spec.module_name}.py")
                    tests_path = Path(tmpdir, f"test_{spec.module_name}.py")
                    runner_path = Path(tmpdir, "run_tests.py")

                    logger.info("Writing correct code and tests to temp files")
                    module_path.write_text(spec.correct_code, encoding="utf-8")
                    tests_path.write_text(spec.unit_tests, encoding="utf-8")
                    runner_path.write_text(runner_code, encoding="utf-8")

                    # Step 2: Verify correct code passes all tests
                    logger.info("Running tests on correct code")
                    pass_result = _run_tests(tmpdir)
                    if pass_result.returncode != 0:
                        logger.error("Correct code failed tests")
                        error_msg = (
                            "Correct code failed its own tests."
                            f"\nstdout:\n{pass_result.stdout}\n"
                            f"stderr:\n{pass_result.stderr}"
                        )
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
                            continue
                        else:
                            raise RuntimeError(error_msg)
                    logger.info("Correct code passed all tests")

                    # Step 3: Inject meaningful bug and ensure it creates debugging challenge
                    logger.info("Starting bug injection")
                    buggy_code, failing_output = self._inject_bug_and_capture(
                        seed=seed,
                        spec=spec,
                        tmpdir=tmpdir,
                        module_path=module_path,
                    )
                    logger.info("Bug injection successful")
                    break
            else:
                raise RuntimeError(f"Failed to generate valid spec after {max_retries} attempts")
            result = DebugConversation(
                domain=seed.domain,
                topic=seed.topic,
                subtopic=seed.subtopic,
                summary=spec.summary,
                problem_description=spec.problem_description,
                solution_outline=spec.solution_outline,
                module_name=spec.module_name,
                correct_code=spec.correct_code,
                buggy_code=buggy_code,
                unit_tests=spec.unit_tests,
                runner_code=runner_code,
                failing_output=failing_output,
                conversation=self._build_conversation(
                    seed=seed,
                    spec=spec,
                    runner_code=runner_code,
                    failing_output=failing_output,
                    buggy_code=buggy_code,
                ),
            )
            logger.info("Conversation generated successfully")
        except Exception as exc:
            logger.error(f"Generation failed for seed {seed}: {str(exc)}")
            raise RuntimeError(f"Failed to generate conversation: {str(exc)}") from exc
        else:
            self._catalog.mark_used(seed)
            logger.info("Marked seed as used")
            return result

    def generate_from_examples(
        self,
        prompt: str,
        correct_code: str,
        input_outputs: List[Dict[str, Any]],
        function_name: str,
        module_name: str,
        domain: str = "General",
        topic: str = "Programming",
        subtopic: str = "Debugging",
        summary: str = "Debugging task based on input-output examples",
        solution_outline: str = "",
    ) -> DebugConversation:
        test_cases = []
        for i, io in enumerate(input_outputs, 1):
            input_data = io["input"]
            expected = io["output"]
            if not isinstance(input_data, (list, tuple)):
                input_data = [input_data]
            args_str = ", ".join(repr(arg) for arg in input_data)
            test_cases.append(
                f"    def test_case_{i}(self):\n        self.assertEqual({function_name}({args_str}), {repr(expected)})\n"
            )

        unit_tests = (
            "import unittest\n"
            f"from {module_name} import {function_name}\n\n"
            f"class Test{function_name.capitalize()}(unittest.TestCase):\n"
            + "".join(test_cases)
        )

        spec = GeneratedSpec(
            domain=domain,
            topic=topic,
            subtopic=subtopic,
            summary=summary,
            module_name=module_name,
            problem_description=prompt,
            solution_outline=solution_outline,
            correct_code=correct_code,
            unit_tests=unit_tests,
        )

        runner_code = _build_test_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir, f"{spec.module_name}.py")
            tests_path = Path(tmpdir, f"test_{spec.module_name}.py")
            runner_path = Path(tmpdir, "run_tests.py")

            module_path.write_text(spec.correct_code, encoding="utf-8")
            tests_path.write_text(spec.unit_tests, encoding="utf-8")
            runner_path.write_text(runner_code, encoding="utf-8")

            logger.info("Running tests on provided correct code")
            pass_result = _run_tests(tmpdir)
            if pass_result.returncode != 0:
                raise RuntimeError(
                    f"Provided correct code failed the input-output tests.\n"
                    f"stdout:\n{pass_result.stdout}\n"
                    f"stderr:\n{pass_result.stderr}"
                )
            logger.info("Correct code passed all tests")

            logger.info("Starting bug injection")
            buggy_code, failing_output = self._inject_bug_and_capture(
                seed=ScenarioSeed(domain=domain, topic=topic, subtopic=subtopic),
                spec=spec,
                tmpdir=tmpdir,
                module_path=module_path,
            )
            logger.info("Bug injection successful")

        result = DebugConversation(
            domain=domain,
            topic=topic,
            subtopic=subtopic,
            summary=spec.summary,
            problem_description=spec.problem_description,
            solution_outline=spec.solution_outline,
            module_name=spec.module_name,
            correct_code=spec.correct_code,
            buggy_code=buggy_code,
            unit_tests=spec.unit_tests,
            runner_code=runner_code,
            failing_output=failing_output,
            conversation=self._build_conversation(
                seed=ScenarioSeed(domain=domain, topic=topic, subtopic=subtopic),
                spec=spec,
                runner_code=runner_code,
                failing_output=failing_output,
                buggy_code=buggy_code,
            ),
        )
        return result

    def generate_from_assertions(
        self,
        problem: str,
        correct_code: str,
        assertion_functions: List[str],
        module_name: str,
        domain: str = "General",
        topic: str = "Programming",
        subtopic: str = "Debugging",
        summary: str = "Debugging task based on assertion functions",
        solution_outline: str = "",
    ) -> DebugConversation:
        assertion_defs = "\n\n".join(assertion_functions)
        test_methods = []
        for i, func_str in enumerate(assertion_functions, 1):
            # Extract function name
            func_name = func_str.split("def ")[1].split("(")[0].strip()
            test_methods.append(
                f"    def test_{i}(self):\n        {func_name}()\n"
            )

        unit_tests = (
            "import unittest\n"
            f"from {module_name} import *\n\n"
            f"{assertion_defs}\n\n"
            f"class TestAssertions(unittest.TestCase):\n"
            + "".join(test_methods)
        )

        spec = GeneratedSpec(
            domain=domain,
            topic=topic,
            subtopic=subtopic,
            summary=summary,
            module_name=module_name,
            problem_description=problem,
            solution_outline=solution_outline,
            correct_code=correct_code,
            unit_tests=unit_tests,
        )

        runner_code = _build_test_runner()

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir, f"{spec.module_name}.py")
            tests_path = Path(tmpdir, f"test_{spec.module_name}.py")
            runner_path = Path(tmpdir, "run_tests.py")

            module_path.write_text(spec.correct_code, encoding="utf-8")
            tests_path.write_text(spec.unit_tests, encoding="utf-8")
            runner_path.write_text(runner_code, encoding="utf-8")

            logger.info("Running tests on provided correct code")
            pass_result = _run_tests(tmpdir)
            if pass_result.returncode != 0:
                raise RuntimeError(
                    f"Provided correct code failed the assertion tests.\n"
                    f"stdout:\n{pass_result.stdout}\n"
                    f"stderr:\n{pass_result.stderr}"
                )
            logger.info("Correct code passed all tests")

            logger.info("Starting bug injection")
            buggy_code, failing_output = self._inject_bug_and_capture(
                seed=ScenarioSeed(domain=domain, topic=topic, subtopic=subtopic),
                spec=spec,
                tmpdir=tmpdir,
                module_path=module_path,
            )
            logger.info("Bug injection successful")

        result = DebugConversation(
            domain=domain,
            topic=topic,
            subtopic=subtopic,
            summary=spec.summary,
            problem_description=spec.problem_description,
            solution_outline=spec.solution_outline,
            module_name=spec.module_name,
            correct_code=spec.correct_code,
            buggy_code=buggy_code,
            unit_tests=spec.unit_tests,
            runner_code=runner_code,
            failing_output=failing_output,
            conversation=self._build_conversation(
                seed=ScenarioSeed(domain=domain, topic=topic, subtopic=subtopic),
                spec=spec,
                runner_code=runner_code,
                failing_output=failing_output,
                buggy_code=buggy_code,
            ),
        )
        return result

    def generate_file_edit_task(
        self,
        *,
        bug_file_path: Path,
        seed: Optional[ScenarioSeed] = None,
    ) -> DebugConversation:
        """Generate a debugging conversation where the buggy code lives on disk.

        The buggy implementation, unit tests, and the runner are written to the
        provided directory. The conversation instructs the developer/LLM to open
        the file at `bug_file_path`, investigate, and fix it in-place.
        """
        logger.info("Starting file-edit task generation")

        if seed is None:
            seed = self._catalog.acquire()
        logger.info(f"Acquired seed: {seed.domain} - {seed.topic} - {seed.subtopic}")

        # Produce a validated spec and a meaningful buggy variant using temp isolation
        max_retries = 3
        for attempt in range(max_retries):
            logger.info(f"Generating spec from LLM (attempt {attempt + 1})")
            spec = self._llm.generate_spec(seed)
            logger.info("Validating spec against seed")
            self._validate_spec_against_seed(spec, seed)
            runner_code = _build_test_runner()

            with tempfile.TemporaryDirectory() as tmpdir:
                module_path = Path(tmpdir, f"{spec.module_name}.py")
                tests_path = Path(tmpdir, f"test_{spec.module_name}.py")
                runner_path = Path(tmpdir, "run_tests.py")

                logger.info("Writing correct code and tests to temp files")
                module_path.write_text(spec.correct_code, encoding="utf-8")
                tests_path.write_text(spec.unit_tests, encoding="utf-8")
                runner_path.write_text(runner_code, encoding="utf-8")

                logger.info("Running tests on correct code")
                pass_result = _run_tests(tmpdir)
                if pass_result.returncode != 0:
                    logger.error("Correct code failed tests in file-edit generation")
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
                        continue
                    raise RuntimeError(
                        "Correct code failed its own tests for file-edit task."
                    )

                logger.info("Injecting bug and capturing failing output")
                buggy_code, failing_output = self._inject_bug_and_capture(
                    seed=seed,
                    spec=spec,
                    tmpdir=tmpdir,
                    module_path=module_path,
                )
                logger.info("Bug injection successful for file-edit task")

            # After the temp session, materialise artefacts to the requested location
            target_dir = bug_file_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            # Ensure filename aligns with module_name.py to satisfy imports in tests
            desired_name = f"{spec.module_name}.py"
            if bug_file_path.name != desired_name:
                bug_file_path = target_dir / desired_name

            bug_file_path.write_text(buggy_code, encoding="utf-8")
            tests_on_disk = target_dir / f"test_{spec.module_name}.py"
            runner_on_disk = target_dir / "run_tests.py"
            tests_on_disk.write_text(spec.unit_tests, encoding="utf-8")
            runner_on_disk.write_text(runner_code, encoding="utf-8")

            conversation = self._build_file_edit_conversation(
                seed=seed,
                problem_description=spec.problem_description,
                solution_outline=spec.solution_outline,
                module_name=spec.module_name,
                failing_output=failing_output,
                bug_file_path=bug_file_path,
                tests_path=tests_on_disk,
                runner_path=runner_on_disk,
            )

            result = DebugConversation(
                domain=seed.domain,
                topic=seed.topic,
                subtopic=seed.subtopic,
                summary=spec.summary,
                problem_description=spec.problem_description,
                solution_outline=spec.solution_outline,
                module_name=spec.module_name,
                correct_code=spec.correct_code,
                buggy_code=buggy_code,
                unit_tests=spec.unit_tests,
                runner_code=runner_code,
                failing_output=failing_output,
                conversation=conversation,
                bug_file_path=str(bug_file_path),
            )

            self._catalog.mark_used(seed)
            logger.info("File-edit conversation generated successfully and seed marked used")
            return result

        raise RuntimeError("Failed to generate valid file-edit task after retries")

    def _inject_bug_and_capture(
        self,
        *,
        seed: ScenarioSeed,
        spec: GeneratedSpec,
        tmpdir: str,
        module_path: Path,
    ) -> tuple[str, str]:
        """Inject a meaningful bug that creates a non-trivial debugging challenge."""
        logger.info("Generating bug candidates")
        original_code = spec.correct_code
        candidate_generator = self._bug_injector.generate_candidates(
            correct_code=original_code,
            spec=spec,
            seed=seed,
            num_llm_candidates=self._num_llm_candidates,
        )

        def find_meaningful_bug(candidate: str, depth: int = 0, max_depth: int = 2) -> Optional[tuple[str, str]]:
            if depth > max_depth:
                return None

            module_path.write_text(candidate, encoding="utf-8")
            fail_result = _run_tests(tmpdir)
            module_path.write_text(original_code, encoding="utf-8")  # Always restore

            if fail_result.returncode != 0:
                failing_output = (fail_result.stdout + fail_result.stderr).strip()
                if self._is_meaningful_bug(failing_output, spec):
                    return candidate, failing_output

                # If not meaningful, try to complicate
                if hasattr(self._bug_injector, '_llm_strategy') and self._bug_injector._llm_strategy:
                    complicated = self._bug_injector._llm_strategy.mutate(
                        correct_code=candidate,
                        spec=spec,
                        seed=seed,
                    )
                    if complicated and complicated != candidate:
                        result = find_meaningful_bug(complicated, depth + 1, max_depth)
                        if result:
                            return result

            return None

        first_failing = None
        evaluated = 0
        for candidate in candidate_generator:
            if candidate.strip() == original_code.strip():
                continue
            evaluated += 1
            logger.info(f"Testing candidate {evaluated}")
            result = find_meaningful_bug(candidate)
            if result:
                logger.info("Found meaningful bug")
                return result
            # Track first failing for fallback
            module_path.write_text(candidate, encoding="utf-8")
            fail_result = _run_tests(tmpdir)
            module_path.write_text(original_code, encoding="utf-8")
            if fail_result.returncode != 0 and first_failing is None:
                first_failing = (candidate, (fail_result.stdout + fail_result.stderr).strip())

        if first_failing:
            logger.warning("No meaningful bug found; falling back to first failing candidate")
            return first_failing

        raise RuntimeError("Bug injection strategies failed to produce any failing implementation.")

    def _is_meaningful_bug(self, failing_output: str, spec: GeneratedSpec) -> bool:
        """Determine if a test failure represents a meaningful debugging challenge."""
        # Parse the test output to count failures and errors
        lines = failing_output.split('\n')
        failure_count = 0
        error_count = 0
        error_types = set()

        for line in lines:
            line = line.strip()
            if line.startswith('FAIL') or 'FAIL' in line:  # Broader matching for failures
                failure_count += 1
            if line.startswith('ERROR') or 'ERROR' in line:
                error_count += 1
            # Extract error types
            for etype in ['AssertionError', 'RuntimeError', 'RecursionError', 'IndexError', 'ValueError', 'TypeError']:
                if etype in line:
                    error_types.add(etype)

        total_failures = failure_count + error_count
        total_tests_approx = len(spec.unit_tests.split('def test_')) - 1  # Approximate test count
        logger.debug(f"Bug evaluation: failures={total_failures}, errors={error_count}, types={error_types}, approx_tests={total_tests_approx}")

        if total_failures == 0:
            logger.debug("Rejected: No failures")
            return False

        # Avoid obvious/easy bugs
        obvious_patterns = [
            'SyntaxError', 'IndentationError', 'ImportError', 'ModuleNotFoundError',
            'NameError'  # Removed TypeError to allow more
        ]
        for pattern in obvious_patterns:
            if pattern in failing_output:
                logger.debug(f"Rejected: Contains obvious pattern {pattern}")
                return False

        # Relaxed: Allow if 1+ failures, not all tests fail, and at least one interesting error
        if 0 < total_failures < total_tests_approx and (len(error_types) > 0 or total_failures >= 1):
            logger.debug("Accepted: Meets relaxed meaningful criteria")
            return True

        logger.debug(f"Rejected: Does not meet criteria (failures={total_failures}/{total_tests_approx}, types={error_types})")
        return False

    def _build_conversation(
        self,
        *,
        seed: ScenarioSeed,
        spec: GeneratedSpec,
        runner_code: str,
        failing_output: str,
        buggy_code: str,
    ) -> List[Dict[str, str]]:
        logger.info("Building conversation structure")
        return [
            {
                "role": "architect",
                "content": (
                    f"Domain: {seed.domain} | Topic: {seed.topic} | Subtopic: {seed.subtopic}\n"
                    f"{spec.problem_description}"
                ),
            },
            {
                "role": "planner",
                "content": f"Solution strategy:\n{spec.solution_outline.strip()}",
            },
            {
                "role": "developer",
                "content": f"Correct implementation (`{spec.module_name}.py`):\n```python\n{spec.correct_code.strip()}\n```",
            },
            {
                "role": "qa",
                "content": (
                    "Validation assets for the feature:\n"
                    f"Unit tests:\n```python\n{spec.unit_tests.strip()}\n```\n"
                    f"Test runner:\n```python\n{runner_code.strip()}\n```"
                ),
            },
            {
                "role": "developer",
                "content": f"Injected defect version:\n```python\n{buggy_code.strip()}\n```",
            },
            {
                "role": "qa",
                "content": f"""Test suite failure trace:\n```
{failing_output}
```""",
            },
            {
                "role": "developer",
                "content": "Restored the original implementation so all unit tests pass again.",
            },
        ]

    def _build_file_edit_conversation(
        self,
        *,
        seed: ScenarioSeed,
        problem_description: str,
        solution_outline: str,
        module_name: str,
        failing_output: str,
        bug_file_path: Path,
        tests_path: Path,
        runner_path: Path,
    ) -> List[Dict[str, str]]:
        """Conversation variant that instructs editing a real file on disk.

        This variant avoids inlining the buggy source and instead references the
        absolute file paths where the artefacts have been materialised. The task
        for the developer/LLM is to open and modify the file in-place until the
        tests pass.
        """
        logger.info("Building file-edit conversation structure")
        return [
            {
                "role": "architect",
                "content": (
                    f"Domain: {seed.domain} | Topic: {seed.topic} | Subtopic: {seed.subtopic}\n"
                    f"{problem_description}"
                ),
            },
            {
                "role": "planner",
                "content": f"Solution strategy:\n{solution_outline.strip()}",
            },
            {
                "role": "qa",
                "content": (
                    "Debugging task setup (file-based):\n"
                    f"- Buggy module path: {str(bug_file_path)}\n"
                    f"- Unit tests path: {str(tests_path)}\n"
                    f"- Test runner path: {str(runner_path)}\n\n"
                    "Open the buggy module, run the tests using the runner, and edit the file in-place until all tests pass."
                ),
            },
            {
                "role": "qa",
                "content": f"""Current failing test output (for reference):\n```
{failing_output}
```""",
            },
            {
                "role": "developer",
                "content": (
                    "I have opened the specified file, investigated the failures, and applied fixes. "
                    "Re-running the test suite now passes locally."
                ),
            },
        ]

    @staticmethod
    def _validate_spec_against_seed(spec: GeneratedSpec, seed: ScenarioSeed) -> None:
        if _normalize_label(spec.domain) != _normalize_label(seed.domain):
            raise ValueError(
                f"Spec domain '{spec.domain}' does not match requested domain '{seed.domain}'"
            )
        if _normalize_label(spec.topic) != _normalize_label(seed.topic):
            raise ValueError(
                f"Spec topic '{spec.topic}' does not match requested topic '{seed.topic}'"
            )
        if _normalize_label(spec.subtopic) != _normalize_label(seed.subtopic):
            raise ValueError(
                f"Spec subtopic '{spec.subtopic}' does not match requested subtopic '{seed.subtopic}'"
            )


def _normalize_label(value: str) -> str:
    return " ".join(value.split()).casefold()


class BugInjector(Protocol):
    def generate_candidates(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Iterable[str]:
        ...


class BugStrategy(Protocol):
    def mutate(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Optional[str]:
        ...


class CompositeBugInjector:
    def __init__(self, strategies: Sequence[BugStrategy]) -> None:
        self._strategies = list(strategies)
        # Prioritize LLM strategy
        self._llm_strategy = next((s for s in self._strategies if isinstance(s, LLMBugInjectionStrategy)), None)

    def generate_candidates(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
        num_llm_candidates: int = 1,
    ) -> Iterable[str]:
        seen: set[str] = set()
        
        # Always start with LLM-based combined bugs
        if self._llm_strategy:
            for _ in range(num_llm_candidates):
                candidate = self._llm_strategy.mutate(
                    correct_code=correct_code,
                    spec=spec,
                    seed=seed,
                )
                if candidate and candidate.strip() not in seen:
                    seen.add(candidate.strip())
                    yield candidate
        
        # Optionally add rule-based for fallback, without random LLM combination
        for strategy in self._strategies:
            if isinstance(strategy, LLMBugInjectionStrategy):
                continue  # Already handled
            try:
                candidate = strategy.mutate(
                    correct_code=correct_code,
                    spec=spec,
                    seed=seed,
                )
                if candidate:
                    normalized = candidate.strip()
                    if normalized not in seen:
                        seen.add(normalized)
                        yield candidate
                    
                    # Add mixed: Combine with LLM mutation for variety
                    if self._llm_strategy:
                        mixed = self._llm_strategy.mutate(
                            correct_code=candidate,
                            spec=spec,
                            seed=seed,
                        )
                        if mixed and mixed.strip() not in seen:
                            seen.add(mixed.strip())
                            yield mixed
            except Exception:
                continue


class ArithmeticOperatorFlipStrategy:
    def mutate(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Optional[str]:
        return _apply_ast_transform(correct_code, _ArithmeticFlipTransformer)


class ComparatorInversionStrategy:
    def mutate(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Optional[str]:
        return _apply_ast_transform(correct_code, _ComparatorFlipTransformer)


class NumericPerturbationStrategy:
    def mutate(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Optional[str]:
        return _apply_ast_transform(correct_code, _NumericConstantPerturber)


class LLMBugInjectionStrategy:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_attempts: int = 2,
    ) -> None:
        self._model = model or os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._cache: Dict[str, Optional[str]] = {}  # Cache by input hash

    SYSTEM_PROMPT = (
        "You are an expert Python debugger specializing in injecting realistic, challenging bugs into scientific code. "
        "Your goal is to create buggy versions that require deep debugging, domain knowledge, and multi-step reasoning to fix. "
        "Always inject COMBINATIONS of 2-3 subtle bugs to avoid simple recoveries. Draw from these bug types: "
        "- Off-by-one errors in loops, indices, or slices (e.g., range(n) to range(n+1)). "
        "- Logical operator swaps (e.g., 'and' to 'or', or negating conditions). "
        "- Exception handling omissions or mismatches (e.g., remove try-except or wrong except type). "
        "- Data structure misuses (e.g., list instead of set, mutations on immutables). "
        "- Domain-specific formula perturbations (e.g., slight changes to scientific constants or equations). "
        "- Recursion/loop depth issues (e.g., wrong base cases leading to infinite recursion). "
        "- Structural syntax issues such as missing parentheses or malformed JSON literals to trigger parser failures. "
        "Ensure bugs are plausible developer mistakes, cause partial test failures, and are challenging but fixable. "
        "Do not add any new comments explaining the bugs or changes. Preserve all original comments from the correct code unless the bug specifically requires modifying them. "
        "Respond ONLY with the mutated code."
    )

    USER_PROMPT_TEMPLATE = """
    Inject a combination of 2-3 bugs into this correct code based on the types above. 
    Domain: {domain}
    Topic: {topic}
    Subtopic: {subtopic}
    Correct Code:
    ```python
    {correct_code}
    ```
    """

    def mutate(
        self,
        *,
        correct_code: str,
        spec: GeneratedSpec,
        seed: ScenarioSeed,
    ) -> Optional[str]:
        import hashlib
        cache_key = hashlib.md5(correct_code.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            domain=seed.domain,
            topic=seed.topic,
            subtopic=seed.subtopic,
            correct_code=correct_code,
        )

        for attempt in range(self._max_attempts):
            try:
                completion = litellm.completion(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=2000,
                )
                candidate = completion.choices[0].message.content.strip()
                if candidate and candidate != correct_code:
                    self._cache[cache_key] = candidate
                    return candidate
            except Exception:
                continue
        self._cache[cache_key] = None
        return None


def _apply_ast_transform(code: str, transformer_cls: type["_TrackingTransformer"]) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    transformer = transformer_cls()
    tree = transformer.visit(tree)
    if not transformer.mutated:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _TrackingTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.mutated = False


class _ArithmeticFlipTransformer(_TrackingTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
            self.mutated = True
        elif isinstance(node.op, ast.Sub):
            node.op = ast.Add()
            self.mutated = True
        return node


class _ComparatorFlipTransformer(_TrackingTransformer):
    _mapping = {
        ast.Gt: ast.Lt,
        ast.Lt: ast.Gt,
        ast.GtE: ast.LtE,
        ast.LtE: ast.GtE,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.mutated or len(node.ops) != 1:
            return node
        op = node.ops[0]
        for from_type, to_type in self._mapping.items():
            if isinstance(op, from_type):
                node.ops[0] = to_type()
                self.mutated = True
                break
        return node


class _NumericConstantPerturber(_TrackingTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if (
            not self.mutated
            and isinstance(node.value, (int, float))
            and node.value not in {0, 1, -1}
        ):
            node.value = float(node.value) * 1.05
            self.mutated = True
        return node


def _default_bug_injector() -> BugInjector:
    strategies: list[BugStrategy] = [
        LLMBugInjectionStrategy(),
        ArithmeticOperatorFlipStrategy(),
        ComparatorInversionStrategy(),
        NumericPerturbationStrategy(),
    ]
    return CompositeBugInjector(strategies)


def _parse_yaml_document(text: str) -> Dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Received empty response from LLM")

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency notice
        raise ImportError(
            "PyYAML is required to parse LiteLLM YAML output. Install it with `pip install PyYAML`."
        ) from exc

    data = yaml.safe_load(stripped)
    if not isinstance(data, dict):
        raise ValueError("YAML payload must deserialize to a mapping")
    return data


def _spec_from_payload(payload: Dict[str, object]) -> GeneratedSpec:
    required_keys: Sequence[str] = (
        "domain",
        "topic",
        "subtopic",
        "summary",
        "module_name",
        "problem_description",
        "solution_outline",
        "correct_code",
        "unit_tests",
    )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"LLM payload missing keys: {missing}")

    return GeneratedSpec(
        domain=str(payload["domain"]),
        topic=str(payload["topic"]),
        subtopic=str(payload["subtopic"]),
        summary=str(payload["summary"]),
        module_name=str(payload["module_name"]),
        problem_description=str(payload["problem_description"]),
        solution_outline=str(payload["solution_outline"]),
        correct_code=str(payload["correct_code"]),
        unit_tests=str(payload["unit_tests"]),
    )


def _run_tests(workdir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "run_tests.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


def _build_test_runner() -> str:
    return """import sys
import unittest


def main() -> int:
    suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


__all__ = [
    "GeneratedSpec",
    "LLMGenerator",
    "LiteLLMGenerator",
    "DebugConversation",
    "DebugConversationPipeline",
    "BugInjector",
    "ScenarioCatalog",
    "ScenarioSeed",
    "DEFAULT_CATALOG_PATH",
]
