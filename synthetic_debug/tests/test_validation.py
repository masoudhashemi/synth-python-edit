import json
import tempfile
import unittest
from pathlib import Path

from synthetic_debug.catalog import ScenarioCatalog, ScenarioSeed
from synthetic_debug.pipeline import DebugConversationPipeline, GeneratedSpec, LLMGenerator
from synthetic_debug.validation import (
    ValidationError,
    extract_code_artifacts,
    load_conversation,
    run_tests_for_variant,
)


class FakeLLM(LLMGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate_spec(self, seed: ScenarioSeed) -> GeneratedSpec:
        self.calls += 1
        assert seed.domain == "Computational Physics"
        assert seed.topic == "Orbital mechanics simulation"
        assert seed.subtopic == "Symplectic integration for n-body systems"
        return GeneratedSpec(
            domain=seed.domain,
            topic=seed.topic,
            subtopic=seed.subtopic,
            summary="Approximates planetary trajectories using symplectic Euler integration.",
            module_name="orbital_integrator",
            problem_description=(
                "We simulate planar n-body motion for research teams comparing numerical integrators. "
                "The core routine should propagate velocities and positions with time-stepped updates."
            ),
            solution_outline=(
                "1. Validate the input bodies and coerce them into mutable float vectors.\n"
                "2. Compute pairwise gravitational acceleration contributions for each body.\n"
                "3. Apply a symplectic Euler step to update velocities before advancing positions."
            ),
            correct_code='''"""Symplectic Euler integrator for planar n-body motion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

G = 6.67430e-11  # gravitational constant


@dataclass
class Body:
    mass: float
    position: List[float]
    velocity: List[float]


def _coerce_bodies(entries: Iterable[Tuple[float, Sequence[float], Sequence[float]]]) -> List[Body]:
    bodies: List[Body] = []
    for entry in entries:
        mass, position, velocity = entry
        bodies.append(
            Body(
                mass=float(mass),
                position=[float(position[0]), float(position[1])],
                velocity=[float(velocity[0]), float(velocity[1])],
            )
        )
    if not bodies:
        raise ValueError("at least one body is required")
    return bodies


def _acceleration(target: Body, others: Sequence[Body]) -> List[float]:
    ax = ay = 0.0
    for other in others:
        if target is other:
            continue
        dx = other.position[0] - target.position[0]
        dy = other.position[1] - target.position[1]
        distance_sq = dx * dx + dy * dy
        if distance_sq == 0:
            continue
        distance = distance_sq ** 0.5
        strength = G * other.mass / distance_sq
        ax += strength * dx / distance
        ay += strength * dy / distance
    return [ax, ay]


def symplectic_step(bodies: Iterable[Tuple[float, Sequence[float], Sequence[float]]], dt: float) -> List[Body]:
    if dt <= 0:
        raise ValueError("time step must be positive")
    state = _coerce_bodies(bodies)
    accelerations = {id(body): _acceleration(body, state) for body in state}
    for body in state:
        ax, ay = accelerations[id(body)]
        body.velocity[0] += ax * dt
        body.velocity[1] += ay * dt
        body.position[0] += body.velocity[0] * dt
        body.position[1] += body.velocity[1] * dt
    return state
''',
            unit_tests='''import math
import unittest

from orbital_integrator import symplectic_step


class SymplecticStepTests(unittest.TestCase):
    def test_two_body_orbit_step(self) -> None:
        bodies = [
            (5.97e24, (0.0, 0.0), (0.0, 0.0)),  # Earth-like
            (7.35e22, (384_400_000.0, 0.0), (0.0, 1022.0)),  # Moon-like
        ]
        result = symplectic_step(bodies, dt=10.0)
        moon = result[1]
        self.assertLess(moon.velocity[0], 0.0)
        self.assertLess(moon.position[0], 384_400_000.0)
        self.assertAlmostEqual(result[0].velocity[1], 0.0, places=3)

    def test_invalid_time_step_raises(self) -> None:
        with self.assertRaises(ValueError):
            symplectic_step([], dt=-1.0)


if __name__ == "__main__":
    unittest.main()
''',
            buggy_code='''"""Symplectic Euler integrator for planar n-body motion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

G = 6.67430e-11


@dataclass
class Body:
    mass: float
    position: List[float]
    velocity: List[float]


def _coerce_bodies(entries: Iterable[Tuple[float, Sequence[float], Sequence[float]]]) -> List[Body]:
    bodies: List[Body] = []
    for entry in entries:
        mass, position, velocity = entry
        bodies.append(
            Body(
                mass=float(mass),
                position=[float(position[0]), float(position[1])],
                velocity=[float(velocity[0]), float(velocity[1])],
            )
        )
    if not bodies:
        raise ValueError("at least one body is required")
    return bodies


def _acceleration(target: Body, others: Sequence[Body]) -> List[float]:
    ax = ay = 0.0
    for other in others:
        if target is other:
            continue
        dx = other.position[0] - target.position[0]
        dy = other.position[1] - target.position[1]
        distance_sq = dx * dx + dy * dy
        if distance_sq == 0:
            continue
        distance = distance_sq ** 0.5
        strength = G * other.mass / distance_sq
        ax += strength * dx / distance
        ay += strength * dy / distance
    return [ax, ay]


def symplectic_step(bodies: Iterable[Tuple[float, Sequence[float], Sequence[float]]], dt: float) -> List[Body]:
    if dt <= 0:
        raise ValueError("time step must be positive")
    state = _coerce_bodies(bodies)
    accelerations = {id(body): _acceleration(body, state) for body in state}
    for body in state:
        ax, ay = accelerations[id(body)]
        body.velocity[0] += ax * dt
        body.velocity[1] += ay * dt
        body.position[0] += body.velocity[0] * dt
        body.position[1] += body.velocity[1] * dt
    # BUG: forgets to return deep copies, leaking mutation to callers and hiding defects
    return []
''',
        )


class ValidationUtilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_llm = FakeLLM()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        catalog_path = Path(self.tmpdir.name) / "catalog.json"
        catalog_path.write_text(
            json.dumps(
                [
                    {
                        "domain": "Computational Physics",
                        "topic": "Orbital mechanics simulation",
                        "subtopic": "Symplectic integration for n-body systems",
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.catalog = ScenarioCatalog(catalog_path, random_seed=99)
        self.pipeline = DebugConversationPipeline(llm=self.fake_llm, catalog=self.catalog)
        self.conversation = self.pipeline.generate()

    def test_run_tests_for_variant(self) -> None:
        passing = run_tests_for_variant(self.conversation, "correct")
        failing = run_tests_for_variant(self.conversation, "buggy")

        self.assertEqual(passing.returncode, 0)
        self.assertNotEqual(failing.returncode, 0)
        combined = failing.stdout + failing.stderr
        self.assertIn("FAIL", combined.upper())

    def test_extract_code_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "artefacts"
            extract_code_artifacts(self.conversation, dest)

            correct_path = dest / "orbital_integrator_correct.py"
            buggy_path = dest / "orbital_integrator_buggy.py"
            tests_path = dest / "test_orbital_integrator.py"
            runner_path = dest / "run_tests.py"

            self.assertTrue(correct_path.exists())
            self.assertTrue(buggy_path.exists())
            self.assertTrue(tests_path.exists())
            self.assertTrue(runner_path.exists())

            self.assertIn("symplectic_step", correct_path.read_text(encoding="utf-8"))
            self.assertIn("symplectic_step", tests_path.read_text(encoding="utf-8"))

    def test_load_conversation_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "conversation.json"
            file_path.write_text(self.conversation.to_json(), encoding="utf-8")

            loaded = load_conversation(file_path)
            self.assertEqual(loaded.module_name, self.conversation.module_name)
            self.assertEqual(loaded.solution_outline, self.conversation.solution_outline)

    def test_load_conversation_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bad.json"
            file_path.write_text(json.dumps({"module_name": "foo"}), encoding="utf-8")

            with self.assertRaises(ValidationError):
                load_conversation(file_path)


if __name__ == "__main__":
    unittest.main()
