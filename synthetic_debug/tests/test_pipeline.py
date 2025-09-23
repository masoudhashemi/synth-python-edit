import json
import tempfile
import unittest
from pathlib import Path

from synthetic_debug.catalog import ScenarioCatalog, ScenarioSeed
from synthetic_debug.pipeline import DebugConversationPipeline, GeneratedSpec, LLMGenerator


class FakeLLM(LLMGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate_spec(self, seed: ScenarioSeed) -> GeneratedSpec:
        self.calls += 1
        assert seed.domain == "Financial Analytics"
        assert seed.topic == "Quarterly budget variance"
        assert seed.subtopic == "Variance analytics for CFO dashboards"
        return GeneratedSpec(
            domain=seed.domain,
            topic=seed.topic,
            subtopic=seed.subtopic,
            summary="Detects budget variance drift using exponential smoothing.",
            module_name="budget_anomalies",
            problem_description=(
                "Finance analysts monitor quarter-over-quarter variance between actual and forecast "
                "spend. Sudden spikes should raise alerts after smoothing residuals with an EWMA."
            ),
            solution_outline=(
                "1. Validate and coerce both series into comparable floating point vectors.\n"
                "2. Compute normalized residuals using forecast magnitudes as the baseline.\n"
                "3. Apply an EWMA to the residuals and flag indices whose smoothed value breaches the threshold."
            ),
            correct_code='''"""Budget variance change-point detector."""
from typing import Dict, Iterable, List, Sequence


def _normalize_series(label: str, values: Iterable[float]) -> List[float]:
    series = [float(value) for value in values]
    if len(series) < 2:
        raise ValueError(f"{label} series must contain at least two points")
    return series


def _compute_residuals(actual: Sequence[float], forecast: Sequence[float]) -> List[float]:
    residuals: List[float] = []
    for actual_value, forecast_value in zip(actual, forecast):
        baseline = abs(forecast_value) if abs(forecast_value) > 1e-9 else 1.0
        residuals.append((actual_value - forecast_value) / baseline)
    return residuals


def _ewma(values: Sequence[float], alpha: float) -> List[float]:
    smoothed: List[float] = []
    previous = 0.0
    for index, value in enumerate(values):
        if index == 0:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def detect_budget_anomalies(
    actual: Iterable[float],
    forecast: Iterable[float],
    *,
    window: int = 3,
    threshold: float = 0.05,
) -> Dict[str, List[float]]:
    """Return a dictionary containing smoothed residuals and alert indices."""
    if window < 2:
        raise ValueError("window must be at least 2")
    actual_series = _normalize_series("actual", actual)
    forecast_series = _normalize_series("forecast", forecast)
    if len(actual_series) != len(forecast_series):
        raise ValueError("actual and forecast must have equal length")

    residuals = _compute_residuals(actual_series, forecast_series)
    alpha = 2.0 / (window + 1.0)
    smoothed = _ewma(residuals, alpha)
    alerts = [index for index, value in enumerate(smoothed) if abs(value) >= threshold]
    return {"smoothed": smoothed, "alerts": alerts}
''',
            unit_tests='''import unittest

from budget_anomalies import detect_budget_anomalies


class DetectBudgetAnomaliesTests(unittest.TestCase):
    def test_detects_alerts_with_smoothing(self) -> None:
        actual = [1050.0, 980.0, 1010.0, 1150.0]
        forecast = [1000.0, 1000.0, 1000.0, 1050.0]
        report = detect_budget_anomalies(actual, forecast, window=3, threshold=0.04)

        smoothed = [round(value, 4) for value in report["smoothed"]]
        self.assertEqual(smoothed[:3], [0.05, 0.015, 0.0125])
        self.assertIn(0, report["alerts"])
        self.assertIn(3, report["alerts"])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_budget_anomalies([100.0, 110.0], [100.0])

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_budget_anomalies([100.0, 110.0], [100.0, 100.0], window=1)


if __name__ == "__main__":
    unittest.main()
''',
            buggy_code='''"""Budget variance change-point detector."""
from typing import Dict, Iterable, List, Sequence


def _normalize_series(label: str, values: Iterable[float]) -> List[float]:
    series = [float(value) for value in values]
    if len(series) < 2:
        raise ValueError(f"{label} series must contain at least two points")
    return series


def _compute_residuals(actual: Sequence[float], forecast: Sequence[float]) -> List[float]:
    residuals: List[float] = []
    for actual_value, forecast_value in zip(actual, forecast):
        baseline = abs(actual_value) if abs(actual_value) > 1e-9 else 1.0  # BUG: normalize by actual
        residuals.append((actual_value - forecast_value) / baseline)
    return residuals


def _ewma(values: Sequence[float], alpha: float) -> List[float]:
    smoothed: List[float] = []
    previous = 0.0
    for index, value in enumerate(values):
        if index == 0:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def detect_budget_anomalies(
    actual: Iterable[float],
    forecast: Iterable[float],
    *,
    window: int = 3,
    threshold: float = 0.05,
) -> Dict[str, List[float]]:
    if window < 2:
        raise ValueError("window must be at least 2")
    actual_series = _normalize_series("actual", actual)
    forecast_series = _normalize_series("forecast", forecast)
    if len(actual_series) != len(forecast_series):
        raise ValueError("actual and forecast must have equal length")

    residuals = _compute_residuals(actual_series, forecast_series)
    alpha = 2.0 / (window + 1.0)
    smoothed = _ewma(residuals, alpha)
    alerts = [index for index, value in enumerate(smoothed) if abs(value) >= threshold]
    return {"smoothed": smoothed, "alerts": alerts}
''',
        )


class DebugConversationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_llm = FakeLLM()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        catalog_path = Path(self.tmpdir.name) / "catalog.json"
        catalog_path.write_text(
            json.dumps(
                [
                    {
                        "domain": "Financial Analytics",
                        "topic": "Quarterly budget variance",
                        "subtopic": "Variance analytics for CFO dashboards",
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.catalog = ScenarioCatalog(catalog_path, random_seed=123)
        self.pipeline = DebugConversationPipeline(llm=self.fake_llm, catalog=self.catalog)

    def test_generate_creates_conversation_and_runs_tests(self) -> None:
        convo = self.pipeline.generate()

        self.assertEqual(convo.domain, "Financial Analytics")
        self.assertEqual(convo.topic, "Quarterly budget variance")
        self.assertEqual(convo.subtopic, "Variance analytics for CFO dashboards")
        self.assertIn("1.", convo.solution_outline)
        planner_turn = convo.conversation[1]
        self.assertEqual(planner_turn["role"], "planner")
        self.assertIn("solution strategy", planner_turn["content"].lower())
        self.assertEqual(self.fake_llm.calls, 1)
        self.assertIn("detect_budget_anomalies", convo.correct_code)
        self.assertIn("FAIL", convo.failing_output.upper())
        self.assertEqual(convo.conversation[-1]["role"], "developer")
        self.assertIn("tests pass", convo.conversation[-1]["content"].lower())
        self.assertEqual(self.catalog.remaining, 0)


if __name__ == "__main__":
    unittest.main()
