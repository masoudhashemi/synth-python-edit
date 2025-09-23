"""Scenario catalog management ensuring unique domain/topic/subtopic combinations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class ScenarioSeed:
    """Represents a single domain/topic/subtopic combination."""

    domain: str
    topic: str
    subtopic: str

    @property
    def key(self) -> str:
        return f"{self.domain}::{self.topic}::{self.subtopic}"


class ScenarioCatalog:
    """Loads and manages unique scenario seeds from a JSON catalog."""

    def __init__(
        self,
        catalog_path: Path,
        *,
        state_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self._catalog_path = Path(catalog_path)
        if not self._catalog_path.exists():
            raise FileNotFoundError(f"Catalog path does not exist: {self._catalog_path}")

        self._state_path = Path(state_path) if state_path is not None else None
        if self._state_path is not None:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

        self._rng = random.Random(random_seed)
        self._seeds_by_key = self._load_catalog(self._catalog_path)

        self._used_keys: Set[str] = set()
        if self._state_path is not None and self._state_path.exists():
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(state, list):
                raise ValueError("State file must contain a list of used scenario keys")
            for key in state:
                if key not in self._seeds_by_key:
                    raise ValueError(
                        f"State key '{key}' not present in catalog {self._catalog_path}"
                    )
                self._used_keys.add(key)

        self._available_keys: List[str] = [
            key for key in self._seeds_by_key if key not in self._used_keys
        ]
        self._inflight: Set[str] = set()

        if not self._available_keys:
            raise RuntimeError(
                "Scenario catalog exhausted: no unused domain/topic/subtopic combinations remain"
            )

    @property
    def remaining(self) -> int:
        return len(self._available_keys)

    def acquire(self) -> ScenarioSeed:
        if not self._available_keys:
            raise RuntimeError(
                "Scenario catalog exhausted: unable to acquire additional combinations"
            )

        index = 0 if len(self._available_keys) == 1 else self._rng.randrange(len(self._available_keys))
        key = self._available_keys.pop(index)
        self._inflight.add(key)
        return self._seeds_by_key[key]

    def release(self, seed: ScenarioSeed) -> None:
        key = seed.key
        if key in self._used_keys:
            return
        if key in self._inflight:
            self._available_keys.append(key)
            self._inflight.remove(key)

    def mark_used(self, seed: ScenarioSeed) -> None:
        key = seed.key
        self._used_keys.add(key)
        self._inflight.discard(key)
        if self._state_path is not None:
            self._state_path.write_text(
                json.dumps(sorted(self._used_keys), indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _load_catalog(path: Path) -> Dict[str, ScenarioSeed]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Sequence):
            raise ValueError("Scenario catalog must be a list of mappings")

        seeds: Dict[str, ScenarioSeed] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("Each catalog entry must be an object")
            try:
                domain = str(entry["domain"]).strip()
                topic = str(entry["topic"]).strip()
                subtopic = str(entry["subtopic"]).strip()
            except KeyError as exc:
                raise ValueError(f"Catalog entry missing required field: {exc.args[0]}") from exc

            seed = ScenarioSeed(domain=domain, topic=topic, subtopic=subtopic)
            if seed.key in seeds:
                raise ValueError(f"Duplicate scenario detected: {seed.key}")
            seeds[seed.key] = seed
        return seeds


DEFAULT_CATALOG_PATH = (Path(__file__).resolve().parent.parent / "scenario_catalog.json").resolve()


__all__ = [
    "ScenarioSeed",
    "ScenarioCatalog",
    "DEFAULT_CATALOG_PATH",
]
