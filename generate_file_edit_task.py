"""CLI to generate a file-edit debugging task (buggy code only on disk).

This writes a buggy module, unit tests, and a runner to a target directory and
emits a conversation JSON that references those file paths and includes the
failing test output. The buggy code is NOT inlined in the conversation content.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from synthetic_debug.catalog import DEFAULT_CATALOG_PATH, ScenarioCatalog
from synthetic_debug.pipeline import DebugConversationPipeline, LiteLLMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to the domain/topic/subtopic catalog JSON.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Optional path to persist catalog usage state for uniqueness across runs.",
    )
    parser.add_argument(
        "--catalog-seed",
        type=int,
        default=None,
        help="Random seed for catalog selection order.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional LiteLLM model identifier (defaults to env LITELLM_MODEL or gpt-4o-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to LiteLLM (default: 1.0).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry count for parsing LiteLLM output before failing (default: 3).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("file_edit_tasks"),
        help="Target directory where files will be written (default: file_edit_tasks).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="placeholder.py",
        help="Initial filename; will be coerced to <module_name>.py (default: placeholder.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the conversation JSON; prints to stdout when omitted.",
    )
    parser.add_argument(
        "--suppress-buggy-code",
        action="store_true",
        help="Remove buggy_code from the emitted JSON payload (recommended for file-edit tasks).",
    )
    return parser.parse_args()


def strip_buggy_code(conversation_json: str) -> str:
    try:
        payload = json.loads(conversation_json)
    except Exception:
        return conversation_json
    if "buggy_code" in payload:
        payload.pop("buggy_code", None)
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    target = args.dir / args.filename

    catalog = ScenarioCatalog(
        args.catalog,
        state_path=args.state,
        random_seed=args.catalog_seed,
    )
    llm = LiteLLMGenerator(
        model=args.model,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
    )
    pipeline = DebugConversationPipeline(llm=llm, catalog=catalog)

    conversation = pipeline.generate_file_edit_task(bug_file_path=target)
    payload = conversation.to_json()
    if args.suppress_buggy_code:
        payload = strip_buggy_code(payload)

    if args.output is None:
        out_name = f"conversation_{timestamp}_file_edit.json"
        out_path = args.dir / out_name
    else:
        out_path = args.output

    out_path.write_text(payload, encoding="utf-8")

    print("Written conversation JSON:", out_path)
    print("Bug file to edit:", conversation.bug_file_path)
    print("Unit tests:", (Path(conversation.bug_file_path).parent / f"test_{conversation.module_name}.py"))
    print("Runner:", (Path(conversation.bug_file_path).parent / "run_tests.py"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


