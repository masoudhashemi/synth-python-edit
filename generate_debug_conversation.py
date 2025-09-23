"""CLI entrypoint to generate a synthetic debugging conversation."""

import argparse
from pathlib import Path

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
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the conversation JSON. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    conversation = pipeline.generate()
    payload = conversation.to_json()

    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
