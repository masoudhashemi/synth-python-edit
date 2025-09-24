"""Generate multiple file-edit debugging tasks via LiteLLM.

Each task materialises a buggy module, unit tests, and a test runner in a
dedicated directory. A conversation JSON referencing those paths is saved in the
same directory. The buggy code is not required to be embedded in the JSON; you
can pass --suppress-buggy-code to remove it from the payload.
"""

import argparse
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import json
import hashlib

from synthetic_debug.catalog import DEFAULT_CATALOG_PATH, ScenarioCatalog, ScenarioSeed
from synthetic_debug.pipeline import DebugConversationPipeline, LiteLLMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch of file-edit debug tasks")
    parser.add_argument("--num", "--count", type=int, default=5, help="Number of tasks to generate (aliases: --num, --count)")
    parser.add_argument("--output", default="file_edit_tasks", help="Base output directory for tasks")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per generation")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH), help="Path to scenario catalog")
    parser.add_argument("--state", help="Path to optional catalog state file (not used in workers)")
    parser.add_argument("--catalog-seed", type=int, help="Random seed for catalog selection order")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries per generation")
    parser.add_argument("--prefix", default="task", help="Directory prefix for each generated sample")
    parser.add_argument("--suppress-buggy-code", action="store_true", help="Remove buggy_code from the JSON payloads")
    return parser.parse_args()


def strip_buggy_code(conversation_json: str) -> str:
    try:
        payload = json.loads(conversation_json)
    except Exception:
        return conversation_json
    payload.pop("buggy_code", None)
    return json.dumps(payload, indent=2, sort_keys=True)


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def make_tmp_dirname(prefix: str, index: int) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{timestamp}_{index:03d}_tmp"


def make_final_dirname(prefix: str, module_name: str, topic: str, hash_value: str) -> str:
    module_slug = _slug(module_name)[:40]
    topic_slug = _slug(topic)[:40]
    short_hash = hash_value[:8]
    return f"{prefix}_{module_slug}_{topic_slug}_{short_hash}"


def generate_single(args_tuple: Tuple[int, ScenarioSeed, Path, dict, int, str, bool]) -> Optional[Path]:
    index, seed, base_dir, llm_config, retries, prefix, suppress_buggy = args_tuple
    pipeline = DebugConversationPipeline(
        llm=LiteLLMGenerator(
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_attempts=llm_config["max_attempts"],
        )
    )

    attempts = 0
    while attempts <= retries:
        attempts += 1
        try:
            # Create a temporary directory for generation
            task_dir_tmp = base_dir / make_tmp_dirname(prefix, index)
            task_dir_tmp.mkdir(parents=True, exist_ok=True)

            placeholder = task_dir_tmp / "placeholder.py"
            convo = pipeline.generate_file_edit_task(bug_file_path=placeholder, seed=seed)

            # Compute stable hash and final directory name based on module, topic, and code hash
            code_hash = hashlib.md5(convo.correct_code.encode("utf-8")).hexdigest()
            final_name = make_final_dirname(prefix, convo.module_name, seed.topic, code_hash)
            task_dir = base_dir / final_name

            # Rename/move the temporary directory to the final destination
            task_dir_tmp.rename(task_dir)

            # Update bug_file_path in payload to the moved location
            payload = json.loads(convo.to_json())
            bug_path = task_dir / f"{convo.module_name}.py"
            payload["bug_file_path"] = str(bug_path)
            payload = json.dumps(payload, indent=2, sort_keys=True)
            if suppress_buggy:
                payload = strip_buggy_code(payload)
            (task_dir / "conversation.json").write_text(payload, encoding="utf-8")
            return task_dir
        except RuntimeError:
            if attempts > retries:
                return None
            continue


def main() -> None:
    args = parse_args()

    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    catalog = ScenarioCatalog(
        Path(args.catalog),
        state_path=Path(args.state) if args.state else None,
        random_seed=args.catalog_seed,
    )

    if catalog.remaining < args.num:
        print(
            f"Warning: catalog only has {catalog.remaining} remaining combinations, fewer than requested {args.num}.",
            file=sys.stderr,
        )

    # Collect unique seeds (domain, topic) for diversity similar to the other batch script
    seen = set()
    seeds = []
    while len(seeds) < args.num and catalog.remaining > 0:
        seed = catalog.acquire()
        key = (seed.domain, seed.topic)
        if key not in seen:
            seen.add(key)
            seeds.append(seed)
        else:
            # allow duplicates if necessary to fill quota
            seeds.append(seed)

    llm_config = {
        "model": args.model,
        "temperature": args.temperature,
        "max_attempts": args.max_attempts,
    }

    payloads = [
        (i + 1, seed, base_dir, llm_config, args.retries, args.prefix, args.suppress_buggy_code)
        for i, seed in enumerate(seeds)
    ]

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(generate_single, payloads)

    successful = [r for r in results if r is not None]
    print(f"Generated {len(successful)} file-edit tasks in {base_dir}")
    if len(successful) < len(seeds):
        print(
            f"Warning: produced {len(successful)} out of {len(seeds)} after retries.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()


