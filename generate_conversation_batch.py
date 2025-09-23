"""Generate multiple synthetic debugging conversations via LiteLLM."""

import argparse
import sys
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from synthetic_debug.catalog import DEFAULT_CATALOG_PATH, ScenarioCatalog
from synthetic_debug.pipeline import DebugConversationPipeline, LiteLLMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch of debug conversations")
    parser.add_argument("--num", "--count", type=int, default=10, help="Number of conversations to generate (aliases: --num, --count)")
    parser.add_argument("--output", default="sample_conversations", help="Output directory")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per generation")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH), help="Path to scenario catalog")
    parser.add_argument("--state", help="Path to catalog state file")
    parser.add_argument("--catalog-seed", type=int, help="Random seed for catalog")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries per generation")
    parser.add_argument("--prefix", default="conversation", help="Filename prefix")
    return parser.parse_args()


def make_filename(prefix: str, index: int) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{timestamp}_{index:03d}.json"


def generate_single(args_tuple):
    seed, output_dir, llm_config, retries = args_tuple
    pipeline = DebugConversationPipeline(
        llm=LiteLLMGenerator(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_attempts=llm_config['max_attempts']
        )
    )
    
    attempts = 0
    while attempts <= retries:
        attempts += 1
        try:
            conversation = pipeline.generate(seed=seed)
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            filename = f"conversation_{timestamp}_{seed.domain}_{seed.topic}.json"
            output_path = output_dir / filename
            output_path.write_text(conversation.to_json(), encoding="utf-8")
            return output_path
        except RuntimeError as exc:
            if attempts > retries:
                raise
            continue
    return None

def main(
    num_conversations: int = 10,
    output_dir: str = "sample_conversations",
    num_workers: int = multiprocessing.cpu_count(),
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_attempts: int = 3,
    catalog_path: str = str(DEFAULT_CATALOG_PATH),
    state_path: Optional[str] = None,
    catalog_seed: Optional[int] = None,
    retries: int = 3,
    prefix: str = "conversation",
) -> None:
    catalog = ScenarioCatalog(
        catalog_path,
        state_path=state_path,
        random_seed=catalog_seed,
    )
    
    if catalog.remaining < num_conversations:
        print(f"Warning: catalog only has {catalog.remaining} remaining combinations, fewer than requested {num_conversations}.", file=sys.stderr)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Collect unique seeds for diversity
    seen = set()
    seeds = []
    while len(seeds) < num_conversations and catalog.remaining > 0:
        seed = catalog.acquire()
        key = (seed.domain, seed.topic)
        if key not in seen:
            seen.add(key)
            seeds.append(seed)
        else:
            # If duplicate, still count it but warn or handle
            pass
    
    llm_config = {
        'model': model,
        'temperature': temperature,
        'max_attempts': max_attempts,
    }
    
    # Parallel generation with retries
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(
            generate_single,
            [(seed, output_path, llm_config, retries) for seed in seeds]
        )
    
    successful = [r for r in results if r is not None]
    print(f"Generated {len(successful)} conversations in {output_dir}")
    if len(successful) < num_conversations:
        print(f"Warning: produced {len(successful)} out of {num_conversations} after retries.", file=sys.stderr)

if __name__ == "__main__":
    args = parse_args()
    main(
        num_conversations=args.num,
        output_dir=args.output,
        num_workers=args.workers,
        model=args.model,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        catalog_path=args.catalog,
        state_path=args.state,
        catalog_seed=args.catalog_seed,
        retries=args.retries,
        prefix=args.prefix,
    )
