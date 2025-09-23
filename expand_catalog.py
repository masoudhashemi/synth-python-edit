import json
import os
import sys
from pathlib import Path
import litellm

CATALOG_PATH = Path("scenario_catalog.json")

SYSTEM_PROMPT = """
You are an expert in STEM fields. Generate a list of new, unique subtopics for the given domain and topic.
Respond with a JSON array of strings, each being a subtopic.
Ensure they are natural, educational, and suitable for Python coding challenges.
"""

def generate_subtopics(domain: str, topic: str, count: int = 5) -> List[str]:
    user_prompt = f"Generate {count} subtopics for domain: {domain}, topic: {topic}."
    response = litellm.completion(
        model=os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.0,
    )
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)

def main(add_count: int = 10, target_domain: Optional[str] = None) -> None:
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}")
        sys.exit(1)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Select random existing entries to expand
    for _ in range(add_count):
        domain_entry = random.choice(catalog) if not target_domain else next(d for d in catalog if d["domain"] == target_domain)
        topic_entry = random.choice(domain_entry["topics"])
        new_subtopics = generate_subtopics(domain_entry["domain"], topic_entry["topic"], 1)
        for sub in new_subtopics:
            topic_entry["subtopics"].append({"subtopic": sub, "used": False})

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Added {add_count} new subtopics to the catalog.")

if __name__ == "__main__":
    import random
    from typing import Optional
    add_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    target_domain = sys.argv[2] if len(sys.argv) > 2 else None
    main(add_count, target_domain)
