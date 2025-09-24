"""Example: generate a file-edit debugging task and print locations.

Run with:
  uv run python examples/case3_file_edit.py
"""

import sys
from pathlib import Path

# Ensure package import when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_debug.pipeline import DebugConversationPipeline


def main() -> int:
    target = Path("sample_conversations") / "placeholder.py"
    target.parent.mkdir(parents=True, exist_ok=True)

    pipeline = DebugConversationPipeline()
    convo = pipeline.generate_file_edit_task(bug_file_path=target)

    print("Bug file:", convo.bug_file_path)
    print("Tests:", (Path(convo.bug_file_path).parent / f"test_{convo.module_name}.py"))
    print("Runner:", (Path(convo.bug_file_path).parent / "run_tests.py"))

    # Optionally, write the conversation JSON next to the files
    json_path = Path(convo.bug_file_path).with_suffix("")
    json_out = Path(str(json_path) + "_conversation.json")
    json_out.write_text(convo.to_json(), encoding="utf-8")
    print("Conversation JSON:", json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


