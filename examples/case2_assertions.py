import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import datetime
import json

from synthetic_debug.pipeline import DebugConversationPipeline

example_data = {
    "problem": "Implement a function to compute the sum of two numbers.",
    "correct_code": """def add(a, b):
    return a + b""",
    "assertion_functions": [
        """def test_add_positive():
    assert add(1, 2) == 3""",
        """def test_add_zero():
    assert add(0, 0) == 0""",
        """def test_add_negative():
    assert add(-1, 1) == 0"""
    ],
    "module_name": "math_utils",
    "domain": "Mathematics",
    "topic": "Arithmetic",
    "subtopic": "Addition",
    "summary": "Debug addition function using assertions",
    "solution_outline": "Simply return the sum of parameters."
}

if __name__ == "__main__":
    pipeline = DebugConversationPipeline()
    conversation = pipeline.generate_from_assertions(**example_data)
    
    # Save to sample_conversations
    root_dir = Path(__file__).parent.parent
    output_dir = root_dir / "sample_conversations"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"conversation_{timestamp}_assertions.json"
    output_path = output_dir / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(conversation.to_json())
    
    print(f"Generated conversation saved to: {output_path}")
