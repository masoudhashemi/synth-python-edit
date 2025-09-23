import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import datetime
import json

from synthetic_debug.pipeline import DebugConversationPipeline

example_data = {
    "prompt": "Implement a Python function that computes the factorial of a non-negative integer.",
    "correct_code": """def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result""",
    "input_outputs": [
        {"input": 0, "output": 1},
        {"input": 1, "output": 1},
        {"input": 5, "output": 120},
        {"input": [3], "output": 6}  # Example with list for single arg (equivalent to factorial(3))
    ],
    "function_name": "factorial",
    "module_name": "math_utils",
    "domain": "Mathematics",
    "topic": "Recursion",
    "subtopic": "Factorials",
    "summary": "Debug factorial function using examples",
    "solution_outline": "Use recursive calls with base case for 0."
}

if __name__ == "__main__":
    pipeline = DebugConversationPipeline()
    conversation = pipeline.generate_from_examples(**example_data)
    
    # Save to sample_conversations
    root_dir = Path(__file__).parent.parent
    output_dir = root_dir / "sample_conversations"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"conversation_{timestamp}_input_output.json"
    output_path = output_dir / filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(conversation.to_json())
    
    print(f"Generated conversation saved to: {output_path}")
