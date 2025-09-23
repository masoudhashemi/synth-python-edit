# Examples for DebugConversationPipeline

This folder contains example scripts demonstrating the new pipeline options for generating debug conversations from provided data.

## Prerequisites

- Ensure you have the project set up and dependencies installed.
- These examples use the `DebugConversationPipeline` from `synthetic_debug.pipeline`.

## Running the Examples

Run the scripts from the project root using `uv run python` (preferred for this project). The scripts include a sys.path adjustment to handle imports from the synthetic_debug package.

### Case 1: Input-Output Examples

```bash
uv run python examples/case1_input_output.py
```

This generates a debug conversation for a factorial function using input-output pairs and saves the JSON to a timestamped file in sample_conversations/. It will print the path to the saved file.

### Case 2: Assertion Functions

```bash
uv run python examples/case2_assertions.py
```

This generates a debug conversation for an addition function using assertion functions and saves the JSON to a timestamped file in sample_conversations/. It will print the path to the saved file.

## Customization

You can modify the `example_data` dictionaries in the scripts to test with different inputs. Ensure the provided correct code passes the derived tests before bug injection.

## Input Data Format

The `example_data` dictionary for each script must follow specific formats. Below are the required and optional fields for each case.

### Case 1: Input-Output Examples (case1_input_output.py)

- **prompt** (required, str): The problem description or prompt containing the questions.
- **correct_code** (required, str): The correct Python code as a string.
- **input_outputs** (required, list of dicts): List of input-output pairs. Each dict should have:
  - "input": The input value(s) – can be a single value or a list/tuple for multiple arguments.
  - "output": The expected output value.
- **function_name** (required, str): The name of the main function to test in the code.
- **module_name** (required, str): The module name used for file and import purposes.
- **domain** (optional, str, default: "General"): The domain category.
- **topic** (optional, str, default: "Programming"): The topic category.
- **subtopic** (optional, str, default: "Debugging"): The subtopic category.
- **summary** (optional, str, default: "Debugging task based on input-output examples"): A brief summary.
- **solution_outline** (optional, str, default: ""): Outline of the solution approach.

### Case 2: Assertion Functions (case2_assertions.py)

- **problem** (required, str): The problem description.
- **correct_code** (required, str): The correct Python code as a string.
- **assertion_functions** (required, list of str): List of strings, each defining an assertion function (e.g., "def test_add(): assert add(1, 2) == 3").
- **module_name** (required, str): The module name used for file and import purposes.
- **domain** (optional, str, default: "General"): The domain category.
- **topic** (optional, str, default: "Programming"): The topic category.
- **subtopic** (optional, str, default: "Debugging"): The subtopic category.
- **summary** (optional, str, default: "Debugging task based on assertion functions"): A brief summary.
- **solution_outline** (optional, str, default: ""): Outline of the solution approach.
