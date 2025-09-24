# Synthetic Debugging Conversation Pipeline

This project builds synthetic debugging transcripts by asking an LLM (via [LiteLLM](https://github.com/BerriAI/litellm)) to invent STEM-heavy Python modules—orbital mechanics, derivative pricing, PDE solvers, and more—alongside matching unit tests and a deliberately buggy variant. The pipeline verifies the healthy implementation, swaps in the broken version to capture failing console output, and assembles the full debugging conversation describing the issue and fix.

## Requirements

- Python 3.9+
- `litellm` installed locally (`pip install litellm`)
- `PyYAML` for parsing model-written YAML (`pip install PyYAML`)
- An API key configured for the upstream provider that LiteLLM will call (e.g. `export OPENAI_API_KEY=...`)

## Running the tests

Tests rely on a stubbed LLM so they can run offline:

```bash
python3 run_unit_tests.py
```

## Generating a conversation

Invoke the CLI to produce a fresh debugging transcript. The command below uses the default model name (`gpt-4o-mini`) unless overridden via `--model` or the `LITELLM_MODEL` environment variable.

```bash
python3 generate_debug_conversation.py --model gpt-4o-mini --catalog scenario_catalog.json --state .catalog_state.json
```

The LLM response is authored as YAML (per the prompt) and the pipeline parses it before packaging the final conversation as JSON.

Use `--output` to write the JSON payload to disk:

```bash
python3 generate_debug_conversation.py --output sample_conversation.json
```

Generate multiple unique conversations in a batch (files saved under `conversations/` by default):

```bash
python3 generate_conversation_batch.py --count 10 --catalog scenario_catalog.json --state .catalog_state.json
```

If the LLM returns a scenario where the injected bug does not fail the tests, the batch script retries automatically (configurable via `--retries`) before skipping that attempt and moving on.

Both commands draw from `scenario_catalog.json`, which enumerates all domain/topic/subtopic combinations. A state file (e.g. `.catalog_state.json`) keeps track of previously generated entries so each conversation remains unique across runs. The catalog also drives the prompt given to LiteLLM so responses stay aligned with the selected context.

### Re-validating saved conversations

To extract the code artefacts from an existing JSON payload and re-run the tests:

```bash
python3 validate_conversation.py conversations/your_file.json --dump-dir extracted
```

This writes the clean and buggy modules, test file, and runner to `extracted/your_file/` while verifying that the correct code still passes and the buggy version fails the suite.

The JSON payload contains:

- Original problem description, module code, and accompanying unit tests
- Buggy implementation supplied by the LLM
- Captured failing test output
- A conversation transcript walking through the diagnosis and resolution
- Non-trivial business logic with multiple cooperating functions and validations, ensuring the debugging task has real depth
- A STEM-focused problem statement plus a step-by-step solution outline ahead of the implementation
- A dedicated bug-injection stage: after validating the clean build, the pipeline applies multiple strategies (LLM-provided variants, AST operator inversions, numeric perturbations) and only accepts a mutant once the tests demonstrably fail

Each run produces a brand-new scenario because the LLM invents the domain, topic, and code on the fly.

## File-edit debugging tasks (buggy code only on disk)

In addition to embedding the buggy implementation inside the conversation JSON, the pipeline supports a file-edit workflow where only the buggy code is written to a real file on disk. The conversation then instructs the developer/agent to open that file, inspect the included failing unittest output, and fix the file content in place until all tests pass.

### Generate a file-edit task via CLI

```bash
python3 generate_file_edit_task.py \
  --dir file_edit_tasks \
  --filename placeholder.py \
  --suppress-buggy-code
```

This creates three files under `file_edit_tasks/`:

- `<module_name>.py`: the buggy module to edit and fix
- `test_<module_name>.py`: the unit tests
- `run_tests.py`: the test runner

It also writes a conversation JSON (path printed at the end) that references these absolute paths and includes the failing test output, but does not inline the buggy code.

To run tests locally while fixing the file:

```bash
cd file_edit_tasks
python3 run_tests.py
```

### Programmatic API

```python
from pathlib import Path
from synthetic_debug.pipeline import DebugConversationPipeline

pipeline = DebugConversationPipeline()
convo = pipeline.generate_file_edit_task(bug_file_path=Path("/tmp/debug_task/placeholder.py"))

print(convo.bug_file_path)  # Absolute path to the buggy module on disk
```
