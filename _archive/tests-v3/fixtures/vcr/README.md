# VCR Test Fixtures

VCR (Video Cassette Recorder) style testing allows replaying recorded LLM interactions for deterministic tests.

## Concept

1. **Record**: Run a workflow with `--log` flag, capture LLM requests/responses
2. **Extract**: Pull out a single role session from the log
3. **Mock**: Use the recorded responses in tests instead of calling real LLM
4. **Replay**: Run tests deterministically without LLM API calls

## Fixture Format

Each fixture is a JSON file containing a complete role session:

```json
{
  "metadata": {
    "role": "gatekeeper",
    "session_id": "abc123",
    "recorded_at": "2025-12-10T12:00:00Z",
    "model": "qwen3:32b",
    "purpose": "Test GK validation pass scenario"
  },
  "initial_state": {
    "hot_store": {
      "scene_001": {"type": "scene", "title": "Opening", "content": "..."}
    },
    "task": "Validate the scene against quality bars"
  },
  "interactions": [
    {
      "iteration": 1,
      "request": {
        "messages": [
          {"role": "system", "content": "You are the Gatekeeper..."},
          {"role": "user", "content": "Validate scene_001..."}
        ]
      },
      "response": {
        "content": "I'll validate the scene against quality bars.",
        "tool_calls": [
          {
            "name": "consult_schema",
            "arguments": {"artifact_type": "gatecheck_report"}
          }
        ]
      }
    },
    {
      "iteration": 2,
      "request": {
        "messages": ["... previous + tool result"]
      },
      "response": {
        "content": "Creating gatecheck report...",
        "tool_calls": [
          {
            "name": "create_gatecheck_report",
            "arguments": {"artifact_id": "scene_001", "passed": true}
          }
        ]
      }
    }
  ],
  "final_result": {
    "status": "completed",
    "artifacts_created": ["gatecheck_001"]
  }
}
```

## Extracting from Logs

Use `jq` to extract a session from `llm.jsonl`:

```bash
# Get session ID
jq 'select(.event == "role_session_start") | {session_id, role}' logs/llm.jsonl

# Extract full session
SESSION_ID="abc123"
jq --arg sid "$SESSION_ID" 'select(.session_id == $sid)' logs/llm.jsonl > fixture.jsonl

# Convert to fixture format (script needed)
python scripts/extract_vcr_fixture.py --session $SESSION_ID logs/llm.jsonl > fixtures/gk_pass.json
```

## Using in Tests

```python
import pytest
from pathlib import Path
from questfoundry.runtime.testing import VCRClient

@pytest.fixture
def vcr_gk_pass():
    fixture_path = Path(__file__).parent / "fixtures/vcr/gk_validation_pass.json"
    return VCRClient.from_fixture(fixture_path)

def test_gatekeeper_validates_scene(vcr_gk_pass):
    """Test GK validates scene using recorded interactions."""
    from questfoundry.runtime.executor import RoleExecutor

    executor = RoleExecutor(
        role="gatekeeper",
        llm_client=vcr_gk_pass,  # Mock client returns recorded responses
    )

    result = executor.run(
        task="Validate scene_001",
        state=vcr_gk_pass.initial_state,
    )

    assert result.status == "completed"
    assert "gatecheck" in result.artifacts_created[0]
```

## Benefits

1. **Deterministic**: Same test, same result, every time
2. **Fast**: No LLM API calls, millisecond execution
3. **Cost-free**: No API usage during test runs
4. **Debuggable**: Can inspect exact recorded conversation
5. **Regressionable**: Re-record when prompts change, compare outputs

## When to Use

| Scenario | Use VCR? |
|----------|----------|
| Unit testing role behavior | Yes |
| Testing tool validation | Yes |
| Integration testing workflow chain | Maybe (mock key roles) |
| E2E testing full workflow | No (use real LLM with checkpoints) |
| Debugging prompt changes | Yes (compare before/after) |

## Recording New Fixtures

1. Run workflow with logging:

   ```bash
   qf ask --project test --log "your prompt"
   ```

2. Find the session you want:

   ```bash
   jq 'select(.event == "role_session_start")' test/logs/llm.jsonl
   ```

3. Extract and save:

   ```bash
   python scripts/extract_vcr_fixture.py --session SESSION_ID test/logs/llm.jsonl \
     > tests/fixtures/vcr/descriptive_name.json
   ```

4. Document the fixture's purpose in this README.

## Fixture Inventory

| File | Role | Purpose |
|------|------|---------|
| `gk_validation_pass.json` | Gatekeeper | GK validates scene, creates passing report |
| `gk_validation_fail.json` | Gatekeeper | GK fails scene, lists issues |
| `lk_promote_scene.json` | Lorekeeper | LK promotes scene to cold_store |
| `pw_create_structure.json` | Plotwright | PW creates scene topology |
