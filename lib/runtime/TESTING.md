# Runtime Testing Guide

This guide covers how to test the QuestFoundry runtime with real LLM providers.

## Quick Start

### Prerequisites

```bash
cd lib/runtime
poetry install
```

### Option 1: Test with OpenAI

```bash
export QF_LLM_PROVIDER="openai"
export QF_DEFAULT_MODEL="gpt-3.5-turbo"
export OPENAI_API_KEY="sk-..."

# Run manual tests
poetry run python tests/test_manual_llm.py

# Or test CLI directly
poetry run qf write "a tense cargo bay scene"
```

### Option 2: Test with Anthropic

```bash
export QF_LLM_PROVIDER="anthropic"
export QF_DEFAULT_MODEL="claude-3-5-haiku-20241022"
export ANTHROPIC_API_KEY="sk-ant-..."

# Run manual tests
poetry run python tests/test_manual_llm.py

# Or test CLI directly
poetry run qf write "the captain discovers missing fuel"
```

---

## Test Suite Overview

The `tests/test_manual_llm.py` script runs 5 test categories:

### 1. **OpenAI Adapter Test**
- Tests direct OpenAI API integration
- Verifies ChatOpenAI wrapper
- Validates response format

### 2. **Anthropic Adapter Test**
- Tests direct Anthropic API integration
- Verifies ChatAnthropic wrapper
- Validates response format

### 3. **NodeFactory LLM Integration**
- Tests NodeFactory with real LLM calls
- Verifies provider selection logic
- Checks artifact creation

### 4. **Story Spark Loop (End-to-End)**
- Tests complete loop execution
- Validates Showrunner orchestration
- Checks quality bar updates
- Verifies result translation

### 5. **Provider Switching**
- Tests dynamic provider selection
- Validates environment variable overrides
- Ensures both providers work

---

## Expected Results

### Successful Test Output

```
============================================================
QuestFoundry Runtime - Manual LLM Integration Tests
============================================================

📋 Current provider: openai
📋 OPENAI_API_KEY: ✓
📋 ANTHROPIC_API_KEY: ✓

=== Testing OpenAI Adapter ===
📤 Sending: 'Say hello in exactly 3 words'
📥 Received: Hello to you
✅ OpenAI adapter working!

=== Testing Anthropic Adapter ===
📤 Sending: 'Say hello in exactly 3 words'
📥 Received: Hello to you
✅ Anthropic adapter working!

=== Testing NodeFactory LLM Integration ===
📋 Testing with provider: openai
📋 TU ID: TU-2025-042
🔧 Creating plotwright node...
🚀 Executing plotwright with real LLM...
✅ NodeFactory created 1 artifacts
   • artifacts.hot.outputs.plotwright: [Plotwright] Execute role...
✅ NodeFactory LLM integration working!

=== Testing Story Spark Loop (End-to-End) ===
📋 Provider: openai
📝 Scene: A tense confrontation in the cargo bay
🚀 Executing story_spark loop...
✅ Loop completed successfully!
📋 TU ID: TU-2025-043
📊 Artifacts: 5

✓ Completed story_spark
Trace Unit: TU-2025-043
Status: hot-proposed
...

=== Testing Provider Switching ===
📋 Test 1 - Provider: openai, Model: gpt-3.5-turbo
📋 Test 2 - Provider: anthropic, Model: claude-3-5-haiku-20241022
✅ Provider switching working!

============================================================
TEST SUMMARY
============================================================
✅ PASS - OpenAI Adapter
✅ PASS - Anthropic Adapter
✅ PASS - NodeFactory LLM
✅ PASS - Story Spark Loop
✅ PASS - Provider Switching

📊 Results: 5/5 tests passed
🎉 All tests passed!
```

---

## CLI Testing

### Test Write Command

```bash
# Simple test
poetry run qf write "a short scene"

# With mode flag
poetry run qf write "an action sequence" --mode production
```

**Expected Output**:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Writing new scene       ┃
┃ a short scene           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛

questfoundry.runtime.core.node_factory - INFO - Selected LLM provider: openai, model: gpt-3.5-turbo
questfoundry.runtime.cli.showrunner - INFO - Executing request: write a short scene
questfoundry.runtime.core.graph_factory - INFO - Creating loop graph: story_spark
...

┏━━━━━━━━━━━━ ✓ Success ━━━━━━━━━━━━┓
┃ ✓ Completed story_spark          ┃
┃ Trace Unit: TU-2025-044          ┃
┃ Status: hot-proposed             ┃
┃                                  ┃
┃ Artifacts created:               ┃
┃   • artifacts.hot.outputs...    ┃
┃                                  ┃
┃ Next steps:                      ┃
┃   • Run 'qf review story'        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Test Review Command

```bash
poetry run qf review story
```

### Test Utility Commands

```bash
# List loops
poetry run qf list-loops

# List roles
poetry run qf list-roles
```

---

## Troubleshooting

### Error: "API key not found"

**Problem**: Missing or invalid API key

**Solution**:
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Set the appropriate key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Error: "Module not found: langchain_openai"

**Problem**: Dependencies not installed

**Solution**:
```bash
cd lib/runtime
poetry install
```

### Error: "Wrong provider being used"

**Problem**: Provider environment variable not set

**Solution**:
```bash
# Force provider
export QF_LLM_PROVIDER="openai"  # or "anthropic"
```

### Error: "Rate limit exceeded"

**Problem**: Too many API requests

**Solution**:
- Wait a few seconds and retry
- Use a different model
- Check API quota/billing

### Error: "Graph compilation fails"

**Problem**: YAML validation or graph structure issue

**Solution**:
```bash
# Run schema validation
poetry run qf test-schema

# Check specific loop
poetry run qf test-graph story_spark
```

---

## Performance Testing

### Test All 10 Loops

```bash
# Create a script to test all loops
export QF_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

for loop in story_spark hook_harvest lore_deepening codex_expansion \
            style_tune_up art_touch_up audio_pass translation_pass \
            narration_dry_run binding_run; do
  echo "Testing $loop..."
  poetry run qf test-graph $loop
done
```

### Measure Execution Time

```bash
time poetry run qf write "test scene"
```

---

## Integration with CI/CD

### Skip LLM Tests in CI

The manual LLM tests are automatically skipped if API keys are not present:

```bash
# In CI/CD, just run unit tests
poetry run pytest tests/test_core_integration.py -v
```

### Run LLM Tests in CI (Optional)

Add API keys as secrets and run:

```bash
export OPENAI_API_KEY="${{ secrets.OPENAI_API_KEY }}"
poetry run python tests/test_manual_llm.py
```

---

## Next Steps

After successful testing:

1. **Verify all loops work** - Test each of the 10 loops
2. **Test multi-loop workflows** - Try narration_dry_run (requires binding_run)
3. **Test error scenarios** - Invalid inputs, missing artifacts, etc.
4. **Performance testing** - Measure execution times
5. **Move to Phase 6** - Python library layer

---

## Quick Reference

```bash
# Setup
cd lib/runtime && poetry install

# Test with OpenAI
export QF_LLM_PROVIDER="openai" OPENAI_API_KEY="sk-..."
poetry run python tests/test_manual_llm.py

# Test with Anthropic
export QF_LLM_PROVIDER="anthropic" ANTHROPIC_API_KEY="sk-ant-..."
poetry run python tests/test_manual_llm.py

# CLI test
poetry run qf write "test scene"

# List commands
poetry run qf --help
```
