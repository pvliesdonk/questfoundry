# Model Comparison Test Runs (2026-01-15)

Example pipeline runs from the model comparison analysis documented in
[docs/analysis/model-comparison-2026-01-15.md](../../docs/analysis/model-comparison-2026-01-15.md).

## Runs

| Directory | Model | Mode | Stages |
|-----------|-------|------|--------|
| `run-1-ni` | ollama/qwen3:4b-instruct-32k | non-interactive | DREAM → BRAINSTORM → SEED |
| `run-2-ni` | openai/gpt-4o | non-interactive | DREAM → BRAINSTORM → SEED |
| `run-3-ni` | ollama/gpt-oss:20b | non-interactive | DREAM → BRAINSTORM → SEED |
| `run-1-int` | ollama/qwen3:4b-instruct-32k | interactive | DREAM → BRAINSTORM → SEED |
| `run-2-int` | openai/gpt-4o | interactive | DREAM → BRAINSTORM → SEED |

## Contents

Each run directory contains:

```
run-X-{ni,int}/
├── artifacts/
│   ├── dream.yaml      # DREAM stage output
│   ├── brainstorm.yaml # BRAINSTORM stage output
│   └── seed.yaml       # SEED stage output
├── logs/
│   ├── debug.jsonl     # Structured application logs
│   └── llm_calls.jsonl # Full LLM request/response logs
└── project.yaml        # Project configuration
```

## Key Findings

- **qwen3:4b** (run-1) produced the best quality output despite being the smallest model
- **gpt-4o** (run-2) produced the thinnest output, only 6 beats vs 16 for qwen3:4b
- **gpt-oss:20b** (run-3) performed well but with some schema compliance issues
- Interactive mode improved gpt-4o's BRAINSTORM but not SEED output

## Reproduction

```bash
# Non-interactive runs
uv run qf dream --no-interactive --provider ollama/qwen3:4b-instruct --project run-1-ni "classic murder mystery"
uv run qf brainstorm --no-interactive --provider ollama/qwen3:4b-instruct --project run-1-ni
uv run qf seed --no-interactive --provider ollama/qwen3:4b-instruct --project run-1-ni

# Interactive runs (requires human interaction)
uv run qf dream --provider ollama/qwen3:4b-instruct --project run-1-int "classic murder mystery"
uv run qf brainstorm --provider ollama/qwen3:4b-instruct --project run-1-int
uv run qf seed --provider ollama/qwen3:4b-instruct --project run-1-int
```

## User Prompt

All runs used the same initial prompt:
> "A genre-typical Clue style Agatha Christie Murder mystery"
