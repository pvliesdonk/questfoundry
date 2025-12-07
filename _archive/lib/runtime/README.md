# QuestFoundry Runtime

The QuestFoundry runtime engine transforms YAML role and loop definitions into executable LangGraph StateGraphs.

## Features

- **Cartridge Architecture**: Specifications ARE executable code
- **Role Profiles**: Complete role definitions with behavior and interfaces
- **Loop Patterns**: Executable loop state machines
- **Quality Gates**: Reusable quality bar validators
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Ollama, LiteLLM

## Installation

```bash
pip install questfoundry-runtime
```

## Quick Start

```bash
# Basic usage
qf ask "Create a mystery story about a haunted lighthouse"

# With local tracing
qf ask "Create a story" --trace --trace-file trace.log

# With verbose output
qf ask "Create a story" -v
```

## Debugging and Tracing

### Local Trace Mode

Use `--trace` to capture agent-to-agent communication locally:

```bash
qf ask "Create a story" --trace                    # Console output only
qf ask "Create a story" --trace --trace-file out.log  # Console + file
```

This shows role prompts, LLM iterations, tool calls, and results.

### LangSmith Integration (Cloud Tracing)

For comprehensive tracing with visualization, latency metrics, and chain replay, enable [LangSmith](https://smith.langchain.com):

```bash
# Set environment variables (works automatically with LangChain)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
export LANGCHAIN_PROJECT=questfoundry

# Run as normal - all LLM calls are automatically traced
qf ask "Create a story"
```

No code changes required - LangChain auto-detects LangSmith configuration.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QF_SPEC_SOURCE` | Spec source: `monorepo`, `bundled`, `download`, `auto` | `auto` |
| `QF_LLM_PROVIDER` | Preferred LLM provider | auto-detect |
| `QF_REACT_MAX_ITERATIONS` | Max tool call iterations | `5` |
| `QF_REACT_DEBUG` | Enable ReAct debug logging | `false` |
| `QF_PROMPT_ERROR_THRESHOLD` | Error on prompts exceeding (chars) | `32000` |
| `QF_PROMPT_WARNING_THRESHOLD` | Warn on prompts exceeding (chars) | `16000` |
| `QF_OLLAMA_NUM_CTX` | Ollama context window size | `32768` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup instructions.

## Testing

See [TESTING.md](TESTING.md) for testing instructions.

## Documentation

Full documentation is available in the [QuestFoundry specification](../../spec/README.md).
