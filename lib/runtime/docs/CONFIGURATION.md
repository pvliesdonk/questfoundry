# QuestFoundry Runtime Configuration

This document describes all configurable options for the QuestFoundry runtime.

## Overview

Configuration is loaded from multiple sources in priority order (highest first):

1. **CLI arguments** - Command-line options passed to `qf` commands
2. **Environment variables** - Variables with `QF_` prefix
3. **`.env` file** - In current directory or project root
4. **Config file** - `questfoundry.yaml`, `questfoundry.toml`, or `~/.config/questfoundry/config.yaml`
5. **Built-in defaults** - Sensible defaults for all options

## Configuration File

Create `questfoundry.yaml` in your project root:

```yaml
runtime:
  recursion_limit: 50
  max_failures: 3
  max_iterations: 5
  max_parallel_roles: 4
  debug: false

llm:
  default_model: "claude-3-5-sonnet-20241022"
  default_temperature: 0.7
  ollama_host: "http://localhost:11434"

memory:
  prompt_error_threshold: 32000
  memory_cap: 8000

paths:
  project_dir: "~/.questfoundry/projects"
  spec_source: "auto"

logging:
  level: "INFO"
  show_time: true

network:
  spec_fetch_timeout: 30
  search_timeout: 15.0
```

Or use TOML format (`questfoundry.toml`):

```toml
[runtime]
recursion_limit = 50
max_failures = 3
debug = false

[llm]
default_model = "claude-3-5-sonnet-20241022"
default_temperature = 0.7

[paths]
project_dir = "~/.questfoundry/projects"
```

## Environment Variables

All settings can be overridden via environment variables with the `QF_` prefix.
Nested settings use double underscore (`__`) as delimiter.

### Runtime Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_RUNTIME__RECURSION_LIMIT` | `50` | Maximum graph iterations before forced termination |
| `QF_RUNTIME__MAX_FAILURES` | `3` | Maximum consecutive tool call failures before giving up |
| `QF_RUNTIME__MAX_ITERATIONS` | `5` | Maximum tool iterations per role execution |
| `QF_RUNTIME__MAX_PARALLEL_ROLES` | `4` | Maximum roles to execute in parallel |
| `QF_RUNTIME__MAX_PING_PONG` | `3` | Maximum identical message exchanges before intervention |
| `QF_RUNTIME__MAX_ROLE_EXECUTIONS` | `15` | Maximum times a single role can execute |
| `QF_RUNTIME__EXECUTION_RESET_THRESHOLD` | `20` | Total executions before halving per-role counts |
| `QF_RUNTIME__MAX_CONSECUTIVE_ROLE_EXECUTIONS` | `3` | Maximum consecutive executions of the same role |
| `QF_RUNTIME__DEBUG` | `false` | Enable debug mode with verbose output |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_LLM__DEFAULT_MODEL` | `claude-3-5-sonnet-20241022` | Default model when not specified by role |
| `QF_LLM__DEFAULT_MODEL_TIER` | `creative-writing` | Default model tier for role capability matching |
| `QF_LLM__DEFAULT_TEMPERATURE` | `0.7` | Default temperature (0.0-2.0) |
| `QF_LLM__DEFAULT_MAX_TOKENS` | `4096` | Default maximum tokens for generation |
| `QF_LLM__PROVIDER` | (auto-detect) | Preferred LLM provider |
| `QF_LLM__MODEL_TIERS_CONFIG` | (auto) | Path to custom model tiers YAML |
| `QF_LLM__OLLAMA_HOST` | `http://localhost:11434` | Ollama server endpoint URL |
| `QF_LLM__OLLAMA_NUM_CTX` | `32768` | Ollama context window size |
| `QF_LLM__LITELLM_API_BASE` | (none) | LiteLLM proxy API base URL |
| `QF_LLM__LITELLM_API_KEY` | (none) | LiteLLM API key |

**Note:** API keys should be set via standard environment variables:
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `GOOGLE_API_KEY` - Google AI API key
- `OLLAMA_HOST` - Ollama server URL (also supported for compatibility)

### Memory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_MEMORY__PROMPT_ERROR_THRESHOLD` | `32000` | Prompt size (chars) above which to log error |
| `QF_MEMORY__PROMPT_WARNING_THRESHOLD` | `16000` | Prompt size (chars) above which to log warning |
| `QF_MEMORY__MEMORY_CAP` | `8000` | Maximum characters for prior conversation history |
| `QF_MEMORY__SUMMARIZE_MESSAGES_THRESHOLD` | `20` | Messages after which to summarize conversation |
| `QF_MEMORY__SUMMARIZE_CHARS_THRESHOLD` | `12000` | Characters after which to summarize |

### Path Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_PATHS__PROJECT_DIR` | `~/.questfoundry/projects` | Base directory for project storage |
| `QF_PATHS__PROJECT_ID` | `default` | Default project identifier |
| `QF_PATHS__SPEC_SOURCE` | `auto` | Spec source: `auto`, `monorepo`, `bundled`, `download` |
| `QF_PATHS__SPEC_CACHE_DIR` | `~/.cache/questfoundry/spec` | Directory for downloaded spec cache |
| `QF_PATHS__CONFIG_FILE` | (auto-detect) | Path to config file |

### Logging Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_LOGGING__LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `QF_LOGGING__SHOW_TIME` | `true` | Show timestamp in console log output |
| `QF_LOGGING__SHOW_PATH` | `false` | Show file path and line number |
| `QF_LOGGING__RICH_TRACEBACKS` | `true` | Use Rich formatted tracebacks |
| `QF_LOGGING__LOG_FILE` | (none) | Path to log file |
| `QF_LOGGING__STRUCTURED_LOGS_DIR` | (none) | Directory for structured JSONL logs |
| `QF_LOGGING__HTTPX_LEVEL` | `WARNING` | Log level for httpx library |
| `QF_LOGGING__OPENAI_LEVEL` | `WARNING` | Log level for openai library |
| `QF_LOGGING__ANTHROPIC_LEVEL` | `WARNING` | Log level for anthropic library |

### Network Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QF_NETWORK__SPEC_FETCH_TIMEOUT` | `30` | Timeout (seconds) for spec fetch API calls |
| `QF_NETWORK__SPEC_DOWNLOAD_TIMEOUT` | `120` | Timeout (seconds) for spec archive download |
| `QF_NETWORK__SEARCH_TIMEOUT` | `15.0` | Timeout (seconds) for web search requests |
| `QF_NETWORK__SEARXNG_URL` | (none) | SearXNG instance URL for web search |
| `QF_NETWORK__SEARXNG_API_TOKEN` | (none) | SearXNG API token |
| `QF_NETWORK__ELEVENLABS_TIMEOUT` | `30.0` | Timeout for ElevenLabs API calls |
| `QF_NETWORK__ELEVENLABS_DEFAULT_VOICE` | `21m00Tcm4TlvDq8ikWAM` | Default ElevenLabs voice ID |
| `QF_NETWORK__ELEVENLABS_MODEL` | `eleven_turbo_v2` | Default ElevenLabs model |
| `QF_NETWORK__ELEVENLABS_STABILITY` | `0.75` | Voice stability (0.0-1.0) |
| `QF_NETWORK__ELEVENLABS_SIMILARITY` | `0.75` | Voice similarity boost (0.0-1.0) |
| `QF_NETWORK__DALLE_MODEL` | `gpt-image-1` | DALL-E model for image generation |
| `QF_NETWORK__DALLE_SIZE` | `1024x1024` | Default DALL-E image size |
| `QF_NETWORK__GEMINI_IMAGE_MODEL` | `imagen-3.0-generate-002` | Gemini/Imagen model |

## CLI Options

Global options available on all commands:

```bash
qf --verbose/-v     # Increase verbosity (-v INFO, -vv DEBUG, -vvv full)
```

Options for `qf ask`:

```bash
qf ask "message" \
  --verbose/-v              # Show detailed execution info
  --trace/-t                # Enable trace mode (show agent communication)
  --log-file/-l NAME        # Write logs to NAME-debug.log and NAME-trace.log
  --recursion-limit/-r N    # Max graph iterations (default: 50)
  --provider/-p PROVIDER    # Preferred LLM provider
  --project PROJECT_ID      # Project identifier for storage
  --project-dir DIR         # Base directory for projects
  --structured-logs DIR     # Directory for JSONL logs
```

## .env File

Create a `.env` file in your project root for local development:

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Runtime configuration
QF_RUNTIME__DEBUG=true
QF_RUNTIME__MAX_FAILURES=5

# LLM configuration
QF_LLM__DEFAULT_MODEL=claude-3-5-sonnet-20241022
QF_LLM__OLLAMA_HOST=http://localhost:11434

# Logging
QF_LOGGING__LEVEL=DEBUG
```

## Programmatic Access

You can access configuration programmatically in Python:

```python
from questfoundry.runtime.config import get_settings, reload_settings

# Get cached settings instance
settings = get_settings()

# Access configuration values
print(settings.runtime.recursion_limit)  # 50
print(settings.llm.default_model)  # "claude-3-5-sonnet-20241022"
print(settings.paths.project_dir)  # "~/.questfoundry/projects"

# Reload settings after environment changes
settings = reload_settings()
```

## Configuration Categories

### Runtime
Execution limits and safety thresholds that control how the runtime executes roles
and prevents infinite loops or runaway executions.

### LLM
Model selection, provider configuration, and generation parameters. Controls which
models are used and how they generate responses.

### Memory
Prompt and conversation memory limits. Prevents context overflow and controls how
much conversation history is retained between role executions.

### Paths
File system paths and locations for project storage, spec resolution, and cache.

### Logging
Log levels, output destinations, and structured logging configuration.

### Network
Timeouts and endpoints for external services (GitHub, SearXNG, ElevenLabs, etc.).

## Best Practices

1. **Use environment variables for secrets** - API keys should always be in environment
   variables or `.env` files, never in config files that might be committed to git.

2. **Use config files for project-specific settings** - Settings like model preferences
   and project paths work well in `questfoundry.yaml`.

3. **Use CLI options for one-off changes** - Temporary overrides like `--verbose` or
   `--provider` are best as CLI options.

4. **Start with defaults** - The defaults are tuned for typical use cases. Only change
   what you need.

5. **Use `.env` for local development** - Keep development-specific settings in `.env`
   which is typically gitignored.
