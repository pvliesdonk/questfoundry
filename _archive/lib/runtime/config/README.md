# Runtime Configuration Files

This directory contains runtime configuration files that bridge specification-layer definitions with actual runtime implementations.

## Files

### `tool_mappings.yaml`

Maps abstract capabilities (from `spec/05-definitions/capabilities.yaml`) to concrete runtime tool implementations.

**Purpose:**

- Defines how each abstract capability maps to one or more provider implementations
- Specifies provider priority ordering and fallback strategies
- Establishes internal-only tools that are never exposed to agents
- Configures availability checks and runtime behavior

**Structure:**

```
tool_mappings.yaml
├── metadata
├── external_capability_mappings
│   ├── image_generation
│   │   └── providers: [dalle3, stable_diffusion_api, midjourney_api, stub]
│   ├── audio_synthesis
│   │   └── providers: [suno_api, elevenlabs_api, bark_local, stub]
│   ├── web_search
│   │   └── providers: [tavily, perplexity, serper, google_custom_search, stub]
│   └── ... 11 more external capabilities
├── knowledge_capability_mappings
│   ├── consult_specs
│   ├── consult_project_docs
│   └── search_artifacts
├── internal_tools (NEVER exposed to agents)
│   ├── protocol_operations
│   ├── state_operations
│   ├── validation_operations
│   └── orchestration_operations
├── runtime_integration
└── tool_exposure_rules
```

## Capability Mappings

### External Capabilities (14 total)

Capabilities requiring external services or tools:

#### Visual Production

- **image_generation**: DALL-E 3, Stable Diffusion, Midjourney, Stub
- **image_editing**: Stable Diffusion Inpaint, Pillow, ImageMagick, Stub

#### Audio Production

- **audio_synthesis**: Suno AI, ElevenLabs, Bark, Stub
- **audio_editing**: librosa, pydub, FFmpeg, Stub
- **text_to_speech**: ElevenLabs, Azure TTS, Google TTS, pyttsx3, Stub

#### Information Retrieval

- **web_search**: Tavily, Perplexity, Serper, Google Custom Search, Stub
- **document_retrieval**: ArXiv, Semantic Scholar, pypdf2, Stub

#### Localization

- **machine_translation**: DeepL, Google Translate, Azure Translator, googletrans, Stub

#### Document Production

- **format_conversion**: Pandoc, unoconv, pypandoc, Stub
- **pdf_generation**: WeasyPrint, ReportLab, Pandoc PDF, Stub
- **epub_generation**: Calibre, Pandoc EPUB, ebooklib, Stub

### Knowledge Capabilities (3 total)

Always-available capabilities provided through the knowledge base:

- **consult_specs**: Access specification documents
- **consult_project_docs**: Access project documentation
- **search_artifacts**: Search previously created artifacts

## Internal Tools (13 total)

Runtime-only operations **NEVER exposed to agents**:

### Protocol Operations (2)

- `send_protocol_envelope`: SendProtocolEnvelope
- `send_protocol_message`: SendProtocolMessage

### State Operations (4)

- `read_hot_sot`: ReadHotSOT
- `write_hot_sot`: WriteHotSOT
- `read_cold_sot`: ReadColdSOT
- `write_cold_sot`: WriteColdSOT

### Validation Operations (2)

- `validate_artifact`: ValidateArtifact
- `evaluate_quality_bar`: EvaluateQualityBar

### Orchestration Operations (5)

- `create_snapshot`: CreateSnapshot
- `update_tu`: UpdateTU
- `wake_role`: WakeRole
- `sleep_role`: SleepRole
- `trigger_gatecheck`: TriggerGatecheck

## Provider Selection Strategy

Each capability has multiple providers ranked by priority:

1. **Priority Ordering**: Providers are numbered 1 (highest) to 999 (stub fallback)
2. **Fallback Strategy**: If a provider fails, try next in priority order
3. **Availability Checks**: Each provider defines how to verify it's available:
   - `api_key`: Check environment variable + test endpoint
   - `command_available`: Check if command exists in PATH
   - `python_package`: Check if Python package is importable
   - `always_available`: Stub providers (always work)

## Provider Types

- **api_service**: External API (DALLE-3, Elevenlabs, Tavily, etc.)
- **local_tool**: Installed commands or libraries (Pandoc, FFmpeg, librosa, etc.)
- **stub**: Placeholder for testing/development (returns mock results)

## Runtime Integration

The runtime uses this configuration to:

1. **Initialize**: Check external capability availability at startup
2. **Select**: Choose provider based on priority and availability
3. **Execute**: Call the appropriate provider implementation
4. **Fallback**: Try next provider if current one fails
5. **Log**: Track capability usage, provider selection, and fallbacks
6. **Validate**: Ensure internal tools are never exposed to agents

## Configuration Example

Here's how a capability with multiple providers is configured:

```yaml
audio_synthesis:
  description: |
    Synthesize audio from descriptions. Maps to AI audio generation services.

  capability_id: audio_synthesis
  category: audio_production

  providers:
    - id: suno_api
      type: api_service
      tool_class: questfoundry.runtime.tools.media_tools.GenerateAudio
      provider_name: "Suno AI Audio Generation"
      availability_check:
        type: api_key
        env_var: SUNO_API_KEY
        endpoint: https://api.suno.ai/v1/generate
      config:
        model: chirp-v3
        duration: auto
        quality: high
        timeout_seconds: 180
        max_retries: 2
      fallback_strategy: next_provider
      priority: 1  # Try first

    - id: elevenlabs_api
      type: api_service
      tool_class: questfoundry.runtime.tools.media_tools.GenerateAudio
      provider_name: "ElevenLabs Audio Synthesis"
      availability_check:
        type: api_key
        env_var: ELEVENLABS_API_KEY
        endpoint: https://api.elevenlabs.io/v1/sound-generation/generate
      config:
        model: audio-generation
        quality: high
        timeout_seconds: 60
        max_retries: 2
      fallback_strategy: next_provider
      priority: 2  # Try if Suno fails

    - id: bark_local
      type: local_tool
      tool_class: questfoundry.runtime.tools.media_tools.GenerateAudio
      provider_name: "Bark (local text-to-speech synthesis)"
      availability_check:
        type: python_package
        package_name: bark
      config:
        model: bark-small
        sample_rate: 24000
        use_gpu: false
      fallback_strategy: next_provider
      priority: 3  # Try if Elevenlabs fails

    - id: stub_audio
      type: stub
      tool_class: questfoundry.runtime.tools.media_tools.GenerateAudio
      provider_name: "Stub (test/development)"
      availability_check:
        type: always_available
      config:
        duration_seconds: 10
        log_warnings: true
      fallback_strategy: none  # Terminal fallback
      priority: 999
```

## Adding a New Provider

To add a new provider for an existing capability:

1. Add a new entry to the `providers` list for that capability
2. Set appropriate `priority` (lower = higher priority)
3. Define `availability_check` method
4. Provide provider-specific `config`
5. Specify `fallback_strategy`
6. Reference the corresponding `tool_class` from `questfoundry.runtime.tools`

Example:

```yaml
- id: new_provider_id
  type: api_service  # or local_tool or stub
  tool_class: questfoundry.runtime.tools.media_tools.GenerateImage
  provider_name: "Descriptive Provider Name"
  availability_check:
    type: api_key
    env_var: NEW_PROVIDER_API_KEY
    endpoint: https://api.example.com/v1/generate
  config:
    model: default-model
    timeout_seconds: 60
    max_retries: 2
  fallback_strategy: next_provider
  priority: 3  # Adjust based on desired priority
```

## Security: Internal Tool Protection

Internal tools are protected through:

1. **Strict Marking**: All internal tools clearly marked with comments
2. **Separate Section**: Internal tools in dedicated section, separate from external capabilities
3. **Enforcement Flag**: `strict_enforcement: true` in tool exposure rules
4. **Audit Logging**: `audit_exposure_attempts: true` logs any attempted access
5. **Override Prevention**: `override_allowed: false` prevents agent manipulation

Internal tools are **NEVER**:

- Listed in role tool definitions
- Exposed through the agent tool interface
- Callable by agents (even with explicit requests)
- Included in tool schema generation

## Runtime Configuration Settings

### Initialization

```yaml
runtime_integration:
  initialization:
    check_external_capabilities: true
    check_required_first: true
    log_unavailable_capabilities: true
    fail_on_missing_required: true
    fail_on_all_providers_unavailable: true
```

### Provider Selection

```yaml
  provider_selection:
    strategy: priority_ordered
    fallback_on_failure: true
    log_provider_switches: true
    max_provider_fallbacks: 3
```

### Logging

```yaml
  logging:
    log_capability_usage: true
    log_provider_selection: true
    log_fallback_events: true
    log_api_errors: true
    log_level: info
```

## Related Files

- `spec/05-definitions/capabilities.yaml`: Abstract capability definitions
- `lib/runtime/src/questfoundry/runtime/tools/`: Tool implementations
- `lib/runtime/src/questfoundry/runtime/core/`: Runtime execution engine

## Version

Version 1.0.0 (Created 2025-11-27)

## See Also

- Architecture documentation for runtime design
- Protocol specification for internal operations
- Tool implementation guide for extending capabilities
