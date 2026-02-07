# QuestFoundry Model Testing Report

> **Date**: February 2026
> **Pipeline Version**: 5.0
> **Test Prompt**: "Surprise me" (non-interactive mode)

## Executive Summary

Tested 14 model configurations across Ollama (local) and API providers (OpenAI, Google). **qwen3:4b** emerged as the best choice for iteration - fast, free, and reliable. For production quality, **gemini-2.5-pro** offers the best balance of cost, speed, and output quality.

### Quick Reference

| Use Case | Recommended Model | Time | Cost | Notes |
|----------|-------------------|------|------|-------|
| Iteration/Testing | qwen3:4b | 11 min | Free | Fast, reliable, good prose |
| Production (Budget) | gemini-2.5-pro | 93 min | ~$3.56 | Clean run, excellent prose |
| Production (Premium) | gpt-5 | 425 min | ~$20 | Most content, but slow |
| Avoid | gpt-5-nano, gemini-2.0-flash | - | - | Can't handle structured output |

---

## Completed Runs (Reached FILL)

### qwen3:4b (Ollama)

| Metric | Value |
|--------|-------|
| Time | 10.8 min |
| Cost | Free |
| Tokens | 806K (686K in / 120K out) |
| LLM Calls | 270 |
| Entities | 16 |
| Dilemmas | 5 |
| Passages | 63 |
| Genre | Literary noir |

**Warnings**: 26 total
- 6 deterministic label issues
- 5 hard transitions
- 4 incompatible intersections
- 3 semantic validation retries

**Prose Sample**:
> Dust motes dance in the single shaft of light that cuts through the Archive's high, warped ceiling, trembling like insects caught in glass. Elara lifts her fingers—cold, thin, the skin split at the knuckle—and brushes the brass key.

**Prose Quality**: Atmospheric, literary. Good sensory detail despite small model size.

**Verdict**: Excellent for iteration. Fast, reliable, uses corpus tools effectively.

---

### gemini-2.5-flash (Google)

| Metric | Value |
|--------|-------|
| Time | 85.7 min |
| Cost | ~$0.55 |
| Tokens | 1.46M (732K in / 732K out) |
| LLM Calls | 286 |
| Entities | 20 |
| Dilemmas | 6 |
| Passages | 63 |
| Genre | Folk Horror Mystery |

**Warnings**: 61 total (problematic)
- 18 grow_llm_validation_fail
- 10 incompatible intersections
- 6 batch_item_failed
- 6 phase4e_llm_failed

**Prose Sample**:
> Dust motes dance in the slanted light, each speck a tiny ghost in the still air of the study. Your fingers trace the worn leather binding of what must be Aris Thorne's journal, a faint, metallic tang of old ink and decaying paper rising from its pages.

**Prose Quality**: Competent but less distinctive than other models.

**Verdict**: Not recommended. Many GROW validation failures and batch failures despite completing.

---

### gemini-2.5-pro (Google)

| Metric | Value |
|--------|-------|
| Time | 92.6 min |
| Cost | ~$3.56 |
| Tokens | 1.19M (639K in / 552K out) |
| LLM Calls | 257 |
| Entities | 16 |
| Dilemmas | 6 |
| Passages | 59 |
| Genre | Silkpunk |

**Warnings**: 30 total (mostly benign)
- 12 entity_update_skipped
- 2 incompatible intersections
- 15 API key warnings (configuration, not model issue)

**Prose Sample**:
> The Silk-Clad Ambassador moves through the Imperial Court with a liquid economy, their robes of unbroken silver making no sound against the polished jade floors. They find you not in your official capacity, but in a quiet alcove overlooking the Garden of Calculated Silence.

**Prose Quality**: Sophisticated, distinctive voice. Creative genre choice (Silkpunk).

**Verdict**: Best quality/reliability balance for production. Clean run, excellent prose.

---

### gpt-5-mini (OpenAI)

| Metric | Value |
|--------|-------|
| Time | 196.7 min (3.3 hours) |
| Cost | ~$1.23 |
| Tokens | 1.23M (697K in / 529K out) |
| LLM Calls | 254 |
| Entities | 24 |
| Dilemmas | 7 |
| Passages | 56 |
| Genre | Magical realism |

**Warnings**: 31 total
**GROW Retries**: 205 (concerning)

**Prose Sample**:
> Soap-slick water gathers in the flagstone depressions; needle clicks puncture the damp air in a slow metronome. A woman with knuckled hands lifts a bundle of frayed cloth as if opening an old letter; the brass on her middle finger hums when it catches the light.

**Prose Quality**: Tactile, sensory details. Good atmospheric writing.

**Verdict**: Slow with many GROW retries. Cost not justified given qwen3:4b performance.

---

### gpt-5 (OpenAI)

| Metric | Value |
|--------|-------|
| Time | 425.3 min (7+ hours) |
| Cost | ~$20.18 |
| Tokens | 1.99M (964K in / 1024K out) |
| LLM Calls | 319 |
| Entities | 28 |
| Dilemmas | 7 |
| Passages | 72 |
| Genre | Solarpunk |

**Warnings**: 33 total
- 23 phase8c_empty_details (significant issue)
- 5 hard transitions
- 4 incompatible intersections

**GROW Retries**: 252 (very problematic)

**Prose Sample**:
> Sodium sirens strobe across salt-wet ropes and tarred planks, bleaching the Commons in heartbeat flashes. I set my palm on the brass tide-clock inset in the borrowed name ledger; its needle ticks toward Reed's hour.

**Prose Quality**: Distinctive voice, poetic imagery. Best prose quality overall.

**Verdict**: Overkill. 7 hours and $20 for marginally better output. Massive GROW struggles.

---

## Failed Runs

### gpt-5-nano (OpenAI)

| Metric | Value |
|--------|-------|
| Failed At | SEED |
| Time Wasted | 49.5 min |
| Cost Wasted | ~$0.34 |
| Tokens | 1.58M |

**Failure Mode**: Cannot follow ID format instructions. Outputs `truthful_records` instead of required `archive_truthful_or_distorted__truthful_records`.

**BRAINSTORM Quality**: Actually good - 23 entities, creative "memory-geography braid" genre, "Literary speculative fiction".

**Verdict**: Not viable. Cheap but can't handle structured output requirements. No tool usage observed.

---

### gemini-2.0-flash (Google)

| Metric | Value |
|--------|-------|
| Failed At | BRAINSTORM |
| Time | 1.3 min |
| Cost | ~$0.01 |

**Failure Mode**: Generated duplicate entity `location::wandering_emporium`.

**Verdict**: Not viable. Too capability-stripped for structured generation.

---

### gemma2:9b (Ollama)

| Metric | Value |
|--------|-------|
| Failed At | DREAM (immediately) |

**Failure Mode**: Ollama doesn't support tools for gemma2 models.

**Verdict**: Blocked by Ollama limitation, not a model capability issue.

---

## Partial Runs (Stopped at BRAINSTORM)

### gpt-oss:120b (Ollama - CPU)

| Metric | Value |
|--------|-------|
| Time | 19.8 min (DREAM+BRAINSTORM only) |
| Entities | 23 |
| Dilemmas | 8 |
| Genre | Neo-noir cyber-punk mystery |

**Quality**: Excellent creative variety, unique concepts. Best BRAINSTORM output observed.

**Verdict**: Best creative output but too slow for full pipeline (would take hours for FILL).

---

### hermes3:8b (Ollama)

| Metric | Value |
|--------|-------|
| Time | 2.5 min |
| Entities | 12 |
| Dilemmas | 3 |
| Genre | Mystery/Thriller |

**Verdict**: Minimal output despite "agentic capabilities" reputation. Generic tropes.

---

### llama3.1:8b (Ollama)

| Metric | Value |
|--------|-------|
| Time | 1.0 min |
| Entities | 12 |
| Dilemmas | 3 |
| Genre | Dark Fantasy |

**Issue**: Produces meta-commentary instead of story content. Themes like "The power of cumulative player micro-decisions in shaping the story."

**Verdict**: Not suitable - outputs design philosophy instead of story elements.

---

### qwen3:8b (Ollama)

| Metric | Value |
|--------|-------|
| Time | 2-3 min |
| Entities | 10-19 |
| Dilemmas | 5-8 |
| Genres | Time-loop mystery, surreal psychological IF |

**Verdict**: Reliable, good variety. Slightly better creative output than qwen3:4b.

---

## Warning Analysis

| Model | Total Warnings | GROW Retries | Critical Issues |
|-------|---------------|--------------|-----------------|
| qwen3:4b | 26 | 8 | None |
| gemini-2.5-pro | 30 | 3 | None (clean run) |
| gemini-2.5-flash | 61 | 36 | 18 LLM validation fails, 6 batch fails |
| gpt-5-mini | 31 | 205 | Excessive GROW retries |
| gpt-5 | 33 | 252 | 23 empty_details, massive GROW struggles |

### Key Finding

OpenAI models (gpt-5, gpt-5-mini) have significantly more GROW phase retries than Gemini or local models. This suggests chat-optimization may interfere with structured output generation in complex multi-phase pipelines.

---

## Cost Analysis

| Model | Input Cost | Output Cost | Total | Per-Entity |
|-------|------------|-------------|-------|------------|
| qwen3:4b | Free | Free | Free | Free |
| gemini-2.5-flash | $0.11 | $0.44 | $0.55 | $0.03 |
| gemini-2.5-pro | $0.80 | $2.76 | $3.56 | $0.22 |
| gpt-5-mini | $0.17 | $1.06 | $1.23 | $0.05 |
| gpt-5 | $4.82 | $15.36 | $20.18 | $0.72 |
| gpt-5-nano | $0.04 | $0.30 | $0.34 | N/A (failed) |

---

## Recommendations

### For Iteration and Testing
Use **qwen3:4b** (Ollama local). It's fast (11 min), free, and produces reliable output. The corpus tools compensate for the small model size.

### For Production (Budget-Conscious)
Use **gemini-2.5-pro**. At ~$3.56 per run, it offers excellent prose quality, creative genre choices, and a clean run with minimal warnings.

### For Production (Maximum Quality)
Use **gpt-5** only if budget and time are not constraints. Expect 7+ hours and ~$20 per run. The 252 GROW retries indicate the model struggles with the pipeline.

### Models to Avoid
- **gpt-5-nano**: Can't follow structured ID formats
- **gemini-2.0-flash**: Generates duplicates
- **gpt-5-mini**: Slow with excessive retries, not worth the cost
- **gemini-2.5-flash**: Too many validation failures

---

## Known Issues

### 1. OpenAI Omits Optional Fields (TOOL Strategy)

**Root Cause**: OpenAI uses function calling (TOOL strategy) which interprets `default_factory` fields as optional. The model omits them entirely.

**Evidence**: GPT-5 omits `OverlayProposal.details` in Phase 8c, causing 23 `phase8c_empty_details` warnings. The "200+ retries" originally reported were false positives from SDK idempotency keys.

**Fix**: Remove `default_factory` from required fields OR add explicit "REQUIRED" language in prompts.

### 2. Gemini Flash Ignores Length Constraints

**Root Cause**: Flash ignores `maxLength` schema constraints despite Google's API supporting them. This is a model capability limitation.

**Evidence**:
- `path_theme` has `max_length=200`
- Flash outputs: avg 228 chars, **82% exceed limit**
- Pro outputs: avg 164 chars, **0% exceed**

**Fix**: Add explicit length limits to field descriptions AND prompt text.

### 3. Provider Strategy Summary

| Provider | Strategy | Issue | Recommendation |
|----------|----------|-------|----------------|
| OpenAI | TOOL | Omits `default_factory` fields | Use with all-required schemas only |
| Gemini Flash | JSON_MODE | Ignores `maxLength` | Use for exploration phases only |
| Gemini Pro | JSON_MODE | ✅ Works | Safe for structured output |
| Ollama | JSON_MODE | ✅ Works | Safe for structured output |

### 4. Cheap Tier Limitations

nano/flash tier models lack the instruction-following capability needed for QuestFoundry's structured output requirements.

---

## Appendix: Test Configuration

All tests used:
- Pipeline: DREAM → BRAINSTORM → SEED → GROW → FILL
- Mode: `--no-interactive`
- Prompt: "Surprise me"
- Logging: `--log` enabled

Local models ran on Ollama with OLLAMA_HOST pointing to a GPU server.
API models used standard endpoints with keys from environment.
