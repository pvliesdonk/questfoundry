# Model Comparison: Small Models Outperform GPT-4o on Structured Generation

**Date**: 2026-01-15
**Pipeline Version**: Post-PR #159-164 (provider defaults, prompt quality improvements)
**Stages Tested**: DREAM → BRAINSTORM → SEED (non-interactive mode)

## Executive Summary

Three models were tested on the full DREAM→BRAINSTORM→SEED pipeline using identical prompts. A 4B parameter local model (Qwen3:4b-instruct) produced the highest quality output, while OpenAI's flagship GPT-4o produced the weakest results. This suggests that **instruction-following capability matters more than raw model size** for constrained creative generation tasks.

## Test Configuration

| Run | Model | Architecture | Active Params | Provider |
|-----|-------|--------------|---------------|----------|
| run-1-ni | qwen3:4b-instruct | Dense | 4B | Ollama (local) |
| run-2-ni | gpt-4o | Dense | ~200B (est.) | OpenAI API |
| run-3-ni | gpt-oss-20b | MoE | 3.6B | Ollama (local) |

All runs used the same user prompt (classic murder mystery theme) and non-interactive mode (`-ni` flag).

### Performance Metrics

| Metric | qwen3:4b | gpt-4o | gpt-oss-20b |
|--------|----------|--------|-------------|
| Wall clock time | 6.4 min | 2.5 min | 33.7 min |
| LLM time | 6.6 min | 2.6 min | 34.4 min |
| LLM calls | 33 | 17 | 19 |
| Tokens used | N/A* | 46,099 | N/A* |

*Ollama provider does not report token counts.

**Key Observations:**
- GPT-4o was **fastest** (2.5 min) but produced the **weakest output**
- qwen3:4b took 2.5× longer but produced the best quality
- gpt-oss-20b was very slow (34 min) — MoE routing overhead on local inference
- More LLM calls correlated with better quality (validation/repair loops)

### Time Breakdown by Stage

| Stage | qwen3:4b | gpt-4o | gpt-oss-20b |
|-------|----------|--------|-------------|
| DREAM | 204s (16 calls) | 48s (10 calls) | 1169s (12 calls) |
| BRAINSTORM | 77s (7 calls) | 44s (2 calls) | 365s (2 calls) |
| SEED | 114s (7 calls) | 67s (2 calls) | 527s (2 calls) |

## Results Summary

### Quantitative Metrics

| Metric | qwen3:4b | gpt-4o | gpt-oss-20b |
|--------|----------|--------|-------------|
| BRAINSTORM entities | 22 | 15 | 31 |
| BRAINSTORM tensions | 8 | 5 | 6 |
| SEED threads | 4 | 3 | 3 |
| SEED beats total | 16 | 6 | 12 |
| Beats per thread | 4 | **2** | 4 |
| Artifact size (SEED) | 23KB | 9KB | 15KB |

### Qualitative Assessment

| Dimension | qwen3:4b | gpt-4o | gpt-oss-20b |
|-----------|----------|--------|-------------|
| DREAM specificity | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| Entity richness | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Tension quality | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Beat count compliance | ✓ | ✗ | ✓ |
| Location diversity | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| Consequence specificity | ★★★★★ | ★★☆☆☆ | ★★★★☆ |
| Schema compliance | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **Overall** | **Best** | **Worst** | Good |

## Detailed Analysis

### Run 1: qwen3:4b-instruct — Best Overall

**Strengths:**
- Followed all explicit requirements (4 beats per thread, tension progression)
- Rich, interconnected entity concepts with strong thematic coherence
- Tea rituals, hidden truths, and the garden as liminal space created unified atmosphere
- Proper tension impact progression: advances → reveals → complicates → commits
- Specific, narratively meaningful consequences
- Coherent convergence sketch tying threads together

**Weaknesses:**
- All factions cut (missed opportunity for social dynamics)
- Some location_alternatives repetitive (many default to "the_garden")

**Notable Output:**
```yaml
# Example beat showing proper structure
- beat_id: '2.4'
  summary: 'The diary reveals the incident: a fire that killed a guest. The host was present.'
  threads:
    - library_knowledge
  tension_impacts:
    - tension_id: library_knowledge_salvation_or_corruption
      effect: commits
      note: The diary contains a dangerous truth—corruption—showing the host covered up a fatal incident.
```

### Run 2: gpt-4o — Weakest Performance

**Critical Issues:**
- Only 6 beats total (2 per thread) — below the 2-4 minimum requirement
- Thin BRAINSTORM output (15 entities vs 22-31 for other models)
- Vague consequences ("Influences characters' perceptions")
- Generic DREAM vision ("Classic Murder Mystery")
- Minimal convergence sketch

**Example of Weak Output:**
```yaml
# Consequence lacks specificity
- consequence_id: genuine_collector_consequence
  thread_id: genuine_collector
  description: Reveals deeper truths about Blackwood's past and intentions.
  narrative_effects:
    - Influences characters' perceptions of Lord Blackwood.
```

Compare to qwen3:4b's consequence: "The diary reveals a hidden truth about a past fire, which corrupts the estate's history and causes moral collapse among the guests."

**Hypothesis:** GPT-4o's RLHF tuning optimizes for concise, helpful responses in chat contexts. This backfires on creative generation tasks that require elaboration and explicit length requirements.

### Run 3: gpt-oss-20b — Good with Schema Issues

**Strengths:**
- Best DREAM output with specific narrator choice and motif system
- Proper beat count (4 per thread)
- Excellent location diversity across beats
- Logical, dramatic beat progression

**Schema Compliance Issues:**
- `central_entity_ids: []` on all tensions (empty arrays)
- Several tensions have `explored: ['']` (empty strings)
- Consequence for "vault_of_truth" describes a booby-trap (logical inconsistency)

**Notable:** Despite being a MoE model with only 3.6B active parameters, it produced richer output than GPT-4o. The routing mechanism appears to activate appropriate experts for creative writing tasks.

## Implications for QuestFoundry

### 1. Default Model Selection

Current default is `ollama/qwen3:8b`. Results suggest:
- **qwen3:4b-instruct** is viable for resource-constrained environments
- **gpt-4o** should not be the default for BRAINSTORM/SEED stages
- **gpt-oss-20b** is a strong option for local inference with Apache 2.0 licensing

### 2. Prompt Engineering

The pipeline's constrained prompts with clear schemas allow small models to punch above their weight. The Discuss→Freeze→Serialize pattern forces structure that compensates for reduced model capability.

**Recommendations:**
- Add explicit minimums: "You MUST produce at least 8 beats total"
- Consider model-specific prompt variants (GPT-4o may need "do not abbreviate" language)
- Strengthen the "sandwich pattern" for beat count requirements

### 3. Validation Improvements

Schema validation should catch:
- Beat count < 2 per thread
- Empty `central_entity_ids` arrays on tensions
- Empty strings in `explored` arrays
- Logical inconsistencies between alternative selection and consequence description

### 4. Cost Optimization

**API Cost Analysis (GPT-4o run):**
- 46,099 tokens used across 3 stages
- Estimated cost at GPT-4o pricing (~$5/1M input, $15/1M output): **~$0.35 per run**
- Full pipeline (6 stages) would extrapolate to **~$0.70-1.00 per project**

**Local Inference Cost:**
- qwen3:4b: ~200W × 6.4 min = **0.02 kWh** (~$0.003 at $0.15/kWh)
- gpt-oss-20b: ~200W × 34 min = **0.11 kWh** (~$0.017 at $0.15/kWh)

**Recommendations:**
If qwen3:4b produces the best BRAINSTORM/SEED output locally:
- Reserve API calls (GPT-4o, Claude) for FILL stage where prose fluency matters
- Use local models for structural stages where instruction-following dominates
- Cost savings: **99%+ reduction** in API costs for DREAM→BRAINSTORM→SEED stages

## Validation and Repair Loop Analysis

The pipeline includes a validate→feedback→repair loop for structured output. This data shows how each model interacted with it:

| Metric | qwen3:4b | gpt-4o | gpt-oss-20b |
|--------|----------|--------|-------------|
| Total LLM calls | 33 | 17 | 19 |
| Semantic validation failures | 1 | 0 | 0 |
| Errors in failed validation | 8 | — | — |
| Sections requiring repair | consequences | — | — |

### Key Finding: Validation Doesn't Guarantee Quality

**qwen3:4b** triggered a semantic validation failure in the SEED stage (8 errors in the consequences section) and required one repair iteration. The repair loop fixed the issues, and the final output was the richest of all three runs.

**gpt-4o** passed validation on the first attempt for all stages — but produced the thinnest output. It met the minimum schema requirements without triggering semantic validation, essentially "gaming" the validator by being technically correct but creatively lazy.

**gpt-oss-20b** also passed first-try validation and produced mid-quality output.

### Implications

1. **Schema validation catches structure errors, not quality issues** — a model can produce valid but minimal output
2. **The repair loop improves small model output** — qwen3:4b's final quality exceeded its first-attempt quality
3. **Consider content density validation** — beat count, consequence specificity, etc. should be validated programmatically
4. **More LLM calls ≠ worse** — qwen3:4b's 33 calls (vs GPT-4o's 17) produced better results because it engaged more deeply with the task

## Interactive vs Non-Interactive Mode

Additional runs were performed with interactive mode (human in the loop during discuss phases) to test whether GPT-4o's chat optimization would help when there's a real conversation partner.

### Performance Comparison

| Model | Mode | Wall Clock | LLM Time | Calls | Tokens |
|-------|------|------------|----------|-------|--------|
| qwen3:4b | non-interactive | 6.4 min | 6.6 min | 33 | N/A |
| qwen3:4b | interactive | 9.6 min | 6.9 min | 33 | N/A |
| gpt-4o | non-interactive | 2.5 min | 2.6 min | 17 | 46K |
| gpt-4o | interactive | 6.9 min | 3.4 min | 22 | 69K |

### Quality Comparison

| Model | Mode | Entities | Tensions | Threads | Beats |
|-------|------|----------|----------|---------|-------|
| qwen3:4b | non-interactive | 22 | 8 | 4 | 16 |
| qwen3:4b | interactive | 12 | 6 | 6 | 16 |
| gpt-4o | non-interactive | 15 | 5 | 3 | 6 |
| gpt-4o | interactive | 22 | 8 | 3 | **6** |

### Key Finding: Interactive Mode Helps BRAINSTORM, Not SEED

**GPT-4o with human guidance:**
- ✓ BRAINSTORM improved significantly: 15→22 entities, 5→8 tensions
- ✓ More LLM engagement: 17→22 calls, 46K→69K tokens
- ✗ **SEED unchanged**: Still only 6 beats (2 per thread)
- ✗ Beat count issue persists despite human interaction

**qwen3:4b with human guidance:**
- BRAINSTORM more focused: 22→12 entities (human steering reduced sprawl)
- Beat count maintained: 16 beats in both modes
- More threads: 4→6 threads (human encouraged more storylines)
- Similar LLM time despite human pauses

### Interpretation

GPT-4o's RLHF tuning for helpful chat **does** help in the discuss phases — interactive BRAINSTORM output matches qwen3:4b's quality. However, the SEED serialization issue is **fundamental**: GPT-4o under-produces structured output regardless of prior conversation quality.

This suggests the problem is in the serialize phase specifically, not the discuss phase. GPT-4o's tendency toward concise responses is deeply embedded and persists even when:
1. The discuss phase produced rich material
2. A human explicitly guided the conversation
3. The prompt clearly specifies "2-4 beats per thread"

**Recommendation**: For GPT-4o, consider:
- Even stronger "DO NOT ABBREVIATE" language in serialize prompts
- Explicit minimum beat counts with validation rejection
- Or simply use local models for SEED stage

---

## Appendix A: YAML Excerpts

### A.1 DREAM Stage Comparison

**Run 1 (qwen3:4b)** — Atmospheric and specific:
```yaml
tone:
  - gilded
  - elegant
  - slightly eerie
  - 1940s English country house
  - tea parties
  - subtle tension
  - social intrigue
themes:
  - deduction over action
  - identity and trust
  - social dynamics
  - hidden motives
  - locked room
  - single pivotal clue
```

**Run 2 (gpt-4o)** — Generic:
```yaml
tone:
  - Suspenseful
  - Intriguing
  - Cerebral
themes:
  - Deception and Truth
  - Justice and Morality
  - Class and Society
```

**Run 3 (gpt-oss-20b)** — Rich with motifs:
```yaml
tone:
  - Lightly sardonic wit with dark, atmospheric mystery
  - Opulent manor setting, snappy dialogue, dry humor
themes:
  - Justice vs. Vengeance (red locket motif)
  - Class & Social Expectation (velvet‑rope motif)
  - Deception & Truth (mirror motif)
  - Time & Memory (grandfather‑clock motif)
```

### A.2 SEED Beat Quality Comparison

**Run 1 (qwen3:4b)** — Rich tension impacts and specific consequences:
```yaml
- beat_id: '2.4'
  summary: 'The diary reveals the incident: a fire that killed a guest.
    The host was present.'
  threads:
    - library_knowledge
  tension_impacts:
    - tension_id: library_knowledge_salvation_or_corruption
      effect: commits
      note: The diary contains a dangerous truth—corruption—showing
        the host covered up a fatal incident.
  entities:
    - the_family_diary
    - the_host
  location: the_library
```

**Run 2 (gpt-4o)** — Minimal detail:
```yaml
- beat_id: gc_beat_1
  summary: Lord Blackwood discusses his acquisition with Dr. Langford
    in the mansion's library.
  threads:
    - genuine_collector
  tension_impacts:
    - tension_id: blackwood_motivation
      effect: advances
      note: Provides insight into Blackwood's genuine interest.
  entities:
    - the_host
    - the_scholar
    - the_artifact
  location: the_mansion
```

**Run 3 (gpt-oss-20b)** — Good structure, some schema issues:
```yaml
- beat_id: beat_a2
  summary: During the banquet, a poisoned chalice is discovered, and
    the red locket is found near the victim's hand.
  threads:
    - thread_a
  tension_impacts:
    - tension_id: t1
      effect: advances
      note: introduces locket as key clue
  entities:
    - inspector_hawthorne
    - lord_percival_hawthorne
    - lady_eleanor_hawthorne
    - agnes_wren
    - sir_reginald_blythe
    - poisoned_chalice
    - red_locket
  location: ballroom
```

### A.3 Consequence Quality Comparison

**Run 1 (qwen3:4b)** — Specific narrative effects:
```yaml
- consequence_id: library_knowledge_salvation_to_corruption
  thread_id: library_knowledge
  description: The diary reveals a hidden truth about a past fire,
    which corrupts the estate's history and causes moral collapse
    among the guests.
  narrative_effects:
    - Introduces a dangerous truth that undermines trust
    - Creates emotional conflict and personal risk for characters
    - Shifts the resolution from peace to tragedy
```

**Run 2 (gpt-4o)** — Vague effects:
```yaml
- consequence_id: genuine_collector_consequence
  thread_id: genuine_collector
  description: Reveals deeper truths about Blackwood's past and intentions.
  narrative_effects:
    - Influences characters' perceptions of Lord Blackwood.
```

---

## Reproduction

```bash
# Run 1: qwen3:4b-instruct
uv run qf dream -ni --provider ollama/qwen3:4b-instruct --project run-1-ni "classic murder mystery"
uv run qf brainstorm -ni --provider ollama/qwen3:4b-instruct --project run-1-ni
uv run qf seed -ni --provider ollama/qwen3:4b-instruct --project run-1-ni

# Run 2: gpt-4o
uv run qf dream -ni --provider openai/gpt-4o --project run-2-ni "classic murder mystery"
uv run qf brainstorm -ni --provider openai/gpt-4o --project run-2-ni
uv run qf seed -ni --provider openai/gpt-4o --project run-2-ni

# Run 3: gpt-oss-20b
uv run qf dream -ni --provider ollama/gpt-oss:20b --project run-3-ni "classic murder mystery"
uv run qf brainstorm -ni --provider ollama/gpt-oss:20b --project run-3-ni
uv run qf seed -ni --provider ollama/gpt-oss:20b --project run-3-ni
```

## Conclusions

1. **Instruction-following > raw capability** for structured generation tasks
2. **Small models are viable** when prompts provide sufficient constraint
3. **GPT-4o's chat optimization hurts creative generation** — it under-produces
4. **MoE architectures work well** for this pipeline (gpt-oss-20b with 3.6B active)
5. **Local inference is competitive** with cloud APIs for structural stages

## Future Work

- Test qwen3:8b and gpt-oss-120b to find optimal size/quality tradeoff
- Add programmatic beat count validation to SEED schema
- Develop model-specific prompt variants for known under-producers
- Benchmark FILL stage separately (prose fluency may favor larger models)
- Test Claude models (Haiku, Sonnet) for comparison

---

*Analysis performed on artifacts in `projects/run-{1,2,3}-ni/`*
