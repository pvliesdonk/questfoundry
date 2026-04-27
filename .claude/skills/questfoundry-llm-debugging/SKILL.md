---
name: questfoundry-llm-debugging
description: Use when a QuestFoundry pipeline stage produces wrong, empty, or unparseable LLM output, validation fails after retries, YAML parse errors appear, structured-output fields come back empty/missing, or a project run needs post-mortem inspection (looking under `projects/<dir>/logs/`).
---

# QuestFoundry LLM Output Debugging

## Overview

When a stage's structured output is wrong, the pipeline can hide the cause behind several layers: validation, repair-loop retries, JSONL log files. This skill is the lookup map for "where do I look first, what do I look at, what is the fix."

**Core rule:** the source of truth for what the model actually saw and emitted is `projects/<dir>/logs/llm_calls.jsonl`. Inspect that before theorising.

## Where to look first

For any project run, the failure signal lives in:

| File | Purpose |
|---|---|
| `projects/<dir>/logs/debug.jsonl` | structured app logs — primary signal for pipeline failures |
| `projects/<dir>/logs/llm_calls.jsonl` | full prompt + LLM trace per call — use to see what the model actually saw and emitted |
| `projects/<dir>/graph.db` | primary output artifact (SQLite, current story state) |
| `projects/<dir>/snapshots/` | pre-stage graph checkpoints for rollback / diagnosis |
| `projects/<dir>/exports/` | derived outputs from `graph.db` — context only |

Do not start from `exports/`. Always start from `debug.jsonl` (what failed) → `llm_calls.jsonl` (why) → `graph.db` (what state we ended up in).

## Enable LLM logging

```bash
uv run qf --log -vvv dream --project myproject "prompt"
```

Creates:

- `{project}/logs/llm_calls.jsonl` — full request/response per LLM call
- `{project}/logs/debug.jsonl` — structured app logs

`-vvv` is the right verbosity for active debugging; lower levels suppress per-call detail.

## Analyze the response

```bash
# Pretty-print the last LLM response (JSONL = one JSON object per line)
python3 -c "
import json
with open('myproject/logs/llm_calls.jsonl') as f:
    lines = f.readlines()
    d = json.loads(lines[-1])  # last call
    print(d['content'])
"

# Debug YAML parsing
python3 -c "
import json, yaml
with open('myproject/logs/llm_calls.jsonl') as f:
    d = json.loads(f.readlines()[-1])
parsed = yaml.safe_load(d['content'])
print(json.dumps(parsed, indent=2))
"
```

Read the `messages` array, not just the response. If you see Python `repr` leakage in any user message (square brackets `['x']`, angle brackets `<Enum.X: 1>`, dict braces), the prompt context formatting is broken — see the `@prompt-engineer` subagent's Rule 4.

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Empty field in parsed YAML | YAML block extraction stops early | Check `_extract_yaml_block` handles multi-line content |
| YAML parse error | Model includes prose around YAML | Improve fence detection or extraction |
| Missing fields | Model omits optional fields | Make schema fields optional with defaults |
| Phantom IDs in output | Prompt did not inject a closed Valid IDs list | Apply Valid ID Injection (see below) |
| Output collapses to "default" answer | Bare ID listing instead of enriched context | Apply Context Enrichment — see `@prompt-engineer` Rule 3 |
| Output contains `[…]` or `<EnumClass…>` | Python `repr` leaked into prompt | See `@prompt-engineer` Rule 4 |
| Repair loop converges to wrong stage | `field_path` prefix routes feedback to wrong prompt | Change feedback `field_path` to match the producing prompt's collected key (cf. #1243/#1244, #1246/#1247) |

## Prompt patterns reference

When debugging output that is *parseable but semantically wrong*, the fix usually lives in one of these patterns:

- **Sandwich pattern** — repeat critical instructions at start AND end of the prompt. Small models lose constraints buried in the middle.
- **Validate → Feedback → Repair loop** — for structured output, validate and ask the model to fix specific errors. Generic "try again" loops don't converge.
- **Discuss → Freeze → Serialize** — the project's three-phase pattern. Separate exploratory dialogue (with tools) from structured output generation. Freeze the discussion before serializing.
- **Model size considerations** — small models (≤8B) need simpler prompts, fewer tools, shorter context. **Default reaction to bad output is to suspect the prompt, not the model.** See `@prompt-engineer` Rule 5.

For prompt rewrites, dispatch the `@prompt-engineer` subagent rather than rewriting inline — it owns the audit dimensions and severity rubric.

## Schema design for LLM output

Since artifacts will be interpreted by other LLMs (not programmatic code), prefer:

- **Strings with examples** over strict enums (e.g., `audience: str` not `Literal["adult",...]`)
- **Inline examples in prompts** to guide format: `audience: <e.g., adult, young adult, mature>`
- **min_length=1 constraints** to catch empty values while accepting variations

This allows LLM-generated variations like "adults" or "extensive" to pass validation while still providing guidance through examples.

## Valid ID Injection

When serializing structured output that references IDs from earlier stages, always provide an explicit `### Valid IDs` section listing every ID the model is allowed to use. Phantom-ID prevention. Implementation lives in `graph/context.py::format_valid_ids_context()` / `format_path_ids_context()`, injected from `agents/serialize.py`. See `@prompt-engineer` Rule 1 for the full rule.

## What this skill is NOT

- It is not a replacement for the `@prompt-engineer` subagent. For prompt rewrites or audits, dispatch that subagent.
- It is not a stage-implementation guide. For adding a new stage, see `questfoundry-stage-implementation`.
- It does not cover graph-DB integrity bugs. For mutations / DAG / Y-shape issues, read the ontology directly and the relevant procedure doc.
