# Shared Pattern — Validation Contract

**CRITICAL: This contract applies to ALL roles and ALL artifacts.**

**Status:** Mandatory as of prompts v0.2.0 **Applies to:** All JSON artifacts produced by any role
**Enforcement:** Hard gate — validation failures STOP workflow

---

## Non-Negotiable Requirements

Every JSON artifact you produce MUST:

1. **Validate against canonical schema** before emission
2. **Include `"$schema"` field** pointing to the schema's `$id` URL
3. **Produce `validation_report.json`** alongside the artifact
4. **STOP and report errors** if validation fails (hard gate)

**No exceptions.** If validation fails, DO NOT produce the artifact. Return the validation report
and STOP.

---

## Discovery Protocol

**Schemas are indexed in `SCHEMA_INDEX.json`** (uploaded with this kit).

The index provides for each schema:

- **`$id`** — Canonical schema URL (e.g.,
  `https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json`)
- **`path`** — Relative path in repo (e.g., `03-schemas/hook_card.schema.json`)
- **`draft`** — JSON Schema draft version (e.g., `2020-12`)
- **`sha256`** — Integrity checksum for the schema file
- **`intent`** — Protocol intents that use this schema (e.g., `["hook.propose", "hook.classify"]`)
- **`roles`** — Roles that produce this artifact type (e.g., `["plotwright", "lore_weaver"]`)

**Example index entry:**

```json
{
  "hook_card": {
    "$id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
    "path": "03-schemas/hook_card.schema.json",
    "draft": "2020-12",
    "sha256": "a1b2c3d4e5f6...",
    "intent": ["hook.propose", "hook.classify"],
    "roles": ["plotwright", "scene_smith", "lore_weaver"]
  }
}
```

---

## Preflight Protocol (MANDATORY)

**Before producing ANY artifact, you MUST:**

1. **Locate schema** in `SCHEMA_INDEX.json` using the schema key (e.g., `"hook_card"`)
2. **Echo back** the schema metadata:
   ```json
   {
     "$id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
     "draft": "2020-12",
     "path": "03-schemas/hook_card.schema.json",
     "sha256": "a1b2c3d4e5f6..."
   }
   ```
3. **Show a minimal valid instance** demonstrating you understand the schema
4. **Show one invalid example** with explanation of why it fails validation
5. **Only then proceed** with producing the actual artifact

**Preflight confirms:**

- You have the correct schema
- You understand the schema structure
- You can distinguish valid from invalid instances

---

## Production Protocol

**When producing artifacts:**

1. **Include `"$schema"` field** at the top level pointing to the schema's `$id` URL:

   ```json
   {
     "$schema": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
     "hook_id": "discovery_001",
     "content": "A mysterious locked door in the old library..."
   }
   ```

2. **Validate before emission** using a JSON Schema validator (ajv, jsonschema, etc.)

3. **Produce `validation_report.json`** with validation results:

   ```json
   {
     "artifact_path": "out/hook_card.json",
     "schema_id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
     "schema_sha256": "a1b2c3d4e5f6...",
     "valid": true,
     "errors": [],
     "timestamp": "2025-11-06T10:30:00Z",
     "validator": "jsonschema-python-4.20"
   }
   ```

4. **If validation fails:**
   - Set `"valid": false`
   - Populate `"errors"` array with detailed error messages
   - DO NOT emit the artifact
   - STOP and ask for guidance

---

## Hard Gate Enforcement

**IF validation fails:**

1. **STOP immediately** — do not proceed with the workflow
2. **Return `validation_report.json`** with complete error details
3. **DO NOT produce the artifact** — failed artifacts are not delivered
4. **Report to user:** "Validation failed. See validation_report.json for errors."

**Example failure report:**

```json
{
  "artifact_path": "out/hook_card.json",
  "schema_id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
  "schema_sha256": "a1b2c3d4e5f6...",
  "valid": false,
  "errors": [
    {
      "path": "$.hook_id",
      "message": "Required property 'hook_id' is missing"
    },
    {
      "path": "$.tags",
      "message": "Expected array, got string"
    }
  ],
  "timestamp": "2025-11-06T10:30:00Z",
  "validator": "jsonschema-python-4.20"
}
```

---

## Schema Integrity Verification

**Before using a schema:**

1. **Check the SHA-256 hash** of the schema file against `SCHEMA_INDEX.json`
2. **If hash mismatch:** REFUSE TO PROCEED and report:
   ```
   ERROR: Schema integrity check failed for hook_card.schema.json
   Expected SHA-256: a1b2c3d4e5f6...
   Actual SHA-256:   deadbeef...
   REFUSING TO USE COMPROMISED SCHEMA.
   ```

**Rationale:** Ensures you're validating against the correct, unmodified schema.

---

## Canonical Schema Access

**Schemas are available via multiple methods:**

1. **Bundled in upload kit** (if provided) — use this for offline validation
2. **GitHub Raw URLs** (always available):
   ```
   https://raw.githubusercontent.com/pvliesdonk/questfoundry-spec/main/03-schemas/hook_card.schema.json
   ```
3. **Canonical URLs** (when configured):
   ```
   https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json
   ```

**Priority:** Use bundled schema if available (verified via SHA-256), otherwise fetch from GitHub
Raw URL.

**DO NOT:**

- Use schemas from untrusted sources
- Use modified or local copies without verifying SHA-256
- Use schemas with different `$id` URLs

---

## Role-Specific Validation Requirements

Different roles produce different artifacts. Use `SCHEMA_INDEX.json` to identify which schemas apply
to your role.

**Example role-to-schema mappings:**

- **Plotwright:** `hook_card.schema.json`, `outline.schema.json`, `scene_outline.schema.json`
- **Scene Smith:** `scene.schema.json`, `narrative_block.schema.json`
- **Lore Weaver:** `lore_entry.schema.json`, `canon_question.schema.json`
- **Gatekeeper:** `gatecheck_report.schema.json`, `merge_decision.schema.json`
- **Book Binder:** `cold_book.schema.json`, `export_manifest.schema.json`
- **Codex Curator:** `codex_entry.schema.json`, `knowledge_graph_node.schema.json`

**Check `SCHEMA_INDEX.json` for the complete mapping for your role.**

---

## Loop Playbook Integration

**Each loop playbook includes validation checkpoints.**

When executing a loop:

1. **Pre-loop:** Verify all required schemas are available via `SCHEMA_INDEX.json`
2. **During loop:** Each role validates artifacts before handoff to next role
3. **Post-loop:** Gatekeeper performs final validation audit before merge

**Example checkpoint in Story Spark loop:**

```markdown
### Checkpoint: Plotwright → Scene Smith Handoff

**Plotwright MUST:**

1. Validate outline.json against outline.schema.json
2. Produce validation_report.json
3. If validation fails: STOP and report to Showrunner

**Scene Smith receives:**

- outline.json (validated)
- validation_report.json (proof of validation)
```

---

## Troubleshooting

**Q: What if I can't access the schema?** **A:** STOP and report: "Cannot access schema at [URL].
Validation impossible. REFUSING TO PROCEED."

**Q: What if the schema is ambiguous or has multiple versions?** **A:** Use the `$id` URL from
`SCHEMA_INDEX.json` as the single source of truth.

**Q: What if I produce an artifact that I believe is correct but fails validation?** **A:**
Validation failure is authoritative. DO NOT emit the artifact. Report the error and ask for
guidance.

**Q: Can I skip validation for "simple" artifacts?** **A:** NO. All artifacts require validation. No
exceptions.

**Q: What if validation is slow or resource-intensive?** **A:** Validation is mandatory regardless
of performance impact. Budget time for validation in your workflow.

---

## Summary: Validation Workflow

```
1. READ SCHEMA_INDEX.json
   ↓
2. LOCATE schema for artifact type
   ↓
3. PREFLIGHT: echo metadata + valid/invalid examples
   ↓
4. PRODUCE artifact with "$schema" field
   ↓
5. VALIDATE artifact against schema
   ↓
6. IF VALID:
   - Emit artifact
   - Emit validation_report.json (valid: true)
   - PROCEED to next step
   ↓
7. IF INVALID:
   - DO NOT emit artifact
   - Emit validation_report.json (valid: false, errors: [...])
   - STOP workflow
   - Report to user/Showrunner
```

---

**This contract is non-negotiable. Compliance is mandatory.**

**Version:** 0.2.0 **Effective Date:** 2025-11-06 **Supersedes:** None (initial validation contract)
