---
procedure_id: artifact_validation
description: Complete workflow for validating JSON artifacts against schemas
version: 2.0.0
references_schemas:
  - SCHEMA_INDEX.json
references_roles:
  - all
tags:
  - validation
  - quality
  - schema
---

# Artifact Validation Procedure

## Overview

All JSON artifacts MUST be validated against canonical schemas before emission. This is a hard gate with no exceptions.

## Prerequisites

- Access to `SCHEMA_INDEX.json`
- JSON Schema validator (jsonschema, ajv, etc.)
- Target artifact type identified

## Step 1: Discover Schema

Locate the schema in `SCHEMA_INDEX.json` using the artifact type key.

**Input:** Artifact type (e.g., `"hook_card"`, `"canon_pack"`)

**Action:** Read `SCHEMA_INDEX.json` and find the entry for your artifact type.

**Output:** Schema metadata containing:

- `$id`: Canonical schema URL
- `path`: Relative path to schema file
- `draft`: JSON Schema draft version
- `sha256`: Integrity checksum
- `roles`: Which roles produce this artifact
- `intent`: Which protocol intents use this schema

## Step 2: Preflight Protocol

Echo back schema understanding before producing artifact.

**Action:** Output the following:

1. **Schema metadata:**

   ```json
   {
     "$id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
     "draft": "2020-12",
     "path": "03-schemas/hook_card.schema.json",
     "sha256": "a1b2c3d4e5f6..."
   }
   ```

2. **Minimal valid instance:** Show you understand the schema structure
3. **Invalid example:** Show one example that would fail validation with explanation

**Purpose:** Confirms you have correct schema and understand its requirements.

## Step 3: Verify Schema Integrity

Check that the schema file hasn't been modified.

**Action:** Compute SHA-256 hash of schema file and compare to index.

**If hash mismatch:**

```
ERROR: Schema integrity check failed for hook_card.schema.json
Expected SHA-256: a1b2c3d4e5f6...
Actual SHA-256:   deadbeef...
REFUSING TO USE COMPROMISED SCHEMA.
```

**STOP immediately** and report to Showrunner.

## Step 4: Produce Artifact

Create the artifact with required `$schema` field.

**Action:** Generate artifact JSON with:

- `"$schema"` field at top level pointing to schema's `$id` URL
- All required fields per schema
- Proper data types and structure

**Example:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
  "hook_id": "discovery_001",
  "content": "A mysterious locked door in the old library...",
  "tags": ["mystery", "location"],
  "source": "tu-2025-11-06-ss01"
}
```

## Step 5: Validate Against Schema

Run JSON Schema validation on the produced artifact.

**Action:** Use validator to check artifact against schema.

**Validation inputs:**

- Artifact JSON
- Schema from canonical source
- JSON Schema draft version from metadata

**Validation outputs:**

- `valid`: boolean (true/false)
- `errors`: array of validation errors (if any)

## Step 6: Generate Validation Report

Create validation report documenting the results.

**Action:** Produce `validation_report.json` with structure:

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

**If validation failed:**

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

## Step 7: Decision Point

Based on validation result, either emit artifact or stop.

### If Validation Passed (`valid: true`)

**Actions:**

1. Emit artifact file (e.g., `out/hook_card.json`)
2. Emit validation report with `"valid": true`
3. Proceed to next workflow step
4. Include validation report in handoff to next role

**Handoff requirements:**

- Both artifact and validation report must be provided
- Next role should verify validation report before processing

### If Validation Failed (`valid: false`)

**Actions:**

1. **DO NOT emit artifact** - failed artifacts are never delivered
2. Emit validation report with `"valid": false` and error details
3. **STOP workflow immediately** - hard gate, no exceptions
4. Report to user/Showrunner: "Validation failed. See validation_report.json for errors."

**Do not:**

- Attempt to "fix" the artifact and re-validate without guidance
- Proceed with downstream work
- Emit the artifact anyway with a warning

## Loop Integration

In multi-role loops, validation occurs at handoff points.

**Producer role responsibilities:**

1. Validate artifact before handoff
2. Provide both artifact and validation report
3. If validation fails, notify Showrunner immediately

**Consumer role responsibilities:**

1. Verify validation report exists
2. Check `"valid": true` before processing artifact
3. If no validation report or `"valid": false`, refuse to proceed

**Showrunner verification:**
Before allowing role-to-role handoff:

- Artifact file exists with `"$schema"` field
- `validation_report.json` exists
- Report shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP loop and escalate to human.

## Troubleshooting

**Cannot access schema:**

- STOP and report: "Cannot access schema at [URL]. Validation impossible. REFUSING TO PROCEED."
- Check network connectivity or bundled schema availability

**Schema ambiguous or multiple versions:**

- Use `$id` URL from `SCHEMA_INDEX.json` as single source of truth
- Do not use schemas from untrusted sources

**Artifact believed correct but fails validation:**

- Validation failure is authoritative
- DO NOT emit artifact
- Report error and ask for guidance on schema interpretation

**Validation is slow/resource-intensive:**

- Validation is mandatory regardless of performance
- Budget time for validation in workflow planning

## Summary Checklist

- [ ] Locate schema in `SCHEMA_INDEX.json`
- [ ] Preflight: echo metadata + examples
- [ ] Verify schema integrity (SHA-256)
- [ ] Produce artifact with `"$schema"` field
- [ ] Validate against canonical schema
- [ ] Generate validation report
- [ ] If valid: emit both files, proceed
- [ ] If invalid: DO NOT emit artifact, STOP workflow

**This procedure is mandatory for all roles and all artifacts. No exceptions.**
