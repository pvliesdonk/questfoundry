# Flow: Binding Run — Assemble Player-Safe Export View

> **Status:** Normative — defines the message sequence for exporting a view from a Cold snapshot.

---

## Overview

The **Binding Run** flow assembles a player-safe export bundle from a specific Cold snapshot. This
flow is **read-only** on Cold—no new canon is created. The Book Binder (BB) exports manuscript,
codex, and optional surfaces (art, audio, translations) in multiple formats (Markdown, HTML, EPUB,
PDF).

### Purpose

- Produce a stamped export bundle from Cold for:
  - Player-Narrator (PN) Narration Dry-Run
  - Milestone releases
  - External review or print builds

### Constraints

- **Cold-only**: All content MUST come from a stamped Cold snapshot
- **Player-safe**: Bundle MUST be suitable for PN ingestion (spoiler-free)
- **Traceability**: TU linkage required; snapshot ID stamped in metadata

---

## Roles

- **Showrunner (SR)** — Selects snapshot, chooses export options, approves distribution
- **Book Binder (BB)** — Assembles bundle from Cold, ensures navigation and accessibility
- **Gatekeeper (GK)** — Spot-checks Presentation Safety, Integrity, Style on built bundle
- **Player-Narrator (PN)** — Receives bundle for Narration Dry-Run (consumer, not participant in
  this flow)

---

## Message Sequence

```
1. SR → BB: view.export.request
   - Context: cold, tu, snapshot
   - Payload: export options (formats, inclusions)

2. BB → SR: ack
   - Acknowledges export request

3. BB assembles bundle from Cold snapshot
   - Compile manuscript sections with navigation
   - Include codex entries (player-safe)
   - Add optional surfaces (art, audio, translations) per options
   - Run accessibility pass
   - Stamp metadata (snapshot ID, options, View Log)

4. BB → SR: view.export.result
   - Context: cold, tu, snapshot
   - Payload: view_log with manifest, coverage, options used
   - Safety: player_safe=true, spoilers=forbidden

5. SR → GK: gate.report.submit
   - Context: cold, tu, snapshot
   - Payload: gatecheck_report (spot-check request)
   - Target: Presentation Safety, Integrity, Style

6. GK performs spot-check
   - Verify no spoilers in visible surfaces
   - Check link integrity across formats
   - Validate style consistency

7. GK → SR: gate.decision
   - Decision: pass | conditional_pass | block
   - If pass: bundle approved for distribution
   - If conditional: remediation notes included

8. SR → PN: view.export.result (handoff)
   - Context: cold, tu, snapshot
   - Payload: view_log pointing to export bundle
   - Safety: player_safe=true, spoilers=forbidden
   - Note: PN receives same snapshot used by BB
```

---

## Envelope Requirements

### 1. view.export.request (SR → BB)

```json
{
  "intent": "view.export.request",
  "sender": { "role": "SR" },
  "receiver": { "role": "BB" },
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": false,
    "spoilers": "allowed"
  },
  "payload": {
    "type": "view_log",
    "data": {
      "view_name": "milestone-chapter-3-export",
      "snapshot": "Cold @ 2025-10-28",
      "export_options": {
        "formats": ["markdown", "html", "epub", "pdf"],
        "include_art_plans": false,
        "include_art_renders": false,
        "include_audio_plans": false,
        "include_audio_assets": false,
        "include_translations": false,
        "print_friendly_layout": true,
        "include_pn_script": false
      }
    }
  }
}
```

**Required envelope fields:**

- `context.hot_cold = "cold"` (source is Cold)
- `context.tu` (trace unit for this export)
- `context.snapshot` (specific Cold snapshot)
- `safety.player_safe = false` (request is internal, not player-facing yet)

---

### 2. view.export.result (BB → SR, BB → PN)

```json
{
  "intent": "view.export.result",
  "sender": { "role": "BB" },
  "receiver": { "role": "SR" },
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Binding Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "forbidden"
  },
  "payload": {
    "type": "view_log",
    "data": {
      "view_name": "milestone-chapter-3-export",
      "snapshot": "Cold @ 2025-10-28",
      "export_options": { "..." },
      "manifest": [
        "manuscript/section-01.md",
        "manuscript/section-02.md",
        "codex/factions.md"
      ],
      "formats_generated": ["markdown", "html", "epub", "pdf"],
      "coverage": {
        "manuscript_sections": 12,
        "codex_entries": 24,
        "art_plans": 0,
        "translations_languages": []
      }
    }
  }
}
```

**Required envelope fields:**

- `context.hot_cold = "cold"`
- `context.snapshot` (same as request)
- `safety.player_safe = true` (bundle is player-safe)
- `safety.spoilers = "forbidden"` (no spoilers in surfaces)

**PN handoff constraint:**

- When `receiver.role = "PN"`, envelope MUST enforce:
  - `context.hot_cold = "cold"`
  - `safety.player_safe = true`
  - `safety.spoilers = "forbidden"`
  - `context.snapshot` MUST be present

---

## Conformance Validation

### MUST Requirements

1. **Cold-only source:**
   - `view.export.request` MUST have `context.hot_cold = "cold"`
   - `context.snapshot` MUST be present and valid format `"Cold @ YYYY-MM-DD"`

2. **Player-safety:**
   - `view.export.result` payload MUST validate against `view_log.schema.json`
   - Result envelope to PN MUST have `safety.player_safe = true`

3. **Traceability:**
   - All messages MUST include `context.tu`
   - `refs` array SHOULD include upstream TU/hook IDs

4. **PN isolation:**
   - Messages to PN (receiver.role="PN") MUST be Cold + player_safe=true + spoilers=forbidden

### SHOULD Requirements

1. Export options SHOULD be explicit in request payload
2. Result manifest SHOULD list all included files
3. Coverage statistics SHOULD be present in result

---

## Example Validation

To validate example envelopes:

```bash
# Extract payload.data to temp file
jq '.payload.data' view.export.request.json > /tmp/payload.json

# Validate against Layer 3 schema
uv run --directory tools qfspec-check-instance view_log /tmp/payload.json
```

---

## Cross-References

- **North Star:** `00-north-star/LOOPS/binding_run.md`
- **Schemas:** `03-schemas/view_log.schema.json`, `03-schemas/gatecheck_report.schema.json`
- **Intents:** `04-protocol/INTENTS.md` (view.export.request, view.export.result)
- **Lifecycles:** `04-protocol/LIFECYCLES/view.md`
- **PN Principles:** `00-north-star/PN_PRINCIPLES.md`
- **Related Flow:** `04-protocol/FLOWS/narration_dry_run.md`

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30
