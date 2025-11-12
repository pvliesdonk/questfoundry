# Using Layer 5 Prompts To Produce Your First Manuscript — A Practical Guide

**Audience:** Humans using ChatGPT, Claude, or Gemini to commission interactive manuscripts from the
QuestFoundry studio.

**Your role:** You are the **Customer**—the external stakeholder who provides high-level directives
("create a mystery with three acts," "expand this backstory"). The **AI Showrunner** acts as your
product owner, translating your directives into production work and coordinating the 15 internal
studio roles to deliver your manuscript.

## Quick Start

Layer 5 supports **two usage modes**: Standalone (human-led) and **Orchestration (AI-led)** ⭐.

### Quick Start: Orchestration Mode ⭐ **RECOMMENDED**

Best for: **Production workflows with multiple roles**

In this mode, you (the Customer) give directives to the AI Showrunner, which orchestrates the
internal studio roles to execute your project.

1. Choose a chatbot (ChatGPT, Claude, or Gemini)
2. Upload `orchestration-complete.zip` (or Gemini splits: `gemini-orchestration-1` through `-5`)
3. Give your directive: "Load the Story Spark playbook from loops/ and execute it for a 3-scene
   mystery."
4. The AI Showrunner coordinates all roles via loop playbook (single-source-of-truth procedure)
5. Result: Gate-approved manuscript with full traceability

**Why recommended?** 70% context reduction, clear coordination, role interchangeability.

### Quick Start: Standalone Mode

Best for: **Learning, single-role tasks, exploration**

In this mode, you direct the AI Showrunner more manually, giving step-by-step guidance through the
production process.

1. Choose a chatbot (ChatGPT, Claude, or Gemini)
2. Upload `minimal-standalone.zip` (10 files)
3. Direct the bot: "Act as Showrunner and open a TU for my manuscript"
4. Guide the work: "Wake Plotwright, Scene Smith, and Style Lead; iterate outline → scenes → style"
5. Request quality checks: "Pre-gate with Gatekeeper, bind with Book Binder, do a PN dry-run"
6. Complete the work: "Close the TU when satisfied"

## Working with Loop Playbooks

Loop playbooks are the **primary way to execute QuestFoundry workflows**. Each playbook contains
complete procedures for one of the 13 canonical loops.

**Location:** `loops/*.playbook.md`

### Running a Loop (Multi-Role Orchestration)

1. **Showrunner loads playbook**: `loops/lore_deepening.playbook.md`
2. **Playbook specifies procedure**: 9 steps with message sequences
3. **Roles respond to intents**: Lore Weaver drafts canon, Gatekeeper pre-gates, etc.
4. **Showrunner coordinates**: Routes messages per RACI matrix in playbook

**Example workflow:**

```
loops/lore_deepening.playbook.md        # Load procedure
  → Step 1: tu.open (SR → LW)           # Showrunner opens TU
  → Step 2: Frame questions (LW)        # Lore Weaver drafts
  → Step 3: Draft answers (LW)          # Lore Weaver creates canon
  → Step 8: Pre-gate (LW → GK)          # Gatekeeper validates
  → Step 9: Handoff (LW → CC)           # Codex Curator receives summaries
```

### Choosing Format: Full Prompt vs Adapter

**For standalone work:** Upload full role prompt

- Example: "I need Lore Weaver help canonizing backstory"
- Upload: `lore_weaver/system_prompt.md`

**For orchestrated workflow:** Use adapters + loop playbook

- Example: "Run Lore Deepening loop for character hooks"
- Load: `loops/lore_deepening.playbook.md` + `role_adapters/lore_weaver.adapter.md`

## Supported Chatbots And File Uploads

- **ChatGPT**
  - Limit: up to 10 files per upload action (can repeat uploads or use zip)
  - **Orchestration:** Upload `orchestration-complete.zip` (single upload, 36 files)
  - **Standalone:** Upload `minimal-standalone.zip` (10 files), add `optional-standalone.zip` later
- **Claude**
  - No strict limit; handles multiple attachments well
  - **Orchestration:** Upload `orchestration-complete.zip` or individual files (preferred for
    grounding)
  - **Standalone:** Upload `minimal-standalone.zip` + `optional-standalone.zip`
- **Gemini**
  - Limit: 10 files max per zip
  - **Orchestration:** Upload `gemini-orchestration-1` through `-5` in sequence (5 zips, 36 files
    total)
  - **Standalone:** Upload `gemini-minimal-standalone.zip` + `gemini-optional-standalone.zip`

Tip: If your platform drops attachments between threads, reattach your kit at the start of each new
session.

## Upload Kits: Standalone vs. Orchestration

**Choose your mode:**

### Orchestration Mode ⭐ **RECOMMENDED for Production**

**Files:** `orchestration-complete.zip` (36 files) or `gemini-orchestration-1` through `-5` splits

**Contains:**

- 4 shared patterns (`_shared/*.md`)
- 4 showrunner modules (system_prompt, loop_orchestration, manifest_management, protocol_handlers)
- 13 loop playbooks (`loops/*.playbook.md`)
- 15 role adapters (`role_adapters/*.adapter.md`)

**Benefits:**

- **70% context reduction** vs. standalone mode (adapters are 50-100 lines vs. full prompts at
  200-300 lines)
- Single-source-of-truth loop procedures (1 playbook vs 7 role prompts for a loop)
- Clear RACI matrix and role coordination
- Role interchangeability (swap implementations without changing playbooks)

**Best for:** Multi-role workflows, production runs, efficient orchestration

### Standalone Mode (Traditional)

**Minimal:** `minimal-standalone.zip` (10 files)

- 4 shared patterns
- 1 showrunner prompt
- 5 core role prompts (Plotwright, Scene Smith, Style Lead, Gatekeeper, Book Binder)

**Optional Addon:** `optional-standalone.zip` (9 files)

- Player Narrator, Lore Weaver, Codex Curator, Researcher
- Art Director, Illustrator, Audio Director, Audio Producer, Translator

**Full:** `full-standalone.zip` (23 files)

- All shared patterns + all showrunner modules + all 15 role prompts

**Best for:** Learning, single-role tasks, human-led exploration

### Layer 3 Schema Reference (No Upload Required)

**All Cold SoT schemas are available at canonical URLs:**
`https://questfoundry.liesdonk.nl/schemas/`

Gatekeeper and Book Binder prompts reference these schemas by URL for validation. No upload needed.

**Build kits:** `uv run qfspec-build-kits` (output: `dist/upload_kits/`)

See also: `05-prompts/upload_kits/README.md` for complete kit descriptions and platform guidance.

## Boot Sequence (Paste Or Upload, Then Prompt)

1. Upload Minimal Upload Kit (or paste in order): shared patterns → Showrunner → other roles.
2. Send this message to initialize Showrunner as orchestrator:

```
You are the Showrunner. Load and follow the uploaded shared patterns and your system prompt.
Open a new TU for a short manuscript (3–5 scenes). Use tu.open and produce a tu_brief payload
with loop "Story Spark". Target Cold snapshot "Cold @ YYYY-MM-DD" (pick today). After opening,
reply with a gate-safe summary and a plan to wake core roles (Plotwright, Scene Smith, Style Lead).
```

3. Approve the TU plan. Then send:

```
Wake Plotwright, Scene Smith, and Style Lead for this TU. Plotwright: propose a compact outline
(3–5 scenes) with any gateways. Scene Smith: draft the first scene from that outline in the agreed
register. Style Lead: be ready to audit/register‑tune the draft.
```

4. Iterate outline → scene → style. Keep messages short and actionable. Encourage Scene Smith to
   emit `tu.checkpoint` summaries after each scene.

5. Run a pre‑gate check (we’ll do PN after expansion in Stage 2):

```
Gatekeeper: run a pre‑gate on the manuscript. Evaluate Presentation, Integrity, and Style with
player‑safe evidence. Produce a brief gatecheck report and any smallest viable fixes.
```

6. Address gate feedback, then ask Book Binder to bind a player‑safe view:

```
Book Binder: produce a Markdown view for the current Cold snapshot. Include an anchor map summary.
Return a view.export.result envelope with export_artifacts.
```

7. Close the TU when satisfied (or keep it open and continue into Stage 2 for full production):

```
Showrunner: if all bars are green, close the TU. Summarize outcomes, link artifacts, and record
remaining debt if any.
```

## Schema Validation Workflow

**IMPORTANT:** Starting with prompts v0.2.0, schema validation is **mandatory** for all artifacts.

### Why Validation Matters

- **Quality Assurance:** Catches structural errors before they cascade
- **Protocol Compliance:** Ensures artifacts follow Layer 3 specifications
- **Tool Compatibility:** Guarantees downstream tools can process artifacts
- **Traceability:** Provides validation evidence for every artifact

### How Validation Works

**Upload kits v0.2.0+ include:**

1. `validation_contract.md` (file #1) - Non-negotiable validation requirements
2. `SCHEMA_INDEX.json` (file #2) - Schema discovery with $id, paths, SHA-256 hashes
3. Shared patterns and role prompts with validation checkpoints
4. Loop playbooks with validation gates

**Agent workflow:**

1. **Preflight:** Agent reads schema from SCHEMA_INDEX.json, echoes back metadata
2. **Production:** Agent produces artifact with `"$schema"` field
3. **Validation:** Agent validates artifact against canonical schema
4. **Evidence:** Agent produces `validation_report.json` alongside artifact
5. **Gate:** If validation fails, agent STOPS and reports errors (hard gate)

### Validation Contract (v0.2.0+)

When you upload kits v0.2.0+, the validation contract is loaded FIRST (file #1). It requires:

**For every JSON artifact:**

- `"$schema"` field pointing to canonical schema $id
- Validation against schema before emission
- `validation_report.json` with validation evidence
- STOP and report errors if validation fails

**Example validation_report.json:**

```json
{
  "artifact": "hook_card.json",
  "schema": {
    "$id": "https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json",
    "draft": "2020-12",
    "sha256": "a1b2c3..."
  },
  "validator": "jsonschema 4.17.3",
  "timestamp": "2025-11-06T12:00:00Z",
  "valid": true,
  "errors": []
}
```

### Prompting for Validation

**Without validation enforcement (pre-v0.2.0):**

```
Plotwright: Draft a Hook Card for "mysterious letter clue"
```

**With validation enforcement (v0.2.0+):**

```
Plotwright: Draft a Hook Card for "mysterious letter clue"

Validation requirements:
1. Look up hook_card schema in SCHEMA_INDEX.json
2. Echo schema metadata ($id, draft, sha256)
3. Show minimal valid instance
4. Produce artifact with "$schema" field
5. Validate and produce validation_report.json
6. If validation fails: STOP and show errors
```

**Or rely on upload kit (recommended):**

Since v0.2.0 kits include `validation_contract.md` as file #1, agents automatically follow
validation protocol. You can simply:

```
Plotwright: Draft a Hook Card for "mysterious letter clue"
```

Agent will automatically:

- Look up schema
- Run preflight
- Validate before returning
- Produce validation report
- Stop if validation fails

### File Ordering Importance

**Kits v0.2.0+ load files in this order:**

1. `validation_contract.md` ← Rules loaded FIRST
2. `SCHEMA_INDEX.json` ← Discovery mechanism
3. Shared patterns ← Cross-role standards
4. Role-specific content ← Builds on foundation

**Why this matters:** LLMs read uploaded files in order. Loading validation contract FIRST ensures
agents see validation requirements BEFORE producing any outputs.

### Checking Validation Evidence

**After agent produces artifact, verify:**

```
Showrunner: Show me the validation report for the Hook Card
```

Agent should provide `hook_card_validation_report.json` showing:

- `"valid": true` (passed validation)
- Empty `"errors": []` array
- Schema $id used
- Timestamp

**If validation failed:**

- `"valid": false`
- `"errors"` array with specific violations
- Agent should have STOPPED before continuing loop

### Gatekeeper Role Enhanced

Gatekeeper now checks for validation evidence as **Quality Bar 8**:

**Prompt:**

```
Gatekeeper: Run pre-gate on the TU. Include schema validation check.

For each artifact:
1. Verify "$schema" field present
2. Verify validation_report.json exists
3. Verify "valid": true
4. Report any validation failures

GATE RULE: REJECT TU if any artifact lacks validation evidence or failed validation.
```

### Migrating to v0.2.0 Validation

**If using pre-v0.2.0 kits:**

1. Download kits v0.2.0+ from releases
2. Re-upload to ChatGPT/Claude/Gemini
3. Validation is now automatic (no prompt changes needed)

**If validation is missing:**

- Check kit version (should include `validation_contract.md`)
- Verify agent echoes schema during preflight
- Verify `validation_report.json` produced with artifacts
- If agent skips validation: upload v0.2.0 kit (has validation as file #1)

### Schema Canonical URLs

All schemas available at:

```
https://questfoundry.liesdonk.nl/schemas/{schema-name}.schema.json
```

Agents fetch schemas from canonical URLs using SCHEMA_INDEX.json for discovery.

**No need to upload schemas separately** - they're referenced by URL.

---

## Role Cheat Sheets (What To Ask)

- Showrunner (SR)
  - Open TU, wake/dormant roles, plan loops, run checkpoints.
  - Useful prompt: "Open TU, propose plan, and wake PW/SS/ST for a 3–5 scene manuscript."
- Plotwright (PW)
  - Propose outline, hubs/gateways, and nonlinearity constraints.
  - Useful prompt: “Draft a compact outline (3–5 scenes), note any gateways and consequences.”
- Scene Smith (SS)
  - Draft scenes, keep register and continuity; emit checkpoints.
  - Useful prompt: “Write Scene 1 per the outline; add a tu.checkpoint with a 1–2 line summary.”
- Style Lead (ST)
  - Audit tone/register; propose targeted rewrites, update register hints.
  - Useful prompt: “Audit Scene 1 for register drift; propose minimal rewrites.”
- Gatekeeper (GK)
  - Evaluate bars (Presentation, Integrity, Style, Determinism, etc.) with player‑safe evidence.
  - **Runs Cold SoT validation** before allowing Binder (8 preflight checks: manifest validation,
    file existence, SHA-256 verification, asset approval metadata). Schemas at
    <https://questfoundry.liesdonk.nl/schemas/>
  - Useful prompt: "Pre‑gate manuscript; list smallest fixes per bar; don't leak spoilers."
- Book Binder (BB)
  - Bind views (Markdown/HTML) from Cold manifest, summarize anchors, return export artifacts.
  - **All inputs MUST come from `cold/manifest.json`** (no directory scanning/heuristics). Validates
    against schemas at <https://questfoundry.liesdonk.nl/schemas/>
  - Useful prompt: "Bind to Markdown for PN; include anchor_map summary; return paths."
- Player‑Narrator (PN)
  - Perform in‑world; enforce gateways diegetically; report issues.
  - Useful prompt: “Dry‑run performance; never leak internals; send pn.playtest.submit with issues.”
- Lore Weaver (LW) [optional early]
  - If working from canon, provide player‑safe summaries and continuity checks.
- Codex Curator (CC) [optional]
  - Produce codex entries for terms; can run in parallel after draft.
- Art Director / Illustrator (AD/IL) [optional]
  - Shotlists and art prompts after manuscript stabilizes.
- Audio Director / Producer (AuD/AuP) [optional]
  - Cuelist and renders after structure stabilizes.
- Translator (TR) [optional]
  - Localize after English pass; keep style/register consistent.

## Safety, Context, And Memory Tips

- PN Safety: Never route Hot content to PN. If PN is receiver, enforce Cold + `player_safe=true` +
  `snapshot`.
- Context Pressure: Summarize older turns as state notes; keep raw quotes only when needed.
- Traceability: Maintain `correlation_id` and use `refs` to link artifacts.
- Escalation: Use `human.question` when blocked; propose options and a default.

## Production Run — Stage 1 (Minimal Kit Only)

Goal: produce a small, player‑safe manuscript using only the Minimal Upload Kit roles (SR, PW, SS,
ST, GK, BB).

1. Open TU and plan

```
Showrunner: open TU for a 3–5 scene manuscript (loop "Story Spark"). Propose plan and wake PW/SS/ST.
```

2. Outline

```
Plotwright: propose a compact outline with any gateways and their consequences. Keep hubs modest.
```

3. Scene drafting

```
Scene Smith: write Scene 1 per the outline, in the agreed register. Aim for 3+ paragraphs unless this is a micro‑beat transit; if shorter, auto‑extend with a movement/vector paragraph. Emit tu.checkpoint with a 1–2 line summary.
```

4. Style audit

```
Style Lead: audit Scene 1 for register drift; propose minimal rewrites. Keep rationale short.
```

5. Iterate

- Repeat steps 3–4 for Scene 2…N.
- Encourage checkpoints after each scene to keep context tight.

6. Pre‑gate and remediation

```
Gatekeeper: run pre‑gate (Presentation, Integrity, Style). Provide smallest viable fixes and owners.
```

7. Bind a view

```
Book Binder: produce a Markdown view for the current Cold snapshot with anchor_map summary; return export_artifacts.
```

8. Close or proceed

- If satisfied and bars are green, close TU now.
- Or proceed to Stage 2 to enrich with canon/codex, art, audio, PN, and localization.

## Production Run — Stage 2 (Full Kit Expansion)

Attach the Optional zip (or upload the additional role prompts) and continue in the same TU
(preferred) or a follow‑on TU.

1. Canon continuity (Lore Weaver)

```
Lore Weaver: review the manuscript and provide player‑safe summaries and continuity checks (timeline, invariants, refs).
If any changes are needed, propose smallest viable edits without leaking spoilers.
```

2. Codex entries (Codex Curator)

```
Codex Curator: generate/update codex entries for new terms introduced by the manuscript. Ensure entries are player‑safe and
validated against codex_entry.schema.json. Provide crosslinks and coverage notes.
```

3. Art pass (Art Director → Illustrator)

```
Art Director: produce a shotlist for key scenes (subjects, composition, mood).
Illustrator: draft image prompts per the shotlist; align with style guardrails. Do not leak spoilers.
```

4. Audio pass (Audio Director → Audio Producer)

```
Audio Director: propose a cuelist (music/SFX) with motifs and transitions.
Audio Producer: render one or two cues; log parameters. Keep outputs player‑safe.
```

5. Localization (Translator)

```
Translator: produce a language_pack for one locale (e.g., en‑GB → nl‑NL). Keep register consistent; list terms of art.
```

6. PN dry‑run (now that PN prompt is uploaded)

```
Player‑Narrator: perform a dry‑run on the bound Markdown (Cold + player_safe=true + snapshot). Enforce gateways diegetically;
report issues via pn.playtest.submit with smallest fixes and evidence.
```

7. Final gatecheck and binding

```
Gatekeeper: run a full gatecheck (all relevant bars). If green, approve.
Book Binder: produce final exports (Markdown + HTML; add locale exports if available); update anchor_map and coverage.
```

8. Close TU and summarize

```
Showrunner: close TU; summarize outcomes, link artifacts (view exports, cues, images), and list any deferred items.
```

## Calling Explicit Loops (Copy/Paste Prompts)

Use these when you want to explicitly start or switch loops mid‑thread. Send to Showrunner unless
otherwise noted.

- Story Spark (outline → first scenes)

```
Showrunner: open a TU for Story Spark (3–5 scenes). Wake Plotwright, Scene Smith, Style Lead. Propose plan and checkpoints.
```

- Hook Harvest (turn sparks into discrete hooks)

```
Showrunner: start Hook Harvest. Ask Plotwright to extract 3–5 actionable hooks from current notes and file them.
```

- Lore Deepening (lore pass; LW)

```
Showrunner: start Lore Deepening. Wake Lore Weaver. LW: provide player‑safe summaries + continuity checks for the active slice.
```

- Codex Expansion (codex pass; CC)

```
Showrunner: start Codex Expansion. Wake Codex Curator. CC: draft/update codex entries for new terms; validate against schema; provide crosslinks.
```

- Style Tune‑up (register/voice tightening; ST)

```
Showrunner: start Style Tune‑up. Wake Style Lead. ST: audit the current scenes for register drift and propose minimal rewrites.
```

- Binding Run (export views; BB)

```
Showrunner: start Binding Run. Wake Book Binder. BB: bind a Markdown view for the current Cold snapshot; return export_artifacts + anchor_map.
```

- Narration Dry‑Run (player narrates; PN — Cold only)

```
Showrunner: start Narration Dry‑Run. Wake PN. PN: perform a brief dry‑run on the bound Markdown (Cold + player_safe=true + snapshot). Report issues via pn.playtest.submit.
```

- Gatecheck (pre‑gate or full)

```
Showrunner: run pre‑gate. GK: evaluate Presentation, Integrity, Style; provide smallest fixes. If green, proceed to full gatecheck for all relevant bars.
```

- Translation Pass (localization; TR)

```
Showrunner: start Translation Pass. Wake Translator. TR: produce a language_pack for <locale>; keep register; list terms of art.
```

- Art Touch‑up (AD/IL)

```
Showrunner: start Art Touch‑up. Wake Art Director + Illustrator. AD: shotlist key scenes. IL: prompts for each shot (player‑safe).
```

- Audio Pass (AuD/AuP)

```
Showrunner: start Audio Pass. Wake Audio Director + Producer. AuD: cuelist with motifs. AuP: render 1–2 cues and log parameters.
```

- Post‑Mortem (optional health check)

```
Showrunner: run a quick Post‑Mortem. Summarize what worked, what didn’t, and next loop candidates. Emit tu.checkpoint.
```

## Appendix — Uploading Files, Zipping, And Link Folders

Upload individually (preferred): Drag the Minimal kit (10 files) into your chat. Upload PN later as
an addon.

Zips (when hitting attachment limits):

- ChatGPT: Use the Minimal kit zip (all 10 files) or the Full core/optional zips.
- Gemini: Upload the core zip first (≤10 files), then the optional zip (≤10 files).

Auto‑build link folders and zips:

- Preferred: `uv run qfspec-build-kits` (writes to `dist/upload_kits/*`). You’ll get top‑level
  `minimal/` (10 files) + `minimal.zip`, `optional/` + `optional.zip`, and `full/` + `full.zip`.
- Alternatively, scripts:
  - Bash: `spec-tools/scripts/build_upload_kits.sh`
  - PowerShell: `spec-tools/scripts/build_upload_kits.ps1`

See 05-prompts/upload_kits/ for manifests and details.
