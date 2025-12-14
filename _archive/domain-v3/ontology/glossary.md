# QuestFoundry Glossary

> **Purpose:** Common language for QuestFoundry v3.
> Clear terms prevent confusion across roles and documentation.

---

## Narrative Topology

**Scene**
A unit of prose that ends in one or more choices. Scenes are the atomic content containers filled by Scene Smith.

**Choice**
A player-facing action at the end of a scene that links to another scene. Choices should be contrastive (meaningfully different).

**Hub**
A scene (or cluster) with meaningful fan-out to multiple distinct routes. Hubs are entry points for exploration.

**Loop**
A design pattern that returns the player to a prior location *with difference*—new context, options, or consequences. Not a dead circle.

**Gate**
A diegetic condition controlling access to content (token, reputation, knowledge, tool). Gates are always phrased in-world, never as system messages.

**Codeword**
A hidden state flag used by the system to alter future outcomes. Codeword names never appear on player surfaces.

---

## Sources of Truth

**Hot (hot_store)**
Discovery space for working drafts, hooks, canon work-in-progress, and plan-only assets. Spoilers allowed. Mutable.

**Cold (cold_store)**
Curated canon and player-safe surfaces. No spoilers, no internal labels. Append-only once canonized.

**Snapshot**
An immutable tag of cold_store at a point in time (e.g., `cold@2025-10-28`). Exports are cut from snapshots.

**View**
A player-safe export bundle (MD/HTML/EPUB/PDF) built from a snapshot by Publisher.

---

## Work Units

**Brief**
A work order that defines scope for a focused task. Specifies active roles, quality bars to press, and exit criteria.

**HookCard**
A compact proposal (idea, question, gap) small enough to triage. The primary unit of work tracking.

**Canon**
Spoiler-level truth authored by Lorekeeper—causal chains, timelines, invariants. Player-safe summaries go to the Codex, not surfaces.

**Artifact**
Any typed data object in the system (Brief, Scene, CanonEntry, HookCard, etc.). Artifacts have lifecycles and belong to stores.

---

## Workflow

**Loop (Workflow)**
A content workflow with defined graph structure (e.g., Story Spark, Hook Harvest, Scene Weave). Not to be confused with narrative loops.

**Playbook**
An operational procedure for recovery, setup, or meta-processes (e.g., gate_failure, emergency_retcon). Playbooks don't produce content artifacts.

**Gatecheck**
A validation checkpoint where Gatekeeper applies quality bars. Content must pass gatecheck before canonization.

**Canonization**
The process of promoting approved hot_store content to cold_store. Handled by Canon Commit loop.

---

## Quality Bars

**Integrity**
No contradictions in canon. Facts must be consistent.

**Reachability**
All content accessible via valid paths. No orphaned scenes.

**Nonlinearity**
Multiple valid paths exist. Avoid railroading.

**Gateways**
All gates have valid unlock conditions. Fair and achievable.

**Style**
Voice and tone consistency. Match the register.

**Determinism**
Same inputs produce same outputs. Reproducibility when promised.

**Presentation**
Formatting correct, spoiler-safe, no internal labels on surfaces.

**Accessibility**
Content usable by all players. Alt text, captions, readable structure.

---

## The Eight Roles

**Showrunner (SR)**
Strategic orchestrator. Manages by exception, delegates work, approves merges. High agency.

**Lorekeeper (LK)**
Guardian of canonical truth. Verifies facts, maintains consistency. Medium agency.

**Narrator (NR)**
Improvisational storyteller. Runs interactive sessions, respects gates. High agency.

**Publisher (PB)**
Deterministic assembler. Transforms approved content into exports. Zero creative discretion.

**Creative Director (CD)**
Aesthetic visionary. Ensures voice, tone, style coherence. High agency.

**Plotwright (PW)**
Structural architect. Designs topology, hubs, gates. Medium agency.

**Scene Smith (SS)**
Prose craftsman. Fills structural shells with narrative. Medium agency.

**Gatekeeper (GK)**
Quality auditor. Enforces bars, validates content. Low agency (rules-based).

---

## Safety Terms

**Player-safe**
Content that can appear on surfaces: manuscript, codex, captions. No spoilers, no internal labels, no technique details.

**Spoiler hygiene**
The practice of keeping revelations, twists, and internal logic out of player-facing content.

**Diegetic**
In-world, from the story's perspective. Gates, explanations, and choices should be diegetic, not meta.

**Meta**
Out-of-world, referring to system mechanics. Meta language ("click here," "missing FLAG_X") is forbidden on surfaces.

---

## Production Terms

**Shotlist**
Visual asset requirements for a section. Defines what to depict without specifying technique.

**AudioPlan**
Sound requirements for a section. Ambient, music cues, SFX without DAW/plugin details.

**TranslationPack**
Localization status for a content slice. Tracks coverage, glossary terms, blockers.

**Coverage**
Percentage of content that's completed for a given dimension (translation, art, audio).

**Deferred**
Asset planned but not yet produced. Marked in exports so players know what's pending.

---

## Protocol Terms

**Intent**
A declaration of work status and routing to the next role. Intents drive the workflow graph.

**Handoff**
An intent type indicating work is ready for the next role.

**Escalation**
An intent type indicating a blocker that requires Showrunner or human intervention.

**DelegationResult**
The return value from a role to SR, containing status, artifacts, and recommendations.

---

## Miscellaneous

**TU (Trace Unit)**
Legacy term from v2. In v3, use "Brief" for work tracking.

**ADR (Architecture Decision Record)**
Documentation of significant policy or architecture decisions. Not part of normal workflow.

**Lineage**
A pointer noting the source (Brief, TU) that produced an artifact. Supports traceability.
