# Protocol Changelog

All notable changes to QuestFoundry protocol specifications will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-11-10

### Added

- **Canon Workflow Domain (`canon.*`):**
  - New protocol domain for canon-centric workflow operations
  - Enables canon transfer between projects and proactive worldbuilding

- **New Intents:**
  - `canon.transfer.export` - Export Canon Transfer Package from completed project
    - Responsible: Lore Weaver, Approver: Showrunner
    - Schema: `canon_transfer_package.schema.json`
    - Packages stabilized canon as invariant (immutable) or mutable (extensible) for downstream
      projects
  - `canon.transfer.import` - Import Canon Transfer Package into new project
    - Responsible: Lore Weaver, Approver: Showrunner
    - Schema: `canon_transfer_package.schema.json`
    - Includes conflict detection and resolution for invariant canon vs. project seed ideas
  - `canon.genesis.create` - Create World Genesis Manifest
    - Responsible: Lore Weaver, Approver: Showrunner
    - Schema: `world_genesis_manifest.schema.json`
    - Tracks proactive worldbuilding execution before Story Spark

- **Authorization Matrix:**
  - Added `canon.*` domain authorization entry
  - Lore Weaver responsible for all canon workflow intents
  - Showrunner approves; broadcasts to all roles

### Context

This release extends the protocol to support canon-centric workflows that complement the standard
story-driven flow. Canon Transfer enables shared universes, sequels, and franchise continuity by
exporting/importing canon between projects with clear invariant/mutable boundaries. World Genesis
supports epic fantasy/sci-fi projects requiring extensive worldbuilding before plot design. All
three intents integrate with existing Gatekeeper validation and TU traceability.

## [0.2.1] - 2025-11-06

### Summary

Initial stable release of QuestFoundry Protocol specification. Defines complete message-passing
protocol for role collaboration, quality enforcement, and player-safe export boundaries. All core
lifecycles, flows, and conformance requirements are specified and ready for implementation.

### Added

**Core Protocol Specifications:**

- **`ENVELOPE.md`** — Complete envelope v0.2.1 specification
  - Protocol metadata and semver versioning
  - Message identity, routing, and correlation
  - Hot/Cold context tracking
  - PN safety boundaries (non-negotiable)
  - TU traceability requirements
  - Payload structure and schema validation
  - Error handling and forward compatibility
- **`envelope.schema.json`** — JSON Schema Draft 2020-12 for envelope validation
- **`INTENTS.md`** — Complete intent catalog for all protocol domains
  - Hook lifecycle intents (create, accept, start, resolve, canonize, etc.)
  - TU lifecycle intents (open, submit, merge, defer, etc.)
  - Gate intents (report, decision, remediation)
  - View/export intents (snapshot selection, binding, PN feedback, publish)
  - Artifact submission intents for all Layer 2 types
- **`CONFORMANCE.md`** — Protocol conformance requirements and validation procedures
  - Envelope validation rules
  - Payload schema validation requirements
  - PN safety enforcement rules
  - Traceability requirements
  - Test matrix and validation scripts

**Lifecycle State Machines (4 Complete):**

- **`LIFECYCLES/hooks.md`** — Hook Card lifecycle v0.2.1
  - 7-state machine (proposed → accepted → in-progress → resolved → canonized/deferred/rejected)
  - Complete transition matrix with sender role authorization
  - Required intents and payload schemas per transition
  - Blocking hook enforcement and quality bar integration
- **`LIFECYCLES/tu.md`** — Trace Unit lifecycle v0.2.1
  - 6-state machine (hot-proposed → stabilizing → gatecheck → cold-merged/deferred/rejected)
  - TU-bound requirement for all Cold artifacts
  - Gatecheck enforcement and quality gates
  - Dormancy/deferral integration with wake rubric
- **`LIFECYCLES/gate.md`** — Gatecheck lifecycle v0.2.1
  - 4-state machine (pre-gate → gatecheck → decision)
  - Quality bar evaluation rules (green/yellow/red per bar)
  - Smallest viable fix requirements
  - Merge approval and snapshot stamping coordination
- **`LIFECYCLES/view.md`** — View/Export lifecycle v0.2.1
  - 5-state machine (snapshot-selected → export-binding → pn-dry-run → feedback-collected →
    view-published)
  - Cold-only enforcement (PN never sees Hot)
  - Player-safety validation at every boundary
  - Snapshot reference requirements for reproducibility

**End-to-End Flow Specifications (6 Core Flows):**

- **`FLOWS/hook_harvest.md`** — Hook Harvest message sequences
  - Hook creation, classification, and routing
  - Showrunner triage and loop assignment
- **`FLOWS/lore_deepening.md`** — Lore Deepening message sequences
  - Canon development from accepted hooks
  - Player-safe summary generation
  - Handoff to Codex Expansion
- **`FLOWS/codex_expansion.md`** — Codex Expansion message sequences
  - Player-safe codex entry creation
  - Spoiler hygiene enforcement
  - Gatecheck and Cold merge coordination
- **`FLOWS/gatecheck.md`** — Gatecheck and Merge message sequences
  - Gate report submission with bar status
  - Gatekeeper decision (pass/conditional pass/block)
  - Remediation coordination and merge approval
  - Snapshot stamping per TRACEABILITY
- **`FLOWS/binding_run.md`** — Binding Run message sequences
  - Export binding from Cold snapshots
  - Player-safe surface assembly
  - View log generation
- **`FLOWS/narration_dry_run.md`** — Narration Dry-Run message sequences
  - PN playtesting workflow
  - UX feedback collection and routing
  - Friction point identification

**Examples and Validation:**

- **`EXAMPLES/`** — Complete example message suite
  - Envelope examples for all lifecycle states
  - Intent payload examples for all artifact types
  - Multi-message flow sequences
  - Error cases and validation failures
- **Validation Scripts:**
  - `scripts/validate-examples.sh` (Bash/Unix)
  - `scripts/validate-examples.ps1` (PowerShell/Windows)
  - CI integration examples (GitHub Actions)

### Context

This release completes Layer 4 protocol specification, providing the foundation for:

- **Layer 5** (Prompts) — AI agent system prompts that implement protocol roles
- **Layer 6** (Libraries) — SDK implementations in Python/TypeScript
- **Layer 7** (UI) — CLI/GUI tools that consume protocol messages

The protocol is transport-agnostic and can be implemented over HTTP APIs, file-based workflows, or
event-driven architectures. See CONFORMANCE.md for implementation requirements.

### Design Decisions

**Included in v0.2.1:**

- All 4 core lifecycles (Hook, TU, Gate, View) — covers creation through export
- 6 core workflow flows — covers primary collaboration patterns
- Complete intent catalog — enables all Layer 0 loops
- Conformance requirements — enables implementation validation

**Deferred to Future Releases:**

- Additional loop flows (Story Spark, Style Tune-up, Art/Audio passes, Translation, Post-Mortem) —
  follow established patterns, can be added in minor releases
- Non-normative transport mappings (HTTP, files, events) — implementation guidance, not protocol
  requirements
- Performance benchmarks — implementation-specific, not protocol requirements

---

## Version Numbering Strategy

Protocol versions will follow **Semantic Versioning**:

- **Major (X.0.0)**: Breaking changes to envelope structure, intent taxonomy, or required fields
- **Minor (1.X.0)**: New intents, new lifecycle states, backward-compatible additions
- **Patch (1.0.X)**: Clarifications, documentation fixes, example updates

### Breaking Change Policy

Breaking changes require:

1. Major version bump
2. Migration guide in release notes
3. Deprecation notice in previous minor version (when possible)
4. Support window for old version (TBD by maintainers)

### Backward Compatibility

- Envelope `protocol.version` field enables version negotiation
- New intents are backward-compatible (ignored by old implementations)
- New optional fields are backward-compatible
- Required field additions are breaking changes

---

## Release Process

When cutting a protocol release:

1. Update this CHANGELOG.md with complete release notes
2. Update `04-protocol/README.md` with new version status
3. Commit: `protocol: bump to vX.Y.Z`
4. Tag: `git tag -a protocol-vX.Y.Z -m "Release protocol vX.Y.Z: description"`
5. Push: `git push origin protocol-vX.Y.Z`
6. Create GitHub release with changelog excerpt
7. Update dependent documentation (Layer 5 prompts if needed)

---

## Relationship to Schema Versions

- **Protocol versions** (Layer 4) are independent of **schema versions** (Layer 3)
- Protocol specifies message structure (envelope, intents, flows)
- Schemas specify artifact payload structure (hook cards, TU briefs, etc.)
- A protocol version may reference specific schema versions in documentation
- Breaking schema changes do not automatically require protocol version bump

Example:

- `protocol-v0.2.1` may work with `schemas-v0.2.0` through `schemas-v0.5.0`
- If envelope structure changes, bump protocol version
- If artifact payload structure changes, bump schema version

---

## Notes

- Protocol v0.2.1 released 2025-11-06
- Layer 4 (Protocol) now at 100% completion
- See `03-schemas/CHANGELOG.md` for artifact schema version history (current: `schemas-v0.2.0`)
- See root `README.md` for overall layer completion status
- Protocol and schema versions are independent; see "Relationship to Schema Versions" above

---

**Changelog created:** 2025-11-06 **First release:** protocol-v0.2.1 (2025-11-06)

[Unreleased]: https://github.com/pvliesdonk/questfoundry-spec/compare/protocol-v0.2.1...HEAD
[0.2.1]: https://github.com/pvliesdonk/questfoundry-spec/releases/tag/protocol-v0.2.1
