# Playbook — Canon Transfer (Export)

**Use when:** Project is complete and you want to **export canon** for a sequel, shared universe, or
franchise continuation.

**Outcome:** A validated **Canon Transfer Package** artifact containing tagged canon
(invariant/mutable), codex baseline, timeline anchors, and entity registry. Ready for import into a
new project.

---

## 1) One-minute scope (Showrunner)

- [ ] Confirm project is **finalized** (final Binding Run complete, ready for archival).
- [ ] Open TU: `TU-<date>-LW-CanonTransferExport`.
- [ ] Confirm **export scope**: which canon elements are foundational vs. story-specific.
- [ ] Timebox: 2-3 hours for export process.

---

## 2) Inputs you need on screen

- **Cold canon packs** from completed project.
- **Codex entries** (to identify player-safe baseline).
- **Project metadata** (title, slug, final snapshot ID).
- **Timeline** from canon packs (T0/T1/T2 anchors).

---

## 3) Do the thing (compact procedure)

**Lore Weaver (R)**

1. **Canon inventory**: List all canon packs from Cold.
2. **Tag canon elements**:
   - **Invariant** (immutable): Core world rules, source project resolutions.
   - **Mutable** (extensible): Rosters, layouts, partial details.
   - **Local** (exclude): Story-specific arcs, spoilers.
3. **Extract timeline anchors**: T0/T1/T2 from canon packs.
4. **Build entity registry**: Characters, places, factions, items defined in project.

**Codex Curator (C)**

5. **Codex baseline**: Identify player-safe entries for universe-wide transfer; mark as
   `inherited: true`.

**Lore Weaver**

6. **Assemble package**:
   - Metadata (source project, export date, snapshot ID, version).
   - Invariant canon array.
   - Mutable canon array.
   - Codex baseline array.
   - Timeline anchors.
   - Entity registry.
   - Downstream constraints (what can/cannot change).

**Gatekeeper (C)**

7. **Validate**:
   - **Integrity**: All entity references resolve.
   - **Presentation**: No source project spoilers in invariant canon/codex.
   - **Consistency**: Timeline ordered correctly.
   - **Schema**: Package validates against `canon_transfer_package.schema.json`.

**Showrunner (A)**

8. **Approve and export**: Save as `canon_transfer_package_<project-slug>.json`.

---

## 4) Deliverables (Cold)

- **Canon Transfer Package** (`canon_transfer_package_<project-slug>.json`).
- **TU** (`TU-<date>-LW-CanonTransferExport`).
- **Gatekeeper report** (validation summary).

---

## 5) Handoffs

- **To next project**: Canon Transfer Package for import.
- **To archival**: TU and Gatekeeper report for project records.

---

## 6) Success criteria (Y/N)

- [ ] All invariant canon elements tagged and included.
- [ ] All mutable canon elements tagged and included.
- [ ] Local canon excluded from package.
- [ ] Entity registry complete; all references resolve.
- [ ] Codex baseline is player-safe and spoiler-free.
- [ ] Timeline anchors chronologically ordered.
- [ ] Package validates against schema.
- [ ] Gatekeeper approves (Integrity + Presentation + Consistency).

---

## 7) Escape hatches

- **Invariant tagging unclear** → Consult Showrunner; defer ambiguous elements to mutable.
- **Entity reference broken** → Add missing entity to registry or remove reference.
- **Codex baseline contains spoilers** → Curator revises to player-safe summary or excludes entry.
- **Timeline conflicts** → Reorder anchors or remove conflicting anchor.
- **Schema validation fails** → Fix package structure; re-validate.

---

**Typical pass:** 2-3 hours for standard project; 4-6 hours for epic worldbuilding project.
