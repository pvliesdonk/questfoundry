# Playbook — Canon Transfer (Import)

**Use when:** Starting a new project (sequel, shared universe) and you have a **Canon Transfer
Package** from a prior project to seed the canon baseline.

**Outcome:** New project's Hot/Cold initialized with imported canon, timeline anchors, and codex
baseline. Plotwright and Scene Smith receive clear constraint docs. Ready for Story Spark.

---

## 1) One-minute scope (Showrunner)

- [ ] Confirm you have `canon_transfer_package_<source-project>.json`.
- [ ] Open TU: `TU-<date>-LW-CanonTransferImport`.
- [ ] Confirm **project seed idea** doesn't conflict with invariant canon.
- [ ] Timebox: 1-2 hours for import process.

---

## 2) Inputs you need on screen

- **Canon Transfer Package** (`canon_transfer_package_<source-project>.json`).
- **New project seed ideas** (user's initial concept for this project).
- **Project metadata** (new project title, slug).

---

## 3) Do the thing (compact procedure)

**Lore Weaver (R)**

1. **Load transfer package**: Open and parse JSON file.
2. **Validate schema**: Ensure package conforms to `canon_transfer_package.schema.json`.

**Lore Weaver + Gatekeeper (C)**

3. **Conflict detection**:
   - Compare invariant canon against new project seed.
   - If conflict: Escalate to Showrunner (reject import, revise seed, or downgrade invariant to
     mutable).

**Lore Weaver**

4. **Merge invariant canon to Cold**:
   - Create canon packs marked `source: <source-project>`, `immutable: true`.
   - Status: `cold-merged` (cannot be changed).
5. **Seed mutable canon to Hot**:
   - Create canon packs marked `source: <source-project>`, `immutable: false`.
   - Status: `hot-accepted` (can be extended via Lore Deepening).
6. **Import timeline anchors**:
   - Source timeline becomes T0/T1/T2.
   - New project extends from T3+.
7. **Import entity registry**: Copy entities; mark as `source: <source-project>`.

**Codex Curator (C)**

8. **Codex import**: Import inherited codex entries; mark as `inherited: true`,
   `source: <source-project>`.
9. **Flag gaps**: Identify where new project needs additional entries.

**Lore Weaver**

10. **Create constraint documentation**:
    - **Invariants** (cannot change).
    - **Mutables** (can extend).
    - **Timeline guidance** (extend from T3+).
    - **You CAN / You CANNOT** examples.

**Gatekeeper (C)**

11. **Validate import**:
    - **Integrity**: All entity references resolve.
    - **Conflict resolution**: No unresolved contradictions.
    - **Presentation**: Codex baseline player-safe.
    - **Schema**: Imported canon conforms to QF schemas.

**Showrunner (A)**

12. **Approve import**: Confirm merge (invariants to Cold, mutables to Hot).

**Announce**

13. **Notify creative roles**: Plotwright, Scene Smith, Style Lead get constraint docs.

---

## 4) Deliverables (Hot/Cold)

- **Cold canon packs** (invariant canon, immutable).
- **Hot canon packs** (mutable canon, extensible).
- **Codex entries** (inherited baseline, player-safe).
- **Entity registry** (baseline for new project).
- **Timeline foundation** (T0/T1/T2 from source; T3+ for new project).
- **Constraint documentation** (for Plotwright/Scene Smith).
- **TU** (`TU-<date>-LW-CanonTransferImport`).
- **Gatekeeper report** (validation summary).

---

## 5) Handoffs

- **To Story Spark**: Constraint docs, timeline foundation, entity registry.
- **To Lore Deepening**: Mutable canon (can extend via hooks).
- **To Codex Expansion**: Gap list (where new entries needed).

---

## 6) Success criteria (Y/N)

- [ ] Transfer package loads and validates successfully.
- [ ] No unresolved conflicts between imported canon and project seed.
- [ ] Invariant canon merged to Cold (immutable).
- [ ] Mutable canon seeded to Hot (extensible).
- [ ] Codex baseline imported (player-safe).
- [ ] Timeline anchors imported; new project timeline extends from T3+.
- [ ] Entity registry imported and available.
- [ ] Constraint documentation clear and actionable.
- [ ] Gatekeeper approves (Integrity + Conflict resolution + Presentation).
- [ ] Creative roles understand invariants vs. mutables.

---

## 7) Escape hatches

- **Schema validation fails** → Fix package format; re-import.
- **Conflict with project seed**:
  - **Option A**: Reject import (wrong canon baseline).
  - **Option B**: Revise project seed (honor invariant).
  - **Option C**: Downgrade invariant to mutable (coordinate with source project owner).
- **Entity reference broken** → Add missing entity to registry or flag for Lore Deepening.
- **Timeline confusion** → Document timeline offsets clearly; provide examples.
- **Codex spoilers** → Curator revises to player-safe or excludes entry.
- **Creative role confusion** → Lore Weaver clarifies constraint docs; provide use-case examples.

---

**Typical pass:** 1-2 hours for standard import; 3-4 hours with conflict resolution.
