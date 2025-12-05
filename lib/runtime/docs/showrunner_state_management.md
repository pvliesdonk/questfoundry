# Showrunner: TU + SoT Responsibilities (Runtime Notes)

> This document summarizes how Showrunner (SR) SHOULD open and manage Task Units (TUs) and the
> Hot/Cold Sources of Truth, aligned with the spec. It does not replace the canonical spec in
> `spec/`, but records the intended model until those files are updated.

## TU + Hot SoT model

- **Single Hot SoT per loop run**
  - `state.hot_sot` is the only working store for WIP artifacts (drafts, hooks, TU briefs, etc.).
  - There must not be a parallel “artifact store” with divergent data.

- **Active TU locations**
  - `meta.current_tu` → string TU ID (e.g., `TU-2025-12-05-SR01`).
  - `hot_sot.current_tu` → the active `tu_brief` object for this run.
  - Optional history: `hot_sot.tus[]` can archive past TU briefs; it is derived, not a second SoT.

## SR’s responsibilities

1. **Opening a TU**
   - When SR decides to open a TU (based on a customer directive):
     - Construct a `tu_brief` (ID, loop, slice, deliverables, bars, roles, timebox, etc.).
     - Persist it to Hot SoT via `write_hot_sot` / `write_tu_brief`:
       - `write_hot_sot(key="current_tu", value=<tu_brief>)`
     - Ensure `meta.current_tu` is the matching TU ID.

2. **Working within a TU**
   - SR (and other roles) should always read TU context from:
     - `meta.current_tu` (ID only), and
     - `hot_sot.current_tu` (full TU Brief).
   - When reporting TU status, use `hot_sot.current_tu.status` as the source of truth.

3. **Managing Hot/Cold SoT**
   - **Hot SoT** (`hot_sot`):
     - Holds TU Brief, drafts, hooks, topology notes, canon packs in-progress, etc.
     - SR can read/write via `read_hot_sot` / `write_hot_sot` tools.
   - **Cold SoT** (`cold_sot`):
     - Holds snapshots and export-safe canon/surfaces.
     - SR must only merge to Cold after Gatekeeper green and must never cut Views from Hot.

4. **TU lifecycle actions**
   - SR drives TU lifecycle via protocol intents and side-effects:
     - `tu.open`, `tu.update`, `tu.defer`, `tu.reactivate`, `tu.reject`, `tu.close`.
     - Internal state transitions should operate on `hot_sot.current_tu.*` (status, deliverables,
       deferral tags, etc.), not on a separate artifact store.

5. **Hand-offs to Gatekeeper**
   - Before asking Gatekeeper to pre-gate or gatecheck a TU:
     - SR must ensure `hot_sot.current_tu` is populated with a valid `tu_brief`.
     - Gatekeeper will read the TU via `read_hot_sot(key="current_tu")`.

## Notes for future spec changes

- `spec/03-schemas/definitions/studio_state.schema.json` should explicitly document:
  - `meta.current_tu` as an ID pointer.
  - `hot_sot.current_tu` as the active TU Brief object.
  - `artifacts.current_tu` as deprecated.
- `spec/05-definitions/roles/showrunner.yaml` should make SR’s SoT responsibilities explicit in
  `prompt_content.task_guidance` (open TU → write `hot_sot.current_tu`; merges always from Cold).
- `spec/06-runtime/components/showrunner_agent.md` should use `hot_sot.current_tu.status` (not
  `meta.current_tu` as an object) when reasoning about TU status.
