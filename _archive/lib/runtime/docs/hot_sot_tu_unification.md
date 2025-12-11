# Hot SoT & TU Unification (Spec vs Runtime)

> Purpose: capture why Gatekeeper reported “no TU content in hot_sot”, and define a single,
> unambiguous model for TU + Hot SoT across `spec/` and `lib/runtime/` for the upcoming fix.

---

## 1. Desired Model (Target)

- **One Hot SoT instance per loop run**
  - `state.hot_sot` is the only working store for WIP artifacts (drafts, hooks, TU briefs, etc.).
  - There is no second “artifact SoT” competing with it.

- **Single canonical TU artifact location**
  - Active TU brief lives at `state.hot_sot.current_tu` as a `tu_brief` object.
  - `state.meta.current_tu` is a **string TU ID**, pointing at the active TU.
  - Optional history is stored under `state.hot_sot.tus[]` (derived/archive), not a second SoT.

- **Gatekeeper & Showrunner contracts**
  - Showrunner **writes** the active TU brief to `hot_sot.current_tu`.
  - Gatekeeper **reads** the active TU brief from `hot_sot.current_tu` and never from any other
    location.

Backward compatibility is **not** required; the fix can remove/rename legacy paths.

---

## 2. What the Spec Currently Says

### 2.1 Studio State schema

- `spec/03-schemas/definitions/studio_state.schema.json`
  - `meta.current_tu` is defined as a **string TU ID** (`^TU-….` pattern), *not* an object
    (`meta.current_tu`:12–35).
  - `hot_sot` is the working SoT with arrays like `hooks[]`, `tus[]`, etc. (`hot_sot.tus`:60–80).
  - `artifacts.current_tu` is defined as an object:
    `artifacts.current_tu:280–320` — “TU Brief for current trace unit”.

So Layer 3 (schema) already distinguishes:

- `meta.current_tu` → ID string
- `hot_sot.tus[]`   → array of TU briefs
- `artifacts.current_tu` → “current TU Brief” object

### 2.2 Role interfaces (Layer 5)

- `spec/05-definitions/roles/showrunner.yaml:80–118`
  - **Output**: `artifact_type: tu_brief` → `state_key: hot_sot.current_tu`
  - Side effect includes `write_hot_sot`.

- `spec/05-definitions/roles/gatekeeper.yaml:70–106`
  - **Input** (required): `artifact_type: tu_brief` from `state_key: hot_sot.current_tu`.
  - This is the contract Gatekeeper followed in the logs.

Thus Layer 5’s intent is clear: **active TU artifact is `hot_sot.current_tu`.**

### 2.3 TU lifecycle transitions

- `spec/05-definitions/transitions/tu_lifecycle.yaml:48–313`
  - All transitions are expressed in terms of `artifacts.current_tu.*` (status, opened, deliverables,
    closed, etc.).
  - There is **no explicit link** here to `hot_sot.current_tu` or `hot_sot.tus[]`.

So Layer 5 TU lifecycle uses `artifacts.current_tu`, whereas Gatekeeper/Showrunner interfaces use
`hot_sot.current_tu`.

### 2.4 Layer 0–2 narratives

- `spec/00-north-star/SOURCES_OF_TRUTH.md`
  - Defines Hot vs Cold conceptually (Hot = working; Cold = curated), and describes that every
    change destined for Cold has a **change unit** record (TU) with status and bar coverage.

- `spec/02-dictionary/artifacts/tu_brief.md`
  - Defines TU Brief as that change-unit record and references the Gatekeeper/Showrunner/loop
    relationships, but does not pick a storage key; that is delegated to schemas/roles.

---

## 3. What the Runtime Currently Does

### 3.1 State initialization

- `lib/runtime/src/questfoundry/runtime/core/state_manager.py:120–190`
  - Builds one `hot_sot` dict per TU run (good: single Hot SoT instance).
  - Populates arrays/dicts for hooks, tus, drafts, sections, etc.
  - Sets `meta.current_tu = tu_id` (string ID) and does **not** create `hot_sot.current_tu`.

So the runtime already uses:

- `meta.current_tu` as TU ID (aligned with schema).
- A single `hot_sot` instance per loop invocation.

### 3.2 Hot SoT tooling

- `lib/runtime/src/questfoundry/runtime/tools/state_tools.py:140–260`
  - `create_empty_hot_sot()` initializes `hooks`, `tus`, `drafts`, `sections`, etc.
  - `_get_key_to_artifact_mapping()` inverts `{artifact_type → hot_sot_key}` discovered from role
    outputs:
    - For `tu_brief` it picks `current_tu` based on `spec/05-definitions/roles/showrunner.yaml`.
  - `write_hot_sot(key="current_tu", value=...)` is the canonical way to write a TU brief artifact.

Thus, in runtime tooling, **`current_tu` is the hot_sot key for `tu_brief`.**

### 3.3 Role prompt wiring

- `lib/runtime/src/questfoundry/runtime/resources/definitions/roles/showrunner.yaml`
  - Imports `state.meta.current_tu` into the Showrunner prompt (`name: current_tu; source:
    state.meta.current_tu`) and uses `hot_sot` for surrounding artifacts.

- `lib/runtime/src/questfoundry/runtime/resources/definitions/roles/gatekeeper.yaml:40–140`
  - Uses `state.meta.current_tu.deliverables` for the `tu_deliverables` prompt variable.
  - Uses `state.hot_sot.*` for canon, drafts, etc.

This is inconsistent with the schema, where `meta.current_tu` is a string; here it is treated as an
object.

### 3.4 TU lifecycle (runtime view)

- `lib/runtime/src/questfoundry/runtime/resources/definitions/transitions/tu_lifecycle.yaml`
  - Still mirrors the spec version: transitions are expressed using `artifacts.current_tu.*`
    (status, deliverables, etc.).
  - There is no wiring here that ensures `artifacts.current_tu` and `hot_sot.current_tu` are the
    same object.

### 3.5 What happened in the failing run

From `lib/runtime/logs2`:

- Showrunner never calls `write_tu_brief` or `write_hot_sot(key="current_tu")` for
  `TU-2025-12-05-SR01`. It only sends `tu.open` protocol messages with an inline TU Brief in the
  payload (`logs2/tool-calls.jsonl:129`).
- Other roles do write artifacts to Hot SoT:
  - `hot_sot.topology_notes`, `hot_sot.section_briefs`, `hot_sot.drafts` (`logs2/state-sot.jsonl:16–22`).
- Gatekeeper calls `read_hot_sot(key="customer_directives")` (OK) and then
  `read_hot_sot(key="current_tu")`:
  - `read_hot_sot("current_tu")` → `null` (`logs2/tool-calls.jsonl:141`,
    `logs2/state-sot.jsonl:21`), because `hot_sot.current_tu` was never written.
  - Prompt tells it this key is **required**, so it emits:
    - “Current TU is null. Please submit the task unit…”
    - “No TU content found in hot_sot…”

At the same time, the Gatekeeper prompt variables are trying to read from
`state.meta.current_tu.deliverables`, which does not exist according to the schema and is not
aligned with the spec’s Gatekeeper interface.

---

## 4. The Actual Inconsistencies

1. **Two “current_tu” concepts**
   - Schema: `meta.current_tu` (string ID) vs `artifacts.current_tu` (TU object).
   - Roles: Showrunner/Gatekeeper use `hot_sot.current_tu` for the TU brief artifact.
   - Runtime prompts: treat `meta.current_tu` as if it were the TU object.

2. **TU lifecycle vs Gatekeeper/Showrunner**
   - TU lifecycle (`spec/05-definitions/transitions/tu_lifecycle.yaml`) uses `artifacts.current_tu`
     for status and fields.
   - Gatekeeper + Showrunner interfaces use `hot_sot.current_tu`.
   - There is no explicit spec statement saying “`artifacts.current_tu` and
     `hot_sot.current_tu` are the same object.”

3. **Runtime doesn’t force TU brief into Hot SoT for `tu.open` path**
   - `StateManager.initialize_state` sets `meta.current_tu = tu_id` but does not create
     `hot_sot.current_tu`.
   - The `tu.open` protocol handler (in the control plane) currently only opens the loop and passes
     TU info in messages; it does **not** call `write_hot_sot("current_tu")`.

4. **Gatekeeper’s prompt vs its tool usage**
   - Prompt: “Use `read_hot_sot(key="current_tu")` → tu_brief [REQUIRED]”.
   - Variables: `tu_deliverables` are wired from `state.meta.current_tu.deliverables`.
   - Tools: `read_hot_sot("current_tu")` correctly reads `hot_sot.current_tu`, which is `null` in
     the failing run.

This is why Gatekeeper complained: the spec contract (`hot_sot.current_tu` required) wasn’t
followed in the TU-open path, and the prompts mixed “meta TU” and “Hot SoT TU” language.

---

## 5. Fix Direction (Spec + Runtime), No Backwards Compat Required

Given backwards compatibility is not needed, the clean fix is to **collapse all “current_tu”
artifact references into `hot_sot.current_tu` and treat everything else as control-plane or
derived.**

### 5.1 Spec changes

1. **Clarify canonical TU locations in `studio_state.schema.json`**
   - `meta.current_tu`: keep as **string ID** (no change), document explicitly:
     “TU ID; pointer to the active TU Brief in `hot_sot.current_tu`.”
   - Add `hot_sot.current_tu` to the schema (object, `tu_brief`-shaped) or at least document it as
     the canonical home for the active TU brief.
   - Mark `artifacts.current_tu` as **deprecated** or remove it entirely if not used elsewhere.

2. **Align TU lifecycle spec with Hot SoT**
   - In `spec/05-definitions/transitions/tu_lifecycle.yaml`, change all references from
     `artifacts.current_tu.*` to `hot_sot.current_tu.*`, or explicitly state that transitions MUST
     keep `artifacts.current_tu` and `hot_sot.current_tu` identical (if `artifacts` is kept).
   - Ensure any transition that modifies TU status, deliverables, or closure is updating the same
     object Gatekeeper reads.

3. **Clean up unclear references in runtime docs**
   - `spec/06-runtime/components/showrunner_agent.md:316` currently treats
     `state.get("meta", {}).get("current_tu", {}).get("status")` as if `meta.current_tu` were a
     dict. Update this to read:
       - TU ID from `meta.current_tu` (string), and
       - TU status from `hot_sot.current_tu.status`.

4. **Document the single Hot SoT / single TU rule**
   - In `SOURCES_OF_TRUTH.md` and `tu_brief.md`, add a short normative note:
     - “The active TU Brief MUST be stored at `hot_sot.current_tu`. TU ID is `meta.current_tu`.
        There is exactly one Hot SoT instance per loop run; Views are cut from Cold only.”

### 5.2 Runtime changes

1. **On TU open, always write Hot SoT**
   - In the control plane logic that handles `tu.open` (the part that currently just creates loop
     graphs and passes a TU brief in the envelope payload):
     - When a TU Brief is present in `tu.open` payload, call `write_hot_sot(key="current_tu",
       value=<tu_brief>)` before or as part of starting the loop.
     - Ensure `meta.current_tu` (TU ID string) matches `hot_sot.current_tu.id`.

2. **Make Gatekeeper prompt read TU from Hot SoT**
   - In `lib/runtime/src/questfoundry/runtime/resources/definitions/roles/gatekeeper.yaml`:
     - Change `tu_deliverables` source from `state.meta.current_tu.deliverables` to
       `state.hot_sot.current_tu.deliverables`.
   - Now the prompt and `read_hot_sot("current_tu")` will refer to the same object.

3. **Stop treating `meta.current_tu` as an object**
   - Any runtime templates or components that do `meta.current_tu.<field>` must be updated to:
     - Use `meta.current_tu` as ID only, and
     - Pull actual TU fields from `hot_sot.current_tu`.

4. **Remove or rewrite `artifacts.current_tu` usage**
   - If you don’t need `artifacts.current_tu` at runtime, you can:
     - Remove it from the runtime schema/resources, and
     - Update TU lifecycle code to read/write `hot_sot.current_tu` directly.
   - If you retain it in schemas for tooling, implement a strict mirror:
     - Whenever `hot_sot.current_tu` is written, update `artifacts.current_tu` to the same
       object, and vice versa.

5. **TU list / inspection commands**
   - `spec/06-runtime/components/cli.md` describes `qf list tus` & `qf show <tu_id>`. These should
     be implemented against:
       - `hot_sot.tus[]` (history) and/or Cold snapshot records for TUs, not any separate TU store.

---

## 6. Summary

- The bug in the examined run is real but **root cause is structural**: spec and runtime split TU
  state across `meta`, `hot_sot`, and `artifacts` without a single canonical location.
- Spec already intends `hot_sot.current_tu` to be the active TU Brief (Gatekeeper/Showrunner
  interfaces). Runtime tooling (`write_hot_sot`) is aligned with that.
- The inconsistent parts are:
  - `tu_lifecycle.yaml` still operating on `artifacts.current_tu.*`, and
  - Prompts/runtime examples treating `meta.current_tu` as a rich object instead of an ID.
- With no backward-compat requirement, the straightforward fix is:
  - Route all TU creation (including `tu.open`) through `write_hot_sot("current_tu")`,
  - Make all role prompts read TU content from `hot_sot.current_tu`,
  - Treat `meta.current_tu` strictly as a TU ID, and
  - Remove or update `artifacts.current_tu` so there is exactly one Hot SoT / TU artifact home.
