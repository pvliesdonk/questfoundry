# Role Comparison Matrix

**Note:** This file provides a quick reference for role abbreviations and key protocol intents. For
comprehensive role information, see:

- **Role definitions and responsibilities**: [`01-roles/charters/`](../01-roles/charters/)
- **Role abbreviations**:
  [`02-dictionary/role_abbreviations.md`](../02-dictionary/role_abbreviations.md)
- **Loop participation (RACI)**: Each loop playbook in [`loops/`](./loops/) contains a complete RACI
  matrix

## Quick Reference

### Role Abbreviations

From [`02-dictionary/role_abbreviations.md`](../02-dictionary/role_abbreviations.md):

- SR — Showrunner
- GK — Gatekeeper
- PW — Plotwright
- SS — Scene Smith
- ST — Style Lead
- LW — Lore Weaver
- CC — Codex Curator
- RS — Researcher
- AD — Art Director
- IL — Illustrator
- AuD — Audio Director
- AuP — Audio Producer
- TR — Translator
- BB — Book Binder
- PN — Player-Narrator

### Dormancy & Activation

- **Always Active**: SR, GK
- **Active by Default**: PW, SS, ST, LW, CC
- **Optional/Dormant**: RS, AD, IL, AuD, AuP, TR (SR controls activation)
- **Downstream**: BB, PN

SR controls role activation; roles may be set dormant between loops.

### Key Protocol Intents by Role

**Showrunner (SR):**

- `human.question`, `human.response` — Customer interface
- `tu.open`, `tu.update`, `tu.checkpoint`, `tu.close` — TU lifecycle
- `role.wake`, `role.dormant` — Role orchestration

**Gatekeeper (GK):**

- `gate.report.submit`, `gate.decision` — Quality bar validation

**Book Binder (BB):**

- `view.export.request`, `view.export.result` — Export operations

**Art/Illustrator (AD/IL):**

- Produces: `art_plan`, `shotlist` artifacts

**Audio (AuD/AuP):**

- Produces: `audio_plan`, `cuelist` artifacts

**Translator (TR):**

- Produces: `language_pack`, `register_map` artifacts

**Player-Narrator (PN):**

- `pn.playtest.submit` — Playtest feedback (Cold only, player_safe=true)

### LLM Requirements

- **Most roles**: JSON mode, low temperature (0.2-0.5) for consistency
- **PN surfaces**: Strict safety enforcement (Cold data only, player_safe=true)
- **Creative roles** (PW, SS, ST, LW): May benefit from slightly higher temperature (0.5-0.7) for
  variety
