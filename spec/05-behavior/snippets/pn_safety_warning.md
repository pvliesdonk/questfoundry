# PN Safety Warning

**NON-NEGOTIABLE:** Player Narrator receives ONLY Cold snapshot content.

**Hard invariants:**

- Never route Hot content to PN
- If receiver is PN: `context.hot_cold = "cold"`, `context.snapshot` present, `safety.player_safe = true`
- Player-facing text MUST NOT leak internal logic, hidden states, or solution paths

**Forbidden in player surfaces:**

- State variables (e.g., `flag_kestrel_betrayal`)
- Gateway logic (e.g., `if state.dock_access == true`)
- Codewords or meta terminology
- System labels or debug info
- Determinism parameters (seeds, model names)
- Authoring notes or development context

**If violation suspected:** STOP immediately and report via `pn.playtest.submit` or escalate to Showrunner.

**Refer to:** `@procedure:spoiler_hygiene` for complete safety protocol.
