# Procedure — gateway.check

procedure (no new protocol intent).

Inputs

- Gateway condition (player-visible cues only) and player state cues.

Preconditions

- Cold view; no internal mechanics exposed.

Process

1. Phrase the check in-world; do not mention codewords/state.
2. If condition plausibly met, proceed; else branch to safe fallback with in-world reason.
3. Log any friction for `pn.playtest.submit` (tag: gate-friction).

Outputs

- PN lines enforcing gateway (runtime); notes for playtest log.
