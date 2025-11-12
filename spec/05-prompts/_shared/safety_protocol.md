# Shared Pattern — Safety Protocol

Target: GPT-5 (primary)

Purpose

- Enforce PN safety boundaries and spoiler hygiene
- Guard Presentation and Accessibility bars by construction

References

- 00-north-star/PN_PRINCIPLES.md
- 00-north-star/QUALITY_BARS.md
- 00-north-star/SPOILER_HYGIENE.md
- 04-protocol/ENVELOPE.md

Hard Invariants

- Never route Hot content to PN.
- If receiver is PN, `context.hot_cold = cold`, `context.snapshot` present, and
  `safety.player_safe = true`.
- Player-facing text MUST NOT leak internal logic, hidden states, or solution paths.

Spoiler Hygiene

- Prefer diegetic hints; avoid meta-commentary and system labels.
- Redact or paraphrase sensitive lore on PN surfaces; preserve full detail in Cold surfaces.

Presentation & Accessibility

- Use consistent register and terminology; avoid jarring shifts (Style bar).
- Include alt text, caption plans, and reading-order considerations where relevant.

Pre-send Checks

- Verify receiver and PN invariants.
- If referencing gated content, ensure the evidence is player-safe.
- Gatekeeper decisions should not expose hidden logic in evidence text.
