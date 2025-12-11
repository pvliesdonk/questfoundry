# Tracelog — Cold Snapshots & Views

Append-only human ledger of what landed in **Cold**, when we stamped **snapshots**, and which
**views** were cut. Keep entries spoiler-safe.

---

## cold@YYYY-MM-DD

- **Merged TUs**
  - <tu-id> — <title>; Gatekeeper: <green | notes>
  - …
- **View(s)**
  - **view-YYYYMMDD-<slug>**
    - Options: <art plan/renders, audio plan/assets, languages with coverage>
    - Accessibility: <alt text yes, captions yes, print-friendly yes>
    - Included TU titles since previous view:
      - <tu-id> — <player-safe title>
    - Known limitations: <e.g., art renders deferred; nl 74%>

---

<!-- Example -->

## cold@2025-10-28

- **Merged TUs**
  - tu-kestrel-hub-20251027 — Story Spark (Act I hub); Gatekeeper: green
  - tu-codex-docks-20251028 — Codex Expansion (Dock infrastructure); green
- **View(s)**
  - **view-20251028-a1**
    - Options: art-plan only; audio-none; lang: en (100%), nl (74%)
    - Accessibility: alt yes; captions n/a; print-friendly yes
    - Included TU titles:
      - tu-kestrel-hub-20251027 — “Dock 7 hub & union token gateway”
      - tu-codex-docks-20251028 — “Codex: Docks, Maintenance Decks, Salvage Permits”
    - Known limitations: translation incomplete; art renders deferred
