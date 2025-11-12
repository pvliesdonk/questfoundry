# Traceability — Know What Changed, Why, and What It Touches

Traceability lets us answer, quickly and confidently: **what changed, who changed it, why it
changed, and what depends on it**. It’s the paper trail that keeps creativity from turning into
folklore.

This is a **human-level** contract. Later layers (schemas/protocol) will encode it, but you can work
with this today.

---

## 1) The Trace Unit (TU)

Every meaningful change that aims for **Cold SoT** is tracked as a **Trace Unit (TU)**. Think of it
as a slim dossier attached to the change.

**TU minimum fields (no schema, human text):**

- **TU-ID** — short, unique slug (`tu-topology-wormhole-toll-v2`)
- **Title** — human headline (“Wormhole 3 Toll — Hub Pressure Mechanic”)
- **Type** — `topology | prose | style | canon | codex | asset-plan | asset-output | factual`
- **Proposed by** — role + person/agent (“Plotwright: A. Bot”)
- **Rationale** — 1–3 sentences of why this exists (player value & design need)
- **Upstream refs** — section IDs, hooks, briefs, research memos, prior TU-IDs
- **Downstream impacts** — roles/assets likely affected (PN, Binder, Gateways plan)
- **Quality bars touched** — for Gatekeeper focus
- **Status** — `hot-proposed | stabilizing | gatecheck | cold-merged | deferred | rejected`
- **Cold snapshot** — filled on merge (`cold@2025-10-28T14:05Z` or tag/commit)
- **Notes** — spoiler flags, accessibility/localization notes, deliberate mysteries

> TU lives in **Hot** while stabilizing; on merge it records the **Cold snapshot**. TUs that get
> rejected keep their rationale for future archaeologists.

---

## 2) What must reference a TU?

- **Hook cards** that are accepted → reference the TU that integrates them.
- **Lore Deepening** outputs (canon) → reference the TU(s) they implement.
- **Codex entries** (player-safe pages) → reference the canon TU(s) they summarize.
- **Topology updates** → one TU per cohesive change (avoid mega-bundles).
- **Prose rewrites** with structural effect (choices/state) → have their own TU.
- **Art/Audio plans & outputs** → one TU per asset or logical batch.
- **Translation slices** → at least one TU per language per release window.

---

## 3) Linking rules (the minimal web)

- **Upstream refs** point to: sections, hooks, briefs, prior TUs, research memos.
- **Downstream impacts** predict what to recheck: PN lines, Binder nav, QA gate list.
- **Cross-TU links** are allowed but avoid circular dependence. If two TUs truly interlock, merge
  them or set one as “follows TU-XXXX”.

**Pro tip**: prefer _many small TUs_ over one omnibus TU. Smaller TUs stabilize faster and fail more
safely.

---

## 4) Where do TUs live?

- In **Hot**, a `TRACELOG.md` can list active TUs with one-liners.
- On **Cold merge**, the TU gets appended to a dated section in `TRACELOG.md` along with the **Cold
  snapshot ID**.
- Each changed artifact (canon page, codex page, topology note) **mentions the TU-ID** at the top or
  bottom (“Lineage: TU…”) so readers can jump to rationale.

**Suggested files:**

```

/00-north-star/TRACELOG.md         # rolling index: active & merged TUs
/DECISIONS/ADR-*.md                # big, strategic decisions (rare; separate from TU)

```

ADRs are for “change the rules of the game”; TUs are “play the game better”.

---

## 5) Showrunner & Gatekeeper responsibilities

- **Showrunner**
  - Ensures TUs exist for significant Hot work aimed at Cold.
  - Sequences TUs to avoid merge collisions and reader whiplash.
- **Gatekeeper**
  - Requires **Quality Bars** noted per TU; refuses merges lacking a TU.
  - Writes a short **gatecheck note** attached to the TU (pass/fail + remediation).

---

## 6) PN & Binder invariants

- **Binder exports** record: `Cold snapshot ID`, list of **TU-IDs** included since last export, and
  export options (include art plan, etc.).
- **PN dry-runs** cite the same `Cold snapshot ID` so playtests are reproducible.

---

## 7) Examples (miniature)

**TU**

```

TU-ID: tu-topology-wormhole-toll-v2
Title: Wormhole 3 Toll — Hub Pressure Mechanic
Type: topology
Proposed by: Plotwright (A. Bot)
Rationale: Adds recurring resource pressure and faction leverage at a major hub.
Upstream refs: hook "Shadow Toll at Wormhole 3", Faction brief v1
Downstream impacts: Scene Smith (Sections 8, 17, 33), Gateways plan, Codex(Wormhole Tolls), PN lines near hub
Quality bars touched: Nonlinearity, Gateways, Integrity
Status: gatecheck
Notes: spoiler in canon only; codex uses player-safe summary

```

**On merge**

```

Status: cold-merged
Cold snapshot: cold@2025-10-28T14:05Z (#c71a9e2)
Gatecheck: PASS (see Gatekeeper note: GK-20251028-03)

```

The **codex page** “Wormhole Tolls” then adds: `Lineage: TU tu-topology-wormhole-toll-v2`.

---

## 8) Common failure modes & fixes

- **“Drive-by” changes without TUs** → Require a TU before stabilization.
- **Omnibus TU with unrelated changes** → Split into smaller, coherent TUs.
- **Missing downstream impacts** → Gatekeeper kicks back with a checklist (“PN wording, Binder nav,
  codex cross-refs”).
- **Lost rationale over time** → `TRACELOG.md` is the index; keep one-liners crisp.

---

## 9) Privacy & spoilers

- TU rationales may contain spoilers or internal reasoning. That’s fine in **Hot**.
- On **Cold**, link the TU but keep spoiler content in canon notes, not on player surfaces.

---

## 10) When Researcher is dormant

- TUs with **factual** claims must mark `uncorroborated:<risk>`.
- Binder/PN wording should avoid hard claims if risk is `med/high`.
- Create a follow-up TU or TODO to revisit once Researcher activates.

---

**TL;DR**  
Tag each meaningful change with a small **Trace Unit**. Note who/why/what-it-touches. Gate it, merge
it, and stamp the Cold snapshot. Future you (and every collaborating agent) will thank you.
