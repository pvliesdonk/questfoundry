# Factual Appendix — Citations and Uncertainty Log (Layer 2)

> **Use:** Citations for corroborated factual claims in canon, or uncertainty level tracking for
> uncorroborated claims. Supports research posture transparency and downstream neutral phrasing.
>
> **Producer:** Lore Weaver (with Researcher input if active)
> **Consumer:** Researcher (corroboration), Scene Smith (neutral phrasing), PN (performance guidance)

---

## Normative references

- Research posture: `../../00-north-star/SOURCES_OF_TRUTH.md` (§Research Posture Levels)
- Loop: `../../00-north-star/LOOPS/lore_deepening.md`
- Role charters: `../../01-roles/charters/lore_weaver.md` · `../../01-roles/charters/researcher.md`
- Related: `./canon_pack.md` (source of factual claims), `./research_memo.md` (detailed research)

---

## Purpose

Factual appendices document the **research posture** of canon claims:
- **Citations** for corroborated facts (sources, evidence, verification)
- **Uncertainty levels** for uncorroborated claims (low/medium/high risk)
- **Neutral phrasing guidance** for uncorroborated claims (how to avoid asserting as fact)
- **Revisit triggers** (when to schedule Researcher pass)

These appendices keep canon **traceable** and prevent factual claims from appearing authoritative
when they're actually uncertain.

---

## Structure

### Header

```
Factual Appendix — <canon topic or TU>
Canon Source: <canon_pack TU-ID or reference>
Lore Weaver: <name or agent>
Researcher: <name or agent | dormant>
Date: <YYYY-MM-DD>
Overall Posture: <corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high>
```

---

## 1) Corroborated Claims

Factual claims with verified sources and citations:

**Format per claim:**
- **Claim:** <factual statement from canon>
- **Sources:** <list of citations: books, articles, datasets, expert interviews, etc.>
- **Verification method:** <how claim was corroborated: primary source | expert review | cross-reference>
- **Confidence:** <high | medium>
- **Notes:** <any caveats, limitations, or contextual details>

---

## 2) Plausible Claims

Factual claims that are reasonable extrapolations or accepted industry assumptions:

**Format per claim:**
- **Claim:** <factual statement from canon>
- **Basis:** <why this is plausible: analogy | expert opinion | common practice>
- **Not verified by:** <what would be needed for full corroboration>
- **Risk level:** <low | medium>
- **Notes:** <any assumptions or context>

---

## 3) Disputed Claims

Factual claims with conflicting sources or expert disagreement:

**Format per claim:**
- **Claim:** <factual statement from canon>
- **Position A:** <one interpretation or source>
- **Position B:** <conflicting interpretation or source>
- **Canon choice:** <which position canon adopts, or if canon defers>
- **Rationale:** <why canon chose this position (narrative fit | best evidence | pragmatic)>
- **Risk level:** <medium | high>
- **Notes:** <how to acknowledge uncertainty in player-safe text if needed>

---

## 4) Uncorroborated Claims

Factual claims **not yet verified** by Researcher (or Researcher dormant):

**Format per claim:**
- **Claim:** <factual statement from canon>
- **Uncertainty level:** <uncorroborated:low | uncorroborated:medium | uncorroborated:high>
- **Why uncorroborated:** <Researcher dormant | time constraints | sources unavailable>
- **Neutral phrasing:** <how Scene Smith/PN should phrase this without asserting as fact>
- **Revisit trigger:** <when to schedule Researcher pass: before Act II | before export | optional>
- **Risk if wrong:** <narrative impact if claim is false: low | medium | high>

**Uncertainty level guidance:**
- **uncorroborated:low** — Claim is very likely true; low risk if unverified
- **uncorroborated:medium** — Claim is uncertain; moderate risk if false
- **uncorroborated:high** — Claim is speculative; high risk if false; needs verification ASAP

---

## 5) Research Gaps

Areas where canon needs factual input but doesn't have it yet:

**Format per gap:**
- **Topic:** <what needs research>
- **Canon impact:** <why this matters to the story>
- **Hook filed:** <hook-id if filed for Researcher>
- **Workaround:** <how canon proceeds without this info (defer | neutral phrasing | fictional substitute)>

---

## Example

```markdown
Factual Appendix — Maritime Guild Governance
Canon Source: Canon Pack TU-2025-11-24-LW01 (Guild Structure)
Lore Weaver: Claude Agent
Researcher: dormant
Date: 2025-11-24
Overall Posture: uncorroborated:medium

---

## Corroborated Claims

### Claim: Maritime guilds historically used lapel badges for visual identification
- **Sources:**
  - "Guild Systems in Medieval Trade" (Smith, 2018), Chapter 7
  - "Visual Identity in Labor Organizations" (Jones, 2020), pp. 45-52
- **Verification method:** Cross-reference of historical guild practices in primary sources
- **Confidence:** High
- **Notes:** Badges varied by region (pins, sashes, embroidered symbols); choice of bronze pins for this setting is plausible but not historically mandated

---

## Plausible Claims

### Claim: Plasma backflow from valve failure can cause facial burns
- **Basis:** Industrial safety literature documents plasma arc flash injuries; valve failures are common cause
- **Not verified by:** Specific medical case studies or expert review
- **Risk level:** Low (industry knowledge, not narrative-critical)
- **Notes:** Exact valve model is fictional; "plasma backflow" is generic enough to be safe

---

## Disputed Claims

_(None for this canon pack)_

---

## Uncorroborated Claims

### Claim: Bronze is traditionally used for guild emblems due to corrosion resistance
- **Uncertainty level:** uncorroborated:medium
- **Why uncorroborated:** Researcher dormant; time constraints for export
- **Neutral phrasing:** "Bronze pins—common for dock work, sturdy against salt air." (implies utility, not historical fact)
- **Revisit trigger:** Optional (before Act II expansion if guild lore deepens)
- **Risk if wrong:** Low (narrative choice, not factual assertion; player won't fact-check)

### Claim: Guild founding dates to station construction (Y-50)
- **Uncertainty level:** uncorroborated:high
- **Why uncorroborated:** Researcher dormant; fictional timeline has no external verification
- **Neutral phrasing:** "Older pups say the guild's been here since the station's early days." (in-world hearsay, not authorial assertion)
- **Revisit trigger:** Before Act II (if guild history becomes plot-critical)
- **Risk if wrong:** Medium (timeline inconsistencies could break immersion)

---

## Research Gaps

### Topic: Historical maritime safety regulations post-industrial accidents
- **Canon impact:** Establishes precedent for foreman's strict inspection behavior
- **Hook filed:** HK-20251124-08 (to Researcher)
- **Workaround:** Canon uses generic "safety drills tightened after incidents" phrasing; defers specific regulations to optional future research

### Topic: Union badge technology (RFID vs visual-only)
- **Canon impact:** Determines whether foreman uses scanner or visual check
- **Hook filed:** None (resolved narratively as visual-only for diegetic gate clarity)
- **Workaround:** Canon chooses visual badge check for gameplay clarity; tech detail deferred
```

---

## Hot vs Cold

**Hot only** — Factual appendices are working documents:
- Track research posture transparently
- Guide neutral phrasing for uncorroborated claims
- Not exported to players
- May reference specific citations (keep player-safe)

**Cold impact:**
- Uncorroborated claims use neutral phrasing in player-safe text
- Citations may appear in optional author's notes (if project includes them)

---

## Lifecycle

1. **Lore Weaver** identifies factual claims in Canon Pack during Lore Deepening
2. **Lore Weaver** creates Factual Appendix documenting research posture
3. **Researcher** (if active) corroborates claims and provides citations
4. **Lore Weaver** updates appendix with citations or uncertainty levels
5. **Scene Smith** uses neutral phrasing guidance for uncorroborated claims
6. **PN** uses neutral phrasing in performance
7. **Showrunner** schedules Researcher pass if uncorroborated:high claims need verification

---

## Validation checklist

- [ ] All factual claims from canon_pack categorized (corroborated | plausible | disputed | uncorroborated)
- [ ] Citations provided for corroborated claims
- [ ] Uncertainty levels assigned for uncorroborated claims (low | medium | high)
- [ ] Neutral phrasing provided for uncorroborated claims
- [ ] Revisit triggers specified for uncorroborated:medium and uncorroborated:high claims
- [ ] Research gaps documented with hooks or workarounds
- [ ] Overall posture reflects highest-risk claim in appendix

---

**Created:** 2025-11-24
**Status:** Initial template
