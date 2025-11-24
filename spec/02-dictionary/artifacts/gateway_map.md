# Gateway Map — Diegetic Checks & Fairness (Layer 2)

> **Use:** Documents all gateways (conditional access points) in a slice, with diegetic checks and
> fairness paths. Ensures gates are player-understandable and reachable.
>
> **Producer:** Plotwright
> **Consumer:** Scene Smith (phrasing), Gatekeeper (reachability validation), PN (in-world enforcement)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Gateways, §Reachability)
- Role charters: `../../01-roles/charters/plotwright.md` · `../../01-roles/charters/gatekeeper.md`
- **Layer 2 references:** `../../00-north-star/PN_PRINCIPLES.md` · `../glossary.md` (Gateway, Codeword)

---

## Structure

### Header

```
Gateway Map — <slice name>
Scope: <hub/loop/act description>
Author: Plotwright
TU: <tu-id>
Edited: <YYYY-MM-DD>
```

---

## Gateway Entries

For each gateway:

### Gateway: <descriptive name>

**Location:** anchor<NNN> → anchor<MMM>
**Diegetic check:** <what the world checks>
**Qualification paths:** <how player can qualify>
**Failure mode:** <what happens if unqualified>
**Fairness notes:** <how player discovers requirements>

---

## Example

```markdown
Gateway Map — Act I: Lighthouse Investigation
Scope: Anchors 001-015 (Beach to Lighthouse interior)
Author: Plotwright
TU: TU-2025-11-24-PW03
Edited: 2025-11-24

---

### Gateway: Lighthouse Inner Door

**Location:** anchor009 → anchor012 (Inner sanctum)
**Diegetic check:** Player must have **Maritime Guild token** visible
**Qualification paths:**
1. Obtain token from Foreman (anchor006-007 sequence)
2. Find historical token in Mayor's archive (anchor011, requires prior conversation)

**Failure mode:** Guard politely turns player away with hint: "Guild members only beyond this point"
**Fairness notes:**
- Token mentioned by Mayor in anchor002 (foreshadowing)
- Foreman dialogue in anchor006 explains token significance
- Alternative path (archives) signposted if player talks to librarian (anchor010)

---

### Gateway: Rock Passage (Conditional Difficulty)

**Location:** anchor005 → anchor007
**Diegetic check:** Weather state = calm OR player has climbing gear
**Qualification paths:**
1. Wait for weather to clear (PN checks time-of-day state)
2. Acquire rope from maintenance shed (anchor004)

**Failure mode:** PN describes waves as too dangerous; suggests waiting or finding equipment
**Fairness notes:**
- Skye warns about weather in anchor005 dialogue
- Maintenance shed visible from anchor003 (player can backtrack)
- No permanent lock—player can return when qualified

```

---

## Requirements

### Diegetic language

- **Use:** Badge, token, knowledge, reputation, tool, permission
- **Avoid:** Codewords, flag names, internal labels, "LOCKED", mechanics

### Fairness

- Every gateway must have at least **one discoverable qualification path**
- Foreshadowing via dialogue, codex, or environmental cues
- Failure messaging should hint at requirements without spoiling

### Player-safety

- No spoilers in gateway descriptions
- Hot implementation notes (codeword mappings) stay off-surface

---

## Hot vs Cold

**Hot only** — Gateway maps are planning documents:
- Used by Scene Smith for diegetic phrasing
- Validated by Gatekeeper for reachability
- Referenced by PN for in-world enforcement
- Not exported to players

---

## Lifecycle

1. **Plotwright** creates gateway map during topology design
2. **Scene Smith** uses for diegetic phrasing in section prose
3. **Gatekeeper** validates reachability paths during pre-gate
4. **PN** enforces gates in-world during performance
5. **Archived** after Cold merge (retained for traceability)

---

## Validation checklist

- [ ] All gates have diegetic checks (no internal labels)
- [ ] Every gate has at least one qualification path
- [ ] Fairness notes show how player discovers requirements
- [ ] Failure modes are helpful, not punishing
- [ ] Cross-references to section anchors are valid
- [ ] No spoilers in descriptions

---

**Created:** 2025-11-24
**Status:** Initial template
