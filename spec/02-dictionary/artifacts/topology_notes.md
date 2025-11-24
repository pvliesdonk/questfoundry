# Topology Notes — Structure Map (Layer 2)

> **Use:** Living document of hubs, loops, gateways, keystones, and safe returns. Human-readable map
> of narrative structure. Plotwright's working canvas.
>
> **Producer:** Plotwright
> **Consumer:** Plotwright (iterative), Scene Smith, Lore Weaver, Gatekeeper

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Reachability, §Nonlinearity, §Gateways)
- Role charters: `../../01-roles/charters/plotwright.md`
- **Layer 2 references:** `../glossary.md` (Hub, Loop, Gateway, Keystone)

---

## Structure

### Header

```
Topology Notes — <project/act name>
Scope: <anchor range or act description>
Author: Plotwright
TU: <tu-id> (latest update)
Edited: <YYYY-MM-DD>
Version: <semantic version for major restructures>
```

---

## 1) Hubs

**Definition:** Sections with meaningful fan-out (3+ distinct routes)

**Format per hub:**
- **Anchor:** anchor<NNN>
- **Name:** <descriptive name>
- **Fan-out count:** <N exits>
- **Routes:** <brief description of each exit's destination/theme>
- **Convergence notes:** <which paths lead here>

---

## 2) Loops

**Definition:** Designs that return player to prior location WITH DIFFERENCE

**Format per loop:**
- **Name:** <descriptive name>
- **Entry:** anchor<NNN>
- **Return:** anchor<MMM>
- **Difference:** <what changes on return: new affordance, altered dialogue, unlocked path>
- **Trigger:** <what causes the loop>

---

## 3) Gateways

**Definition:** Conditional access points (diegetic checks)

**Format per gateway:**
- **Name:** <descriptive name>
- **Location:** anchor<NNN> → anchor<MMM>
- **Check:** <diegetic condition>
- **Qualification:** <how to qualify>
- **Details:** See gateway_map.md for full fairness analysis

---

## 4) Keystones

**Definition:** Critical sections required for story progression

**Format per keystone:**
- **Anchor:** anchor<NNN>
- **Why critical:** <narrative necessity>
- **Redundancy:** <alternative routes if this fails/skipped>
- **Risk:** <brittleness assessment: low | medium | high>

---

## 5) Safe Returns

**Definition:** Paths that allow player to backtrack without dead ends

**Format:**
- List of sections that allow return to previous hubs
- Note any one-way transitions (with justification)
- Identify potential soft-locks and mitigation

---

## Example

```markdown
Topology Notes — Adventure Bay Mystery, Act I
Scope: anchor001-anchor030 (Beach discovery through lighthouse resolution)
Author: Plotwright
TU: TU-2025-11-24-PW01
Edited: 2025-11-24
Version: 1.2.0

---

## Hubs

### Hub: Beach Landing (anchor001)
- **Fan-out:** 3 routes
- **Routes:**
  1. Follow paw prints → anchor002 (lighthouse path)
  2. Investigate rocks → anchor003 (coastal discovery)
  3. Return to lookout → anchor030 (safe retreat)
- **Convergence:** Entry point (no prior sections)

### Hub: Lighthouse Exterior (anchor005)
- **Fan-out:** 4 routes
- **Routes:**
  1. Search for key → anchor006 (foreman encounter)
  2. Climb rocks → anchor007 (bypass, oil discovery)
  3. Try door → anchor008 (maintenance worker)
  4. Return to beach → anchor001 (backtrack)
- **Convergence:** From anchor002, anchor003

---

## Loops

### Loop: Foreman's Trust
- **Entry:** anchor006 (first foreman meeting)
- **Return:** anchor015 (foreman location after lighthouse entry)
- **Difference:** Foreman now trusts player; offers Maritime Guild introduction
- **Trigger:** Player helped with initial task (key retrieval)

### Loop: Weather Check
- **Entry:** anchor005 (rock passage consideration)
- **Return:** anchor005 (after time passes)
- **Difference:** Weather cleared; rock climb now safe
- **Trigger:** Player waits or explores elsewhere first

---

## Gateways

### Gateway: Lighthouse Inner Door
- **Location:** anchor009 → anchor012
- **Check:** Maritime Guild token visible
- **Qualification:** Obtain from foreman (anchor006) or archives (anchor011)
- **Details:** See gateway_map.md (full fairness analysis)

### Gateway: Rock Passage (soft)
- **Location:** anchor005 → anchor007
- **Check:** Weather = calm OR has climbing gear
- **Qualification:** Wait for weather or get rope (anchor004)
- **Details:** Soft gate—player can return when qualified

---

## Keystones

### Keystone: Lighthouse Discovery (anchor012)
- **Why critical:** Reveals oil leak (Act I resolution trigger)
- **Redundancy:** Two paths lead here:
  1. Front door (token gate) → anchor009 → anchor012
  2. Rock climb → anchor007 → maintenance shaft → anchor012
- **Risk:** Low (multiple paths, both achievable)

### Keystone: Foreman Introduction (anchor006)
- **Why critical:** Only NPC who explains Maritime Guild significance
- **Redundancy:** Alternative: Librarian (anchor010) provides historical context
- **Risk:** Medium (skippable but affects loop quality)

---

## Safe Returns

- All hubs (anchor001, anchor005, anchor015) allow backtracking
- One-way transition: anchor007 → anchor012 (rock climb is one-direction)
  - Justified: Maintenance shaft exit only; player warned by PN before climb
- No dead ends identified; all terminals clearly marked

---

## Notes & Risks

- **Reachability:** All keystones reachable via 2+ paths ✅
- **Nonlinearity:** 3 hubs with 3+ meaningful exits ✅
- **Gateway fairness:** All gates have discoverable qualification paths ✅
- **Risk:** anchor006-007 sequence slightly rushed; consider micro-beat between

```

---

## Hot vs Cold

**Hot only** — Topology notes are living planning documents:
- Iteratively updated as structure evolves
- Consulted by Scene, Lore, Gatekeeper
- Not exported to players

---

## Lifecycle

1. **Plotwright** creates initial topology sketch
2. **Plotwright** updates iteratively as structure evolves
3. **Scene/Lore/GK** consult during their work
4. **Archived** at major milestones (retained for traceability)

---

## Validation checklist

- [ ] All hubs have 3+ meaningful exits
- [ ] All loops return WITH DIFFERENCE (new affordance/context)
- [ ] All gateways have qualification paths (see gateway_map.md)
- [ ] All keystones have redundancy or acceptable risk
- [ ] Safe returns prevent dead ends
- [ ] Cross-references to anchors are valid

---

**Created:** 2025-11-24
**Status:** Initial template
