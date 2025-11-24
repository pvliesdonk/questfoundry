# Pre-Gate Note — Fast Feedback (Layer 2)

> **Use:** Quick, informal feedback from Gatekeeper before formal gatecheck. Identifies likely
> failures and suggests quick wins.
>
> **Producer:** Gatekeeper
> **Consumer:** Plotwright, Scene Smith, Style Lead, Lore Weaver (whoever's making the TU)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (all 8 bars)
- Role charter: `../../01-roles/charters/gatekeeper.md`
- Related: `./gatecheck_report.md` (formal validation)

---

## Purpose

Pre-gate notes are **lightweight, fast feedback** given during TU development:
- **Catch obvious failures early** (before full drafting)
- **Suggest quick wins** (simple fixes that improve bar compliance)
- **Avoid wasted effort** (don't write prose for a structurally broken topology)

Pre-gate notes are **informal** and **conversational** (not exhaustive validation).

---

## Structure

### Header

```
Pre-Gate Note — <TU or slice name>
TU: <tu-id>
Reviewer: Gatekeeper
Date: <YYYY-MM-DD>
Scope: <what was reviewed>
Status: ✅ Looks good | ⚠️ Minor concerns | 🚧 Needs work
```

---

## Quick Assessment

Brief summary of overall impression:
- What's working well
- What needs attention
- Priority fixes

---

## Bar-by-Bar Notes (Optional)

If specific bars need attention, note them:

### [Bar Name]
- **Concern:** <brief description>
- **Quick win:** <simplest fix>
- **Can defer?** yes | no

---

## Example (Clean Pre-Gate)

```markdown
Pre-Gate Note — Story Spark: Lighthouse Investigation
TU: TU-2025-11-24-PW01
Reviewer: Gatekeeper
Date: 2025-11-24
Scope: Topology notes + 3 section briefs (anchor001-003)
Status: ✅ Looks good

---

## Quick Assessment

Nice setup! Hub structure is clear, gateway is well-motivated, and choice intents look contrastive.
A few minor polish items below, but nothing blocking.

Ready for Scene to draft. Let me know when you want full gatecheck.

---

## Minor Notes

### Style
- Section brief anchor002 uses "proceed" twice (choice text); vary language for contrast
- **Quick win:** Change one to "continue" or "advance"

### Presentation
- Gateway map mentions "guild token" but doesn't explain what it looks like
- **Quick win:** Add visual description in brief (bronze pin, lighthouse + anchor symbol)

```

---

## Example (Needs Work)

```markdown
Pre-Gate Note — Story Spark: Market Hub Expansion
TU: TU-2025-11-25-PW02
Reviewer: Gatekeeper
Date: 2025-11-25
Scope: Topology notes + gateway map
Status: 🚧 Needs work before drafting

---

## Quick Assessment

Good ambition, but a few structural issues that'll bite us later. Let's fix topology before Scene
starts writing.

**Priority:** Fix Reachability and Gateway fairness first. Other concerns are polish.

---

## Critical Issues

### Reachability
- **Concern:** Keystone (anchor045) only reachable via anchor040, which is gated
- **Problem:** If player doesn't qualify for gate, keystone becomes unreachable (soft-lock)
- **Quick win:** Add alternate path to anchor045 from anchor042 (already exists nearby)
- **Can defer?** No—this blocks Cold merge

### Gateways
- **Concern:** Gate at anchor040 requires "Mayor's Trust" but no fairness path documented
- **Problem:** Player won't know how to qualify; feels arbitrary
- **Quick win:** Add scene at anchor038 where Mayor offers help if player completed earlier task
- **Can defer?** No—this blocks Presentation bar

---

## Polish Items (Can Defer to Gatecheck)

### Nonlinearity
- **Concern:** anchor042 and anchor043 both lead to anchor044 with identical outcomes
- **Not blocking, but:** Early funnel reduces meaningfulness
- **Suggestion:** Add micro-beat at anchor044 that reflects which path player took

### Style
- **Concern:** Section brief anchor041 uses "marketplace" and "market" inconsistently
- **Quick win:** Pick one term and stick with it (consult Style Lead)

```

---

## Tone & Style

Pre-gate notes should be:
- **Friendly:** "Looks great!" not "You passed"
- **Actionable:** Specific fixes, not vague concerns
- **Prioritized:** Critical vs polish
- **Fast:** Don't write an essay; bullet points are fine

---

## Hot vs Cold

**Hot only** — Pre-gate notes are working documents:
- Informal feedback during development
- Not retained after TU closes (lessons go to post-mortem if needed)
- Not exported

---

## Lifecycle

1. **Plotwright/Author** requests pre-gate feedback
2. **Gatekeeper** does fast review (15-30 min)
3. **Gatekeeper** sends pre-gate note
4. **Author** addresses critical issues
5. **Author** proceeds with drafting (or requests follow-up if needed)
6. **Full gatecheck** happens later (see `gatecheck_report.md`)

---

## When to Request Pre-Gate

Good times to request pre-gate:
- After topology/briefs drafted, before prose starts
- When unsure if structure meets bars
- After major restructure (sanity check)
- When introducing new gateway pattern

**Don't request pre-gate for:**
- Prose polish (just draft it and get full gatecheck)
- Minor tweaks (trust your judgment)
- When you're confident it's ready (go straight to gatecheck)

---

## Validation checklist

- [ ] Feedback is specific and actionable
- [ ] Critical issues are clearly marked
- [ ] Quick wins are suggested (not just "fix this")
- [ ] Tone is encouraging, not punishing
- [ ] Scope is clear (what was reviewed)

---

**Created:** 2025-11-24
**Status:** Initial template
