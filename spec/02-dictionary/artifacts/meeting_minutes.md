# Meeting Minutes — Customer Interface Record (Layer 2)

> **Use:** Structured record of Showrunner ↔ Customer interface sessions. Captures directives,
> clarifying questions, and extracted action items from "vibes" conversations.
>
> **Producer:** Showrunner
> **Consumer:** Showrunner (planning), Gatekeeper (context for merge decisions)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md`
- Role charter: `../../01-roles/charters/showrunner.md`
- Loop: `../../00-north-star/LOOPS/customer_interface.md` (if formalized)

---

## Purpose

Meeting minutes formalize the **Customer Interface** loop output. They transform unstructured
conversations ("I want more mystery at the lighthouse") into actionable studio work.

Key functions:
- **Capture intent** — preserve Customer's vision and priorities
- **Extract directives** — translate requests into TU candidates
- **Record clarifications** — document what was asked, what was answered
- **Trace decisions** — link downstream TUs back to Customer requests

---

## Structure

### Header

```
Meeting Minutes — <session name or date>
Date: <YYYY-MM-DD>
Attendees: <Customer name/role>, Showrunner
Duration: <N minutes>
Context: <project phase, milestone, or trigger>
```

---

## 1) Raw Summary

**Purpose:** Capture the essence of the conversation in free-form prose.

**Format:**
- 1-3 paragraphs summarizing the discussion
- Customer's tone and priorities ("excited about X, concerned about Y")
- Key themes or recurring topics

**Example:**
```markdown
Customer expressed strong interest in deepening the lighthouse mystery. They feel the foreman's
backstory is "flat" and want more emotional weight. Also raised concerns about pacing—Act I feels
rushed. Suggested adding a "breathing room" section before the climax. Very enthusiastic about
the Maritime Guild concept; wants to expand it into Act II.
```

---

## 2) Clarifying Questions

**Purpose:** Record what the Showrunner asked to resolve ambiguity.

**Format:** List of questions with Customer's responses.

**Example:**
```markdown
- **Q:** Should the foreman's backstory be revealed in Act I or Act II?
  - **A:** Act I, but subtly—player should suspect, not know.

- **Q:** By "breathing room," do you mean narrative pacing or more exploration choices?
  - **A:** Both—add a low-stakes section where player can wander and absorb setting.

- **Q:** Maritime Guild expansion—new characters or deeper lore for existing ones?
  - **A:** Deeper lore for existing. Introduce Guild hierarchy but keep cast small.
```

---

## 3) Directives Extracted

**Purpose:** Translate Customer's requests into concrete, actionable studio directives.

**Format:** List of directives with owner, urgency, and scope.

**Example:**
```markdown
1. **Deepen foreman backstory (Act I)**
   - Owner: Lore Weaver (canon), Scene Smith (prose)
   - Loop: Lore Deepening → Story Spark
   - Urgency: High (blocks Act I lock)
   - Scope: 1 canon pack, 1-2 section revisions

2. **Add "breathing room" section before climax**
   - Owner: Plotwright (topology), Scene Smith (prose)
   - Loop: Story Spark
   - Urgency: Medium (improves pacing, not blocking)
   - Scope: 1 new hub + 2-3 sections

3. **Expand Maritime Guild lore (existing characters)**
   - Owner: Lore Weaver (canon), Codex Curator (player-safe entries)
   - Loop: Lore Deepening → Codex Expansion
   - Urgency: Low (Act II prep)
   - Scope: 1 canon pack, 3-5 codex entries
```

---

## 4) Follow-Up Actions

**Purpose:** Next steps for the Showrunner.

**Format:** Checklist of immediate actions.

**Example:**
```markdown
- [ ] File 3 hooks from directives (foreman backstory, breathing room hub, Guild lore)
- [ ] Schedule Hook Harvest TU for this week
- [ ] Wake Lore Weaver for foreman backstory (high urgency)
- [ ] Confirm Act I lock date with Gatekeeper (after breathing room section added)
- [ ] Propose Act II timeline to Customer at next session
```

---

## 5) Open Issues

**Purpose:** Questions or concerns that remain unresolved.

**Format:** List of open items with owners and target resolution date.

**Example:**
```markdown
- **Tone for Maritime Guild lore** — Customer wants "grounded," but does that mean historical
  realism or diegetic plausibility? (Owner: Showrunner, next session)
- **Act II scope** — How much expansion is acceptable before we risk feature creep? (Owner:
  Showrunner + Gatekeeper, policy huddle)
```

---

## Example

```markdown
Meeting Minutes — Lighthouse Mystery Expansion
Date: 2025-11-24
Attendees: Alice Chen (Customer), Showrunner
Duration: 45 minutes
Context: Post-Act I playtest feedback session

---

## 1) Raw Summary

Customer expressed strong interest in deepening the lighthouse mystery. They feel the foreman's
backstory is "flat" and want more emotional weight. Also raised concerns about pacing—Act I feels
rushed. Suggested adding a "breathing room" section before the climax. Very enthusiastic about
the Maritime Guild concept; wants to expand it into Act II.

Tone: Excited but slightly anxious about timeline. Willing to delay Act I lock if quality improves.

---

## 2) Clarifying Questions

- **Q:** Should the foreman's backstory be revealed in Act I or Act II?
  - **A:** Act I, but subtly—player should suspect, not know.

- **Q:** By "breathing room," do you mean narrative pacing or more exploration choices?
  - **A:** Both—add a low-stakes section where player can wander and absorb setting.

- **Q:** Maritime Guild expansion—new characters or deeper lore for existing ones?
  - **A:** Deeper lore for existing. Introduce Guild hierarchy but keep cast small.

---

## 3) Directives Extracted

1. **Deepen foreman backstory (Act I)**
   - Owner: Lore Weaver (canon), Scene Smith (prose)
   - Loop: Lore Deepening → Story Spark
   - Urgency: High (blocks Act I lock)
   - Scope: 1 canon pack, 1-2 section revisions

2. **Add "breathing room" section before climax**
   - Owner: Plotwright (topology), Scene Smith (prose)
   - Loop: Story Spark
   - Urgency: Medium (improves pacing, not blocking)
   - Scope: 1 new hub + 2-3 sections

3. **Expand Maritime Guild lore (existing characters)**
   - Owner: Lore Weaver (canon), Codex Curator (player-safe entries)
   - Loop: Lore Deepening → Codex Expansion
   - Urgency: Low (Act II prep)
   - Scope: 1 canon pack, 3-5 codex entries

---

## 4) Follow-Up Actions

- [ ] File 3 hooks from directives (foreman backstory, breathing room hub, Guild lore)
- [ ] Schedule Hook Harvest TU for this week
- [ ] Wake Lore Weaver for foreman backstory (high urgency)
- [ ] Confirm Act I lock date with Gatekeeper (after breathing room section added)
- [ ] Propose Act II timeline to Customer at next session

---

## 5) Open Issues

- **Tone for Maritime Guild lore** — Customer wants "grounded," but does that mean historical
  realism or diegetic plausibility? (Owner: Showrunner, next session)
- **Act II scope** — How much expansion is acceptable before we risk feature creep? (Owner:
  Showrunner + Gatekeeper, policy huddle)

```

---

## Hot vs Cold

**Hot only** — Meeting minutes are internal planning documents:
- Not exported to players
- Contain Customer-specific details
- May reference spoilers or unfinished work

---

## Lifecycle

1. **Showrunner** conducts Customer session
2. **Showrunner** drafts minutes within 24 hours (capture while fresh)
3. **Showrunner** extracts directives and files hooks
4. **Showrunner** reviews with Customer at next session (optional)
5. **Archived** with TU references for traceability

---

## Validation checklist

- [ ] Session date, attendees, and context recorded
- [ ] Raw summary captures Customer's tone and priorities
- [ ] Clarifying questions include both Q and A
- [ ] Directives have owner, loop, urgency, and scope
- [ ] Follow-up actions are specific and actionable
- [ ] Open issues have owners and target resolution dates

---

**Created:** 2025-11-25
**Status:** Initial template
