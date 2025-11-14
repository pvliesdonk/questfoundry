---
snippet_id: presentation_normalization
name: Presentation Normalization
description: Choices render as bullets with entire line as link; altered-hub returns need two diegetic cues; keystone exits need breadcrumbs
applies_to_roles: [gatekeeper, book_binder, plotwright]
quality_bars: [presentation, accessibility]
---

# Presentation Normalization

## Core Principle

Enforce consistent, accessible presentation patterns across all player-facing surfaces.

## Choice Formatting

### Required Pattern

Choices MUST render as:

- Bulleted list
- Entire line is clickable link
- No mixed formats (partial links, inline links forbidden)

### Valid Choice Formatting

```markdown
✓ - [Slip through maintenance](#section_12)
✓ - [Face the foreman](#section_13)
✓ - [Return to cargo bay](#section_01)
```

Renders as:

- 🔗 Slip through maintenance (entire line clickable)
- 🔗 Face the foreman (entire line clickable)
- 🔗 Return to cargo bay (entire line clickable)

### Invalid Choice Formatting

```markdown
❌ You could [slip through maintenance](#section_12) or [face the foreman](#section_13).
❌ - Slip through [maintenance](#section_12)
❌ Click [here](#section_12) to continue
```

These fail because:

- Inline links break screen reader navigation
- Partial-line links ambiguous for assistive tech
- "Click here" non-descriptive

### Gatekeeper Validation

Block if:

- Choices not in bulleted list
- Links don't span entire line
- Inline narrative + links mixed
- Non-descriptive link text ("here", "this")

## Altered-Hub Returns

### What is Altered-Hub?

Hub section where player returns after something changed:

- New knowledge acquired
- Object obtained
- Relationship shifted
- Environmental change

### Two Diegetic Cues Required

When player returns to altered hub, provide TWO cues that something changed:

**Example: Cargo Bay (Returned After Acquiring Hex-Key)**

```markdown
Cue 1 (Environmental): "The panel you couldn't open before sits accessible now."
Cue 2 (Object Reference): "Your hex-key might finally crack that locked storage unit."

Choices:
- Open storage unit with hex-key
- Continue to maintenance
- Return to airlock
```

**Why Two Cues:**

- First cue: player notices change
- Second cue: player understands affordance
- Prevents player missing new option

### Gatekeeper Validation

For altered-hub returns:

- [ ] At least two diegetic cues present
- [ ] Cues reference the change (not generic)
- [ ] New affordance clearly signaled
- [ ] Cues in-world (not meta: "You now have hex-key")

**Block if:**

- Hub altered but only one cue (or zero)
- Cues too subtle (player likely to miss)
- Cues meta ("New option unlocked")

## Keystone Exits

### What is Keystone?

Bottleneck section where multiple paths converge or diverge.

### Breadcrumb Requirement

At keystone exits, provide at least ONE outbound breadcrumb/affordance:

**Example: Engineering Hub (Keystone)**

```markdown
The engineering bay splits three ways. The airlock passage glows with safety lights, the reactor corridor thrums with a low vibration, and the crew quarters sit silent beyond the far hatch.

Choices:
- Take airlock passage
- Head to reactor corridor
- Enter crew quarters
```

Breadcrumbs:

1. "glows with safety lights" → airlock
2. "thrums with low vibration" → reactor
3. "sit silent" → crew quarters

Each choice has environmental cue.

### Why Breadcrumbs Matter

- Player hasn't been to keystone branches yet
- Generic labels ("Go north", "Go south") not helpful
- Environmental cues help player make informed choice
- Avoid blind guessing

### Gatekeeper Validation

For keystone exits:

- [ ] Each outbound choice has environmental cue
- [ ] Cues differentiate options (not all generic)
- [ ] Cues in-world and sensory
- [ ] At least one breadcrumb per exit

**Block if:**

- Keystone exits lack environmental cues
- All exits described identically
- Cues missing or too vague
- Player forced to guess blindly

## Plotwright Design Support

When designing topology:

**Mark Altered Hubs:**

```yaml
section_id: cargo_bay_01
altered_on_return_if:
  - condition: player_acquired_hex_key
    cues:
      - "Panel now accessible"
      - "Hex-key matches lock type"
```

**Mark Keystones:**

```yaml
section_id: engineering_hub
keystone: true
outbound_breadcrumbs:
  - choice: airlock_passage
    cue: "glows with safety lights"
  - choice: reactor_corridor
    cue: "thrums with low vibration"
  - choice: crew_quarters
    cue: "silent beyond far hatch"
```

## Book Binder Export

During view assembly:

- Validate choice formatting (bullet lists, full-line links)
- Check altered-hub sections for two-cue requirement
- Verify keystone exits have breadcrumbs
- Report violations to Gatekeeper before export

## Common Violations

### Inline Choice Links

```markdown
❌ "You could slip through maintenance or face the foreman."
     (with inline links)
```

Fix: Convert to bulleted list with full-line links

### Single Cue on Altered Hub

```markdown
❌ Return to cargo bay (altered):
   "The locked panel sits before you."
   
   Choices:
   - Open panel with hex-key
   - Leave
```

Fix: Add second cue: "Your hex-key should fit the lock"

### Keystone Without Breadcrumbs

```markdown
❌ Engineering bay splits three ways.
   
   Choices:
   - Go left
   - Go straight
   - Go right
```

Fix: Add environmental cues for each direction

### Meta Cues

```markdown
❌ "New option unlocked: Open panel"
```

Fix: Use diegetic cues: "The panel you couldn't open before sits accessible now"

## Accessibility Connection

These patterns support accessibility:

- **Bulleted lists:** Screen readers navigate by list structure
- **Full-line links:** Clear targets for assistive tech
- **Two cues:** Redundancy helps players with attention/memory differences
- **Breadcrumbs:** Reduces cognitive load at decision points
- **Diegetic cues:** Avoid relying on visual-only indicators
