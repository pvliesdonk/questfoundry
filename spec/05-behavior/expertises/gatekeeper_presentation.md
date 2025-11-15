# Gatekeeper Presentation Expertise

## Mission

Enforce the Presentation Quality Bar (#7) to ensure player-facing surfaces contain zero spoilers, internal terminology, or authoring artifacts.

## The Presentation Bar

**Core Principle:** Player surfaces are hermetically sealed from author-only information.

**Validates:** All player-visible text is spoiler-free and diegetic

**Scope:**

- Section prose
- Choice text
- Codex entries (player stages only)
- UI labels and prompts
- Alt text and accessibility descriptions
- Audio cue text equivalents

## Violation Categories

### 1. Spoilers

**Direct Reveals:**

- Future plot points visible too early
- Character allegiances revealed prematurely
- Outcomes telegraphed before choices
- Twists exposed in descriptive text

**Indirect Leaks:**

- Foreshadowing that's too obvious
- Choice text that implies outcomes
- Codex entries that reveal future content
- Descriptive language that assumes player knowledge

### 2. Internal Plumbing

**Forbidden Content:**

- State variable names or codewords
- Gateway conditions or logic
- Branching structure references
- Topology metadata
- Determinism parameters
- Artifact IDs or internal labels

**Examples:**

- ❌ "If loyalty_count > 3, this path unlocks"
- ❌ "This is section_brief_47"
- ❌ "Gateway: has_sword && trust_level_2"
- ✅ "Your reputation opens new possibilities"

### 3. Meta Terminology

**Out-of-Game Language:**

- Role names (Showrunner, Lore Weaver, etc.)
- System terminology (canon, trace unit, hook)
- Authoring notes or production comments
- Debug information or placeholder text
- File paths or technical references

**Examples:**

- ❌ "TODO: Add lore here"
- ❌ "See canon_pack_17 for background"
- ❌ "Gatekeeper note: Check continuity"
- ✅ [Content is fully diegetic]

### 4. Accessibility Violations

**Alt Text Issues:**

- Technical descriptions instead of sensory
- Spoilers in image alt text
- Missing or placeholder descriptions
- Non-diegetic references

## Remediation Process

### Step 1: Identify Violations

Scan all player-facing text for:

- Search for internal terminology patterns
- Flag forward references to unrevealed content
- Detect meta language and authoring artifacts
- Check alt text and captions

### Step 2: Classify Severity

**CRITICAL:** Direct spoilers or exposed state logic
**HIGH:** Meta terminology or internal references
**MEDIUM:** Accessibility gaps or unclear diegesis
**LOW:** Style issues or minor phrasing improvements

### Step 3: Provide Fixes

For each violation:

1. **Quote exact text** with file and line number
2. **Explain violation** (which bar, why it fails)
3. **Suggest rewrite** using diegetic alternatives
4. **Flag dependencies** (requires Lore Weaver, etc.)

## Collaboration Points

**With Lore Weaver:**

- Validate canon summaries for leaks
- Ensure diegetic consistency

**With Codex Curator:**

- Review unlock logic for spoiler prevention
- Verify progressive reveal staging

**With Scene Smith:**

- Check prose for forward references
- Validate choice text framing

**With Book Binder:**

- Ensure export views maintain spoiler hygiene
- Verify player surfaces remain sealed

## Quick Reference

**Always Forbidden:**

- Variable names in player text
- "If X then Y" gateway logic
- Role or system terminology
- Future content references
- Debug/TODO markers

**Always Required:**

- Diegetic framing
- In-world language
- Player-appropriate knowledge
- Spoiler-free descriptions
- Accessible alternatives
