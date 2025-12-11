# Role Abbreviations — Canonical Short Codes (Layer 2)

> **Purpose:** Define standard abbreviations for all QuestFoundry roles, used in TU briefs, hook
> cards, and trace records.

---

## All 15 Roles

| #   | Role            | Abbreviation | Active/Dormant   | Primary Artifacts            |
| --- | --------------- | ------------ | ---------------- | ---------------------------- |
| 1   | Showrunner      | SR           | Always Active    | TU Briefs, ADRs              |
| 2   | Gatekeeper      | GK           | Always Active    | Gatecheck Reports            |
| 3   | Plotwright      | PW           | Active           | Section Briefs, Gateway Maps |
| 4   | Scene Smith     | SS           | Active           | Sections (prose), Edit Notes |
| 5   | Style Lead      | ST           | Active           | Style Addenda, Register Maps |
| 6   | Lore Weaver     | LW           | Active           | Canon Packs                  |
| 7   | Codex Curator   | CC           | Active           | Codex Entries                |
| 8   | Art Director    | AD           | Optional/Dormant | Art Plans, Shotlists         |
| 9   | Illustrator     | IL           | Optional/Dormant | Art Assets                   |
| 10  | Audio Director  | AuD          | Optional/Dormant | Audio Plans, Cuelists        |
| 11  | Audio Producer  | AuP          | Optional/Dormant | Audio Assets                 |
| 12  | Translator      | TR           | Optional/Dormant | Language Packs               |
| 13  | Book Binder     | BB           | Active           | View Logs, Front Matter      |
| 14  | Player-Narrator | PN           | Active           | PN Playtest Notes            |
| 15  | Researcher      | RS           | Optional/Dormant | Research Memos               |

---

## Usage Notes

**In TU briefs:** List awake/dormant roles by abbreviation

- Example: `Awake: PW, SS, ST, LW, GK · Dormant: AD, IL, AuD, AuP, TR, RS`

**In hook cards:** Indicate proposed owner and consults

- Example: `Owner (R): ST · Consult: PW, SS, GK`

**In deferral tags:** Reference dormant role areas

- Example: `deferred:art deferred:audio deferred:translation deferred:research`

**In RACI matrices:** Use abbreviations for column headers

- Example: `| Task | SR | GK | PW | SS | ST | ... |`

---

## Abbreviation Conventions

1. **2-3 characters:** Most are 2 letters, exceptions: AuD, AuP (to distinguish from AD)
2. **Capital letters only:** No lowercase or mixed case
3. **Avoid ambiguity:**
   - ST = Style Lead (not Showrunner or Scene Smith)
   - SS = Scene Smith (not Showrunner or Style)
   - AD = Art Director (not Audio Director)
   - AuD = Audio Director, AuP = Audio Producer (to avoid collision with AD)
4. **First letter(s) of role words:**
   - Single-word roles: First 2 letters (e.g., SR = Showrunner, GK = Gatekeeper)
   - Multi-word roles: First letter of each word (e.g., PW = Plotwright, CC = Codex Curator)

---

## Role Categories

**Core Coordination (2 roles)** — Always active

- Showrunner (SR), Gatekeeper (GK)

**Primary Creative (5 roles)** — Typically active

- Plotwright (PW), Scene Smith (SS), Style Lead (ST), Lore Weaver (LW), Codex Curator (CC)

**Asset Production (5 roles)** — Optional/dormant until activated

- Art Director (AD), Illustrator (IL), Audio Director (AuD), Audio Producer (AuP), Translator (TR)

**Quality & Export (2 roles)** — Active during relevant loops

- Book Binder (BB), Player-Narrator (PN)

**Research (1 role)** — Optional/dormant until activated

- Researcher (RS)

---

## Dormancy Signals

When a role is **dormant**, use deferral tags in TU briefs and hook cards:

- `deferred:art` — AD and IL dormant
- `deferred:audio` — AuD and AuP dormant
- `deferred:translation` — TR dormant
- `deferred:research` — RS dormant

**Format:** Space-separated list (NOT pipe-separated, NOT comma-separated)

- ✅ Correct: `deferred:art deferred:audio deferred:translation`
- ❌ Wrong: `deferred:art | deferred:audio | deferred:translation`

---

## Cross-References

- **Role definitions:** `02-dictionary/glossary.md` §D (prose definitions)
- **Role charters:** `01-roles/charters/*.md` (15 roles)
- **Role workflows:** Layer 4 protocol (pending Layer 4 completion)
- **Dormancy policy:** `01-roles/interfaces/dormancy_signals.md`

---

**Source:** Extracted from LAYER1_CORRECTIONS.md Issue #7 and observed template usage during Layer
1/2 alignment (2025-10-30)
