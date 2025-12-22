---
title: Scope and Length Guidelines for Interactive Fiction
summary: Sizing IF projects with word counts, section metrics, branching depth, and scope-quality balance guidance.
topics:
  - project-scope
  - word-counts
  - section-metrics
  - branching-depth
  - play-time
  - multiple-endings
  - scope-quality-balance
cluster: scope-and-planning
---

# Scope and Length Guidelines for Interactive Fiction

Craft guidance for sizing interactive fiction projects—word counts, section metrics, branching depth, and balancing scope with quality.

---

## Project Scale Categories

### Overview

Interactive fiction varies enormously in scope. A short Twine experiment might take five minutes to play; an epic Choice of Games title might take twenty hours. Understanding scale helps set realistic expectations.

| Scale | Total Words | Play Time | Sections | Endings |
|-------|-------------|-----------|----------|---------|
| Micro | 1,000–5,000 | 5–15 min | 10–30 | 2–4 |
| Short | 5,000–20,000 | 15–60 min | 30–100 | 3–8 |
| Medium | 20,000–60,000 | 1–3 hours | 100–300 | 5–15 |
| Long | 60,000–150,000 | 3–8 hours | 300–700 | 8–25 |
| Epic | 150,000–500,000+ | 8–20+ hours | 700–2000+ | 15–50+ |

### Micro (1K–5K Words)

Ideal for: Experiments, jam games, vignettes, proof of concept.

- Single session experience
- Simple branching (2–3 major paths)
- Limited character development
- One central question or dilemma
- Often linear with minor variations

### Short (5K–20K Words)

Ideal for: Short stories, focused experiences, commercial demos.

- Complete narrative arc possible
- Character development achievable
- Meaningful choices with visible consequences
- Typical for Twine jam games, early chapters
- Manageable testing burden

### Medium (20K–60K Words)

Ideal for: Novellas, substantial games, commercial releases.

- Multiple character arcs
- Complex branching with convergence points
- State tracking becomes important
- Significant testing required
- Sweet spot for many commercial IF titles

### Long (60K–150K Words)

Ideal for: Full novels, premium commercial titles.

- Multiple storylines possible
- Deep character customization
- Extensive world exploration
- Substantial development time (6–18 months typical)
- Requires systematic testing methodology

### Epic (150K+ Words)

Ideal for: Flagship commercial titles, series entries.

- Comparable to novel series in scope
- Team development often necessary
- Years of development time
- Complex state management required
- Choice of Games hosted games often fall here

---

## Section/Passage Metrics

### Words Per Section

Different platforms have different norms:

| Platform Style | Words per Section | Notes |
|----------------|-------------------|-------|
| Twine (short) | 50–200 | Quick, punchy passages |
| Twine (literary) | 200–500 | More prose-focused |
| Choice of Games | 300–800 | Longer narrative chunks |
| Visual Novel | 50–150 | With accompanying art |
| Parser IF | Variable | Room descriptions + responses |

### Paragraphs Per Section

- **Minimum:** 1–2 paragraphs (for action/tension moments)
- **Typical:** 3–5 paragraphs (balanced reading)
- **Maximum:** 6–8 paragraphs (for key scenes, avoid longer)

Longer passages risk reader fatigue. Break at natural pause points.

### The Screen Rule

A section should fit comfortably on one screen without scrolling. If readers must scroll extensively, consider splitting the section.

### Sentence Count Guidance

- **3–4 sentences per paragraph** for readability
- **Vary paragraph length** for rhythm
- **Short paragraphs for tension**, longer for description
- **One idea per paragraph** as a guideline

---

## Choice Frequency

### How Often to Offer Choices

Readers expect agency. Long stretches without choices feel like reading, not playing.

| Style | Words Between Choices | Notes |
|-------|----------------------|-------|
| High agency | 100–300 | Frequent decisions |
| Balanced | 300–600 | Standard pacing |
| Narrative-heavy | 600–1000 | Story-focused |
| Kinetic sections | 1000+ | Deliberate no-choice zones |

### Choices Per Section

- **Typical:** 2–4 choices per decision point
- **Complex moments:** Up to 5–6 choices acceptable
- **Simple moments:** 2 choices (binary) is fine
- **Avoid:** More than 6 choices (overwhelming)

### Choice Fatigue

Too many choices exhaust readers. Not every moment needs a decision. Use choiceless sections for:

- Emotional beats that shouldn't be interrupted
- Consequences playing out from previous choices
- Establishing scenes and atmosphere
- Building to a significant choice

---

## Branching Depth and Structure

### Branching Models

**Pure Branching (Tree):**
Every choice creates a new path. Words multiply exponentially.

- 3 choices × 3 choices × 3 choices = 27 paths
- Extremely content-hungry
- Only viable for micro-scale projects

**Bottleneck (Funnel):**
Paths diverge then reconverge at key story beats.

- Most sustainable model
- Variations in journey, convergent destinations
- Scales to any project size

**Parallel Tracks:**
Major early choices set you on distinct tracks with limited crossover.

- 2–4 parallel storylines
- Moderate content multiplication
- Good for medium-to-long projects

**State-Based:**
Single path with variations based on accumulated choices/stats.

- Most efficient use of words
- Personalization through variables
- Scales excellently

### Convergence Points

Plan where paths merge:

- **Frequent convergence:** Every 3–5 sections
- **Moderate convergence:** Every chapter/act
- **Rare convergence:** Only at major story beats

More convergence = more manageable scope. Less convergence = more unique content.

### Branch Depth Limits

How many choices deep before convergence?

| Project Scale | Recommended Max Depth |
|---------------|----------------------|
| Micro | 2–3 |
| Short | 3–4 |
| Medium | 4–6 |
| Long | 5–8 |
| Epic | 6–10 |

Beyond these depths, testing becomes unwieldy and readers rarely see all content.

---

## Endings

### How Many Endings?

More endings isn't always better. Each ending must feel earned and distinct.

| Project Scale | Recommended Endings |
|---------------|---------------------|
| Micro | 2–4 |
| Short | 3–8 |
| Medium | 5–15 |
| Long | 8–25 |
| Epic | 15–50 |

### Ending Types

- **Major endings:** Fundamentally different outcomes (3–6 typical)
- **Variations:** Same basic outcome, different details (many possible)
- **Early endings:** Death/failure states (use sparingly)
- **Secret endings:** Hidden paths for dedicated players (1–3)

### The 80/20 Rule

Roughly 80% of readers will see 20% of your endings. Design your most polished content for the most likely paths.

---

## Quality vs Quantity Trade-offs

### The Scope Triangle

You can optimize for two of three:

- **Breadth:** Many branches, choices, paths
- **Depth:** Polished prose, complex characters
- **Speed:** Quick development timeline

Trying for all three leads to burnout or poor quality.

### Content Efficiency Strategies

**Delayed Branching:**
Keep early chapters more linear. Branch later when readers are invested.

**Variable Text:**
Small substitutions based on state rather than whole new passages.

```
You approach the {guard_attitude} guard.
// friendly, nervous, or hostile based on prior choices
```

**Shared Scenes:**
Multiple paths can share key scenes with minor variations.

**Meaningful Dead Ends:**
Some branches can end earlier (not in death, but in resolution) to reduce total content needs.

### Testing Burden

Every branch multiplies testing time:

| Branches | Testing Multiplier | Notes |
|----------|-------------------|-------|
| 2 paths | 2× | Manageable |
| 4 paths | 4× | Significant |
| 8 paths | 8× | Major effort |
| 16+ paths | 16×+ | Team needed |

Factor testing time into scope decisions. Untested branches will have bugs.

---

## Development Time Estimates

### Solo Developer Benchmarks

Rough estimates for experienced IF writers:

| Scale | First Draft | Revision | Testing | Total |
|-------|-------------|----------|---------|-------|
| Micro | 1–3 days | 1–2 days | 1 day | 1 week |
| Short | 1–2 weeks | 1 week | 3–5 days | 1 month |
| Medium | 1–3 months | 2–4 weeks | 2–4 weeks | 3–6 months |
| Long | 3–6 months | 1–2 months | 1–2 months | 6–12 months |
| Epic | 6–18 months | 2–6 months | 2–4 months | 1–3 years |

These assume full-time focus. Part-time work extends timelines significantly.

### Scope Creep Warning Signs

- Adding "just one more branch"
- Expanding backstory into playable content
- Feature additions mid-development
- Perfectionism on low-traffic paths

Set scope early. Stick to it. Save expansions for sequels.

---

## Common Mistakes

### Overscoping

The most common IF project killer. Start smaller than you think you need. You can always expand a completed small project.

### Underestimating Branching Costs

A "simple" early choice that creates two paths doubles everything that follows. Map your structure before writing.

### Neglecting Popular Paths

Spending equal time on all paths means the path most players take gets the same attention as paths 5% will see.

### Inconsistent Section Length

Wildly varying section lengths disrupt reading rhythm. Establish norms and stick to them.

### No Convergence Plan

Branches that never rejoin create exponential content growth. Plan merge points from the start.

---

## Quick Reference

| Decision | Recommendation |
|----------|----------------|
| First project | Micro or Short scale |
| Solo developer | Cap at Medium scale |
| Choice frequency | Every 300–600 words |
| Choices per decision | 2–4 typically |
| Section length | 200–500 words |
| Branching model | Bottleneck for sustainability |
| Testing buffer | 20–30% of development time |
| Endings | 3–8 for most projects |
