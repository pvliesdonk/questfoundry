---
procedure_id: clustering
description: Group related hooks by theme and type for coherent triage and advancement
version: 1.0.0
references_schemas:
  - harvest_sheet.schema.json
  - hook.schema.json
references_roles:
  - showrunner
  - lore_weaver
  - plotwright
tags:
  - hooks
  - organization
  - triage
---

# Clustering Procedure

## Overview

Group newly proposed hooks into thematic clusters and categorize by type to enable coherent batch triage and identify related narrative threads.

## Source

Extracted from v1 `spec/05-prompts/loops/hook_harvest.playbook.md` Step 3: "Cluster"

## Steps

### Step 1: Collect Proposed Hooks

Gather all hooks with status `proposed`:

- Sweep Hot SoT for new hook_card artifacts
- Include provenance links (where hook originated)
- Reject obvious duplicates (link to surviving version)

### Step 2: Identify Themes

Group hooks by narrative or worldbuilding theme:

- Story arcs (e.g., "Kestrel character arc", "Station politics")
- Setting elements (e.g., "Wormhole economy", "Dock culture")
- Factual domains (e.g., "Orbital mechanics", "Medical tech")
- Motif threads (e.g., "Isolation", "Trust")

### Step 3: Categorize by Type

Within each theme cluster, organize by hook type:

- **narrative**: Story events, character development, plot beats
- **scene**: Specific moments, dialogue, sensory details
- **factual**: Research questions, worldbuilding facts requiring validation
- **taxonomy**: Codex entries, glossary terms, player-facing definitions

### Step 4: Order for Triage

Arrange clusters for efficient review:

- High-impact structural hooks first (affect topology)
- Related hooks together (dependencies visible)
- Quick-wins grouped for batch acceptance
- Research-heavy hooks together (coordinate Researcher wake)

### Step 5: Link Dependencies

Note relationships between hooks:

- Prerequisite hooks (must accept A before B makes sense)
- Mutually exclusive hooks (conflicting approaches)
- Complementary hooks (strengthen each other)

### Step 6: Generate Cluster Headers

Create theme headings for harvest_sheet:

- Cluster name (e.g., "Wormhole Economy Expansion")
- Hook count and type breakdown
- Priority level (structural / enhancement / nice-to-have)
- Recommended triage approach for cluster

## Output

Clustered hooks organized by theme and type with dependency links and cluster headers for harvest_sheet.

## Quality Criteria

- All proposed hooks assigned to theme cluster
- Hook types clearly categorized (narrative/scene/factual/taxonomy)
- Dependencies between hooks identified
- Clusters ordered for efficient triage
- Duplicates rejected with provenance links
- Cluster headers summarize content and priority
