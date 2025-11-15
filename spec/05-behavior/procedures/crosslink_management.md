---
procedure_id: crosslink_management
name: Crosslink Management
description: Maintain crosslink map so readers can hop between related concepts without dead ends or loops
roles: [codex_curator]
references_schemas:
  - codex_entry.schema.json
  - crosslink_map.schema.json
references_expertises:
  - codex_curator_publication
quality_bars: [integrity, accessibility]
---

# Crosslink Management

## Purpose

Maintain a coherent crosslink network that enables players to explore related concepts freely without encountering dead ends, circular loops, or broken references.

## Core Principles

- **Navigation Support**: Readers can follow interests from entry to entry
- **No Dead Ends**: Every entry connects to related concepts
- **No Infinite Loops**: Crosslink patterns don't trap readers
- **Descriptive Links**: Link text clarifies destination ("See Salvage Permits")
- **Integrity**: All crosslinks resolve to valid entries

## Steps

1. **Build Crosslink Map**: Maintain network overview
   - List all codex entries with IDs
   - Track outgoing links from each entry ("See also" lists)
   - Track incoming links to each entry (reverse references)
   - Identify orphaned entries (no incoming links)
   - Identify dead-end entries (no outgoing links)

2. **Design Crosslink Patterns**: Plan relationships
   - Hierarchical: general → specific
   - Thematic: related concepts
   - Contextual: appear together in narrative
   - Avoid creating circular "See also" chains

3. **Add Crosslinks to Entries**: Update "See also" sections
   - Select 2-5 most relevant related entries
   - Use descriptive link text (entry title, not "click here")
   - Order by relevance (most to least useful)
   - Bidirectional where appropriate (A links to B, B links to A)

4. **Resolve Dead Ends**: Ensure every entry has outgoing links
   - Find entries with empty "See also" lists
   - Identify at least 1-2 relevant connections
   - Add crosslinks or create hooks for missing entries

5. **Fix Orphans**: Connect isolated entries
   - Find entries with no incoming links
   - Add references from related entries
   - Ensure entry is reachable via navigation

6. **Test Navigation Paths**: Verify usability
   - Trace sample paths through crosslinks
   - Check for circular loops (A → B → C → A)
   - Ensure major concepts reachable from multiple paths
   - Verify link text accurately describes destination

7. **Update Map**: Keep crosslink map current
   - Regenerate after entry additions or updates
   - Track coverage (% entries with adequate crosslinks)
   - Document planned links for future entries

## Outputs

- **Crosslink Map**: Network visualization/list showing:
  - All entries and their connections
  - Orphaned entries (no incoming links)
  - Dead-end entries (no outgoing links)
  - Circular patterns (if any)
- **Updated Entries**: "See also" sections populated
- **Coverage Report**: Crosslink quality metrics
- **Future Link Hooks**: Planned connections requiring new entries

## Quality Checks

- Every entry has at least 1-2 outgoing crosslinks (no dead ends)
- Orphaned entries connected via incoming links
- No tight circular loops (A → B → A direct cycles)
- All crosslinks resolve to valid entry IDs
- Link text descriptive and accurate
- Major concepts reachable via multiple paths
- Network supports exploratory navigation
- Map stays current with entry additions/updates
