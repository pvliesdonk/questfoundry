---
procedure_id: art_determinism_planning
name: Art Determinism Planning
description: Set reproducibility requirements (if promised) to be logged off-surface by Illustrator
roles: [art_director]
references_schemas:
  - art_plan.schema.json
  - determinism_log.schema.json
references_expertises:
  - art_director_planning
quality_bars: [determinism, presentation]
---

# Art Determinism Planning

## Purpose

Establish and communicate reproducibility requirements for illustrations when determinism has been promised, ensuring logs are maintained off-surface to preserve player-safe boundaries.

## Core Principles

- **Off-Surface Logging**: All determinism data (seeds, models, settings) stays off player-visible surfaces
- **Promise-Driven**: Only require determinism when explicitly promised for reproducibility
- **Illustrator Ownership**: Illustrator maintains logs; Art Director sets requirements
- **No Surface Leakage**: Technique details never appear in captions, alt text, or visible metadata

## Steps

1. **Assess Promise Status**: Determine if reproducibility has been committed for this work
2. **Define Requirements**: Specify what needs to be logged if determinism is promised:
   - Generation seeds (if synthetic)
   - Model/tool versions
   - Key settings or parameters
   - Capture/source metadata (if photographic/hybrid)
3. **Communicate to Illustrator**: Include determinism requirements in art plan
4. **Specify Log Location**: Ensure Illustrator knows logs go to off-surface storage
5. **Review Compliance**: Verify logs are complete and off-surface after rendering

## Outputs

- **Determinism Requirements**: Specification in art plan for what must be logged
- **Log Verification**: Confirmation that logs are complete and properly stored off-surface
- **Reproducibility Documentation**: Off-surface records enabling future recreation

## Quality Checks

- Determinism requirements only specified when reproducibility is promised
- All technique data kept off player-visible surfaces
- Illustrator has clear guidance on what to log
- Logs are complete and stored off-surface
- No seeds, models, or technical metadata leak into captions or alt text
