"""Audio Director role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class AudioDirector(Role):
    """
    Audio Director: Plan audio storytelling and coordinate production.

    The Audio Director selects cue targets, specifies placement and timing,
    writes player-safe cue descriptions, and ensures audio language aligns
    with the book's register and motifs.

    Key responsibilities:
    - Select cue targets with clear narrative purpose
    - Specify placement, duration, intensity curves
    - Write player-safe cue descriptions and captions
    - Provide accessibility (text equivalents) and safety notes
    - Coordinate with Style Lead on audio language alignment
    - Support localization for voice-over content
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/audio_director.md"""
        return "audio_director"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Audio Director"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute an audio director task.

        Supported tasks:
        - 'select_cues': Choose audio cues and targets
        - 'create_audio_plan': Write cue specs and placement
        - 'write_cue_description': Create player-safe descriptions
        - 'validate_asset': Check audio alignment with intent
        - 'create_text_equivalent': Generate accessibility captions
        - 'create_cuelist': Create list of audio cues (alias for select_cues)
        - 'review_cuelist': Review and finalize cues (alias for create_audio_plan)

        Args:
            context: Execution context

        Returns:
            Result with audio plans or validation results
        """
        task = context.task.lower()

        if task == "select_cues":
            return self._select_cues(context)
        elif task == "create_audio_plan":
            return self._create_audio_plan(context)
        elif task == "write_cue_description":
            return self._write_cue_description(context)
        elif task == "validate_asset":
            return self._validate_asset(context)
        elif task == "create_text_equivalent":
            return self._create_text_equivalent(context)
        # New tasks for audio_pass loop
        elif task == "create_cuelist":
            return self._select_cues(context)
        elif task == "review_cuelist":
            return self._create_audio_plan(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _select_cues(self, context: RoleContext) -> RoleResult:
        """Choose audio cues and targets."""
        system_prompt = self.build_system_prompt(context)

        sections = context.additional_context.get("sections", [])

        user_prompt = f"""# Task: Select Audio Cues

{self.format_artifacts(context.artifacts)}

## Available Sections
{self._format_list(sections)}

For each proposed cue:
1. **Cue ID**: Unique identifier
2. **Scene Anchor**: Which section/moment
3. **Purpose**: clarify_affordance|intensify_stakes|transition|recall_motif
4. **Type**: ambience|foley|stinger|voice_over
5. **Spoiler Risk**: low|medium|high
6. **Rationale**: Why this cue serves the story

If risk > low, propose alternate cue or note as plan-only.

Prioritize:
- Clarifying affordances (what's interactive, what's blocked)
- Intensifying stakes at key moments
- Smooth transitions between scenes
- Motif recall and reinforcement

Respond in JSON format:
```json
{{
  "selections": [
    {{
      "cue_id": "unique-id",
      "scene_anchor": "section-id",
      "purpose": "clarify_affordance|intensify_stakes|transition|recall_motif",
      "type": "ambience|foley|stinger|voice_over",
      "spoiler_risk": "low|medium|high",
      "rationale": "Why this matters",
      "alternate_if_risky": "Alternative cue if risk too high"
    }}
  ]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "cue_selections",
                    "selections": data.get("selections", []),
                    "count": len(data.get("selections", [])),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error selecting cues: {e}",
            )

    def _create_audio_plan(self, context: RoleContext) -> RoleResult:
        """Write cue specs and placement."""
        system_prompt = self.build_system_prompt(context)

        cue = context.additional_context.get("cue", {})

        user_prompt = f"""# Task: Create Audio Plan

{self.format_artifacts(context.artifacts)}

## Cue
{self._format_dict(cue)}

Create detailed audio plan with:

1. **Description** (player-safe):
   - What the listener perceives
   - NOT how it was made (no DAW/technique talk)

2. **Placement**:
   - Entry/exit points
   - Loop or one-shot
   - Suggested duration

3. **Intensity Curve**:
   - low/medium/high
   - Ramp/fade guidance

4. **Motif Ties**:
   - How cue threads house motifs

5. **Captions/Text Equivalents**:
   - Accessibility text
   - Describes sound effect/music

6. **Safety Notes**:
   - Avoid sudden peaks
   - Caution tags for harsh sounds

7. **Localization Notes** (if VO):
   - Dialect, register
   - Terms to preserve

Respond in JSON format with complete audio plan.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "audio_plan",
                    "plan": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating audio plan: {e}",
            )

    def _write_cue_description(self, context: RoleContext) -> RoleResult:
        """Create player-safe descriptions."""
        system_prompt = self.build_system_prompt(context)

        cue_context = context.additional_context.get("cue_context", {})

        user_prompt = f"""# Task: Write Cue Description

{self.format_artifacts(context.artifacts)}

## Cue Context
{self._format_dict(cue_context)}

Write a player-safe cue description:
- Describe what listener perceives
- No technique talk (no "reverb", "compression", "samples")
- Stay diegetic (in-world references only)
- Consistent with house style
- 1-2 sentences

Focus on the experience, not the production.

Respond with:
```json
{{
  "description": "Cue description",
  "safety_check": "Confirms no technique exposure",
  "style_notes": "Alignment with register"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=data.get("description", response),
                metadata={
                    "content_type": "cue_description",
                    "description": data.get("description", ""),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error writing cue description: {e}",
            )

    def _validate_asset(self, context: RoleContext) -> RoleResult:
        """Check audio alignment with intent."""
        system_prompt = self.build_system_prompt(context)

        audio_plan = context.additional_context.get("audio_plan", {})
        asset_notes = context.additional_context.get("asset_notes", {})

        user_prompt = f"""# Task: Validate Audio Asset

{self.format_artifacts(context.artifacts)}

## Audio Plan
{self._format_dict(audio_plan)}

## Asset Notes
{self._format_dict(asset_notes)}

Validate asset against plan:
1. **Purpose**: Serves intended narrative function?
2. **Intensity**: Matches curve specification?
3. **Style Alignment**: Consistent with house motifs?
4. **Safety**: Reasonable loudness, no shocks?
5. **Accessibility**: Text equivalents present?

Respond in JSON format:
```json
{{
  "approved": true|false,
  "issues": [
    {{
      "aspect": "purpose|intensity|style|safety|accessibility",
      "description": "Issue description",
      "severity": "blocker|request-revision|minor"
    }}
  ],
  "feedback": "Overall assessment and revision guidance if needed"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "asset_validation",
                    "approved": data.get("approved", False),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating asset: {e}",
            )

    def _create_text_equivalent(self, context: RoleContext) -> RoleResult:
        """Generate accessibility captions."""
        system_prompt = self.build_system_prompt(context)

        cue_desc = context.additional_context.get("cue_description", "")

        user_prompt = f"""# Task: Create Text Equivalent

{self.format_artifacts(context.artifacts)}

## Cue Description
{cue_desc}

Create text equivalent for accessibility:
- Describes the sound/music
- Useful for deaf/hard-of-hearing players
- Spoiler-safe
- Focuses on narrative-relevant aspects
- Brief (5-10 words)

Examples:
- [distant alarm wails]
- [footsteps echo in corridor]
- [tense ambient music rises]

Respond with:
```json
{{
  "text_equivalent": "[description]",
  "style_notes": "Formatting guidance"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=data.get("text_equivalent", response),
                metadata={
                    "content_type": "text_equivalent",
                    "text_equivalent": data.get("text_equivalent", ""),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating text equivalent: {e}",
            )

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dictionary as bullet list."""
        if not d:
            return "(empty)"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())

    def _format_list(self, items: list[str]) -> str:
        """Format list as bullet list."""
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)
