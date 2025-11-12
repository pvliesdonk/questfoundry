"""Art Director role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class ArtDirector(Role):
    """
    Art Director: Plan visual storytelling and coordinate illustration.

    The Art Director selects scenes/subjects for illustration, defines
    composition intent and purpose, writes spoiler-safe captions, and
    ensures visual language matches the book's style and motifs.

    Key responsibilities:
    - Select scenes/subjects for illustration with clear narrative purpose
    - Write composition intent and spoiler-safe captions
    - Specify visual constraints (aspect, palette, negative constraints)
    - Provide accessibility notes and alt text guidance
    - Coordinate with Style Lead on visual language alignment
    - Assess spoiler risk and mitigate through alternate subjects
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/art_director.md"""
        return "art_director"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Art Director"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute an art director task.

        Supported tasks:
        - 'select_subjects': Choose scenes/subjects for illustration
        - 'create_art_plan': Write composition intent and constraints
        - 'write_caption': Create spoiler-safe caption
        - 'validate_render': Check illustration alignment with intent
        - 'create_alt_text': Generate accessibility descriptions
        - 'create_shotlist': Create list of shots (alias for select_subjects)
        - 'review_shotlist': Review and finalize shots (alias for create_art_plan)

        Args:
            context: Execution context

        Returns:
            Result with art plans or validation results
        """
        task = context.task.lower()

        if task == "select_subjects":
            return self._select_subjects(context)
        elif task == "create_art_plan":
            return self._create_art_plan(context)
        elif task == "write_caption":
            return self._write_caption(context)
        elif task == "validate_render":
            return self._validate_render(context)
        elif task == "create_alt_text":
            return self._create_alt_text(context)
        # New tasks for art_touch_up loop
        elif task == "create_shotlist":
            return self._select_subjects(context)
        elif task == "review_shotlist":
            return self._create_art_plan(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _select_subjects(self, context: RoleContext) -> RoleResult:
        """Choose scenes/subjects for illustration."""
        system_prompt = self.build_system_prompt(context)

        sections = context.additional_context.get("sections", [])

        user_prompt = f"""# Task: Select Illustration Subjects

{self.format_artifacts(context.artifacts)}

## Available Sections
{self._format_list(sections)}

For each proposed illustration, provide:
1. **Subject**: Who/what to illustrate
2. **Scene Anchor**: Which section
3. **Purpose**: clarify|foreshadow|mood|signpost
4. **Spoiler Risk**: low|medium|high
5. **Rationale**: Why this image serves the story

If risk > low, propose alternate subject or note as plan-only.

Prioritize:
- Iconic moments that anchor player memory
- Clarifying visuals for complex spaces/objects
- Mood reinforcement at key transitions
- Signposting for navigation

Respond in JSON format:
```json
{{
  "selections": [
    {{
      "subject": "Description of subject",
      "scene_anchor": "section-id",
      "purpose": "clarify|foreshadow|mood|signpost",
      "spoiler_risk": "low|medium|high",
      "rationale": "Why this matters",
      "alternate_if_risky": "Alternative subject if risk too high"
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
                    "content_type": "subject_selections",
                    "selections": data.get("selections", []),
                    "count": len(data.get("selections", [])),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error selecting subjects: {e}",
            )

    def _create_art_plan(self, context: RoleContext) -> RoleResult:
        """Write composition intent and constraints."""
        system_prompt = self.build_system_prompt(context)

        subject = context.additional_context.get("subject", {})

        user_prompt = f"""# Task: Create Art Plan

{self.format_artifacts(context.artifacts)}

## Subject
{self._format_dict(subject)}

Create detailed art plan with:

1. **Composition Intent**:
   - Framing (wide/medium/close, angle)
   - Focal points (what draws the eye)
   - Motion cues (energy, direction)
   - Lighting mood

2. **Caption** (player-safe):
   - 1-2 sentences, atmospheric
   - No twist reveals or spoilers
   - Focuses on what player perceives

3. **Constraints**:
   - Aspect ratio preference
   - Palette/motif hooks (house style)
   - Negative constraints (avoid clichés)

4. **Accessibility Notes**:
   - Alt text guidance (succinct, descriptive, spoiler-safe)
   - Contrast considerations
   - Motion safety notes if applicable

Respond in JSON format with complete art plan.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "art_plan",
                    "plan": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating art plan: {e}",
            )

    def _write_caption(self, context: RoleContext) -> RoleResult:
        """Create spoiler-safe caption."""
        system_prompt = self.build_system_prompt(context)

        image_context = context.additional_context.get("image_context", {})

        user_prompt = f"""# Task: Write Caption

{self.format_artifacts(context.artifacts)}

## Image Context
{self._format_dict(image_context)}

Write a player-safe caption:
- 1-2 sentences
- Atmospheric, not expository
- No spoilers or twist reveals
- No internal labels or gate logic
- Consistent with house style

Focus on what the player perceives, not what they'll discover.

Respond with:
```json
{{
  "caption": "Caption text",
  "safety_check": "Confirms no spoilers",
  "style_notes": "Alignment with register"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=data.get("caption", response),
                metadata={
                    "content_type": "caption",
                    "caption": data.get("caption", ""),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error writing caption: {e}",
            )

    def _validate_render(self, context: RoleContext) -> RoleResult:
        """Check illustration alignment with intent."""
        system_prompt = self.build_system_prompt(context)

        art_plan = context.additional_context.get("art_plan", {})
        render_notes = context.additional_context.get("render_notes", {})

        user_prompt = f"""# Task: Validate Render

{self.format_artifacts(context.artifacts)}

## Art Plan
{self._format_dict(art_plan)}

## Render Notes
{self._format_dict(render_notes)}

Validate render against plan:
1. **Composition**: Matches intent?
2. **Focal Points**: Where intended?
3. **Style Alignment**: Consistent with house style?
4. **Constraints**: Respected (aspect, palette, avoidances)?
5. **Accessibility**: Alt text present and adequate?

Respond in JSON format:
```json
{{
  "approved": true|false,
  "issues": [
    {{
      "aspect": "composition|style|constraints|accessibility",
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
                    "content_type": "render_validation",
                    "approved": data.get("approved", False),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating render: {e}",
            )

    def _create_alt_text(self, context: RoleContext) -> RoleResult:
        """Generate accessibility descriptions."""
        system_prompt = self.build_system_prompt(context)

        image_desc = context.additional_context.get("image_description", "")

        user_prompt = f"""# Task: Create Alt Text

{self.format_artifacts(context.artifacts)}

## Image Description
{image_desc}

Create alt text that is:
- Succinct (1-2 sentences)
- Descriptive of key visual elements
- Spoiler-safe (no reveals)
- Useful for screen readers
- Focuses on narrative-relevant details

Respond with:
```json
{{
  "alt_text": "Alt text content",
  "length": "character count",
  "spoiler_check": "Confirms player-safe"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=data.get("alt_text", response),
                metadata={
                    "content_type": "alt_text",
                    "alt_text": data.get("alt_text", ""),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating alt text: {e}",
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
