"""Style Lead role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class StyleLead(Role):
    """
    Style Lead: Maintain consistent voice, register, and motifs across all content.

    The Style Lead diagnoses style drift, writes style addendums with exemplars,
    provides targeted edit notes to content owners, and ensures visual/audio
    language aligns with the book's register and motif kit.

    Key responsibilities:
    - Detect and correct style drift across prose, captions, and PN surfaces
    - Write style addendums with exemplars and motif kits
    - Provide targeted edit notes to Scene Smith, Art Director, Audio Director
    - Validate visual and audio language alignment with register
    - Flag untranslatable idioms and suggest alternatives for Translator
    - Maintain phrase banks for PN diegetic phrasing patterns
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/style_lead.md"""
        return "style_lead"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Style Lead"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a style lead task.

        Supported tasks:
        - 'diagnose_drift': Identify style inconsistencies across sections
        - 'create_style_addendum': Write style rules and exemplars
        - 'generate_edit_notes': Create targeted fixes for content owners
        - 'validate_visual_language': Check art/audio alignment with register
        - 'create_phrase_bank': Generate PN diegetic phrasing patterns
        - 'check_motif_consistency': Validate motif usage across content
        - 'review_progress': Review and validate style improvements

        Args:
            context: Execution context

        Returns:
            Result with style guidance or validation results
        """
        task = context.task.lower()

        if task == "diagnose_drift":
            return self._diagnose_drift(context)
        elif task == "create_style_addendum":
            return self._create_style_addendum(context)
        elif task == "generate_edit_notes":
            return self._generate_edit_notes(context)
        elif task == "validate_visual_language":
            return self._validate_visual_language(context)
        elif task == "create_phrase_bank":
            return self._create_phrase_bank(context)
        elif task == "check_motif_consistency":
            return self._check_motif_consistency(context)
        # New tasks for loops
        elif task == "review_progress":
            return self._check_motif_consistency(context)
        elif task == "check_style":
            return self._diagnose_drift(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _diagnose_drift(self, context: RoleContext) -> RoleResult:
        """Identify style inconsistencies across sections."""
        system_prompt = self.build_system_prompt(context)

        sections = context.additional_context.get("sections", [])

        user_prompt = f"""# Task: Diagnose Style Drift

{self.format_artifacts(context.artifacts)}

## Sections to Analyze
{self._format_list(sections)}

Sample early, middle, and late sections. Tag issues with:
- `voice-shift`: Narrator voice changes inconsistently
- `register-mismatch`: Formality/informality varies inappropriately
- `motif-missing`: Established motifs absent where expected
- `over-exposition`: Info dumps or technical language on player surfaces
- `jargon-spike`: Sudden increase in specialized terms

For each issue, provide:
1. **Location**: Section/line reference
2. **Issue Type**: Tag from above
3. **Description**: What's wrong
4. **Severity**: blocker/warning/info
5. **Example**: Brief quote showing the problem

Respond in JSON format:
```json
{{
  "issues": [
    {{
      "location": "section-id or line ref",
      "type": "voice-shift|register-mismatch|motif-missing|over-exposition
              |jargon-spike",
      "description": "Issue description",
      "severity": "blocker|warning|info",
      "example": "Quote showing the problem"
    }}
  ],
  "summary": "Overall assessment",
  "priority_fixes": ["Fix 1", "Fix 2"]
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
                    "content_type": "drift_diagnosis",
                    "issues": data.get("issues", []),
                    "issue_count": len(data.get("issues", [])),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error diagnosing drift: {e}",
            )

    def _create_style_addendum(self, context: RoleContext) -> RoleResult:
        """Write style rules and exemplars."""
        system_prompt = self.build_system_prompt(context)

        issues = context.additional_context.get("issues", [])

        user_prompt = f"""# Task: Create Style Addendum

{self.format_artifacts(context.artifacts)}

## Issues to Address
{self._format_list([f"{i.get('type')}: {i.get('description')}" for i in issues])}

Create a style addendum with:

1. **Rules/Clarifications**: Sentence rhythm, idiom boundaries, POV distance,
   caption tone
2. **Motif Kit**: Reaffirm house motifs (e.g., "shadow-side neon", "low-G dust")
3. **Exemplars**: Before/after examples (3-5) showing the fix
4. **Localization Notes**: Puns to avoid, alternative phrasings

For each exemplar:
- **Before**: Quote with issue
- **After**: Corrected version
- **Motifs**: Which motifs used
- **Rationale**: Why this works

Respond in JSON format with complete addendum.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "style_addendum",
                    "addendum": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating style addendum: {e}",
            )

    def _generate_edit_notes(self, context: RoleContext) -> RoleResult:
        """Create targeted fixes for content owners."""
        system_prompt = self.build_system_prompt(context)

        issues = context.additional_context.get("issues", [])

        user_prompt = f"""# Task: Generate Edit Notes

{self.format_artifacts(context.artifacts)}

## Issues Requiring Fixes
{self._format_list([f"{i.get('location')}: {i.get('description')}" for i in issues])}

Create annotated edit notes: `file/section → issue → fix suggestion (1-2 lines)`

For each note:
- **Location**: Specific file/section/line
- **Owner**: scene_smith, art_director, audio_director, or player_narrator
- **Issue**: Brief problem description
- **Suggested Fix**: 1-2 line fix that preserves intent
- **Priority**: high/medium/low

Group by owner for easy routing.

Respond in JSON format with edit notes grouped by owner.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "edit_notes",
                    "notes": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error generating edit notes: {e}",
            )

    def _validate_visual_language(self, context: RoleContext) -> RoleResult:
        """Check art/audio alignment with register."""
        system_prompt = self.build_system_prompt(context)

        content = context.additional_context.get("content", {})
        content_type = context.additional_context.get("content_type", "visual")

        user_prompt = f"""# Task: Validate {content_type.title()} Language

{self.format_artifacts(context.artifacts)}

## Content to Validate
{self._format_dict(content)}

Check alignment with house style:
1. **Register Match**: Does tone fit the book's voice?
2. **Motif Presence**: Are established motifs present/appropriate?
3. **Drift Detection**: Any style inconsistencies?
4. **Technical Exposure**: Any technique-talk on player surfaces?

Respond in JSON format:
```json
{{
  "is_aligned": true|false,
  "issues": [
    {{
      "type": "register|motif|drift|technical",
      "description": "Issue description",
      "severity": "blocker|warning|info",
      "suggestion": "How to fix"
    }}
  ],
  "approved": true|false
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "visual_validation",
                    "is_aligned": data.get("is_aligned", False),
                    "approved": data.get("approved", False),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating visual language: {e}",
            )

    def _create_phrase_bank(self, context: RoleContext) -> RoleResult:
        """Generate PN diegetic phrasing patterns."""
        system_prompt = self.build_system_prompt(context)

        scenarios = context.additional_context.get("scenarios", [])

        user_prompt = f"""# Task: Create PN Phrase Bank

{self.format_artifacts(context.artifacts)}

## Scenarios Needing Diegetic Phrasing
{self._format_list(scenarios)}

Create phrase bank for Player-Narrator diegetic phrasing:
- Gate enforcement (access denied, requirements)
- Navigation cues (discovery, transitions)
- Status updates (inventory, conditions)
- Recall/memory triggers

For each phrase:
- **Scenario**: When used
- **Diegetic Phrase**: In-world phrasing (no plumbing)
- **Avoid**: Technical/meta phrasing to never use
- **Variants**: 2-3 alternative phrasings

Keep register consistent, stay in-world, avoid exposing game mechanics.

Respond in JSON format with phrase bank.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "phrase_bank",
                    "phrases": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating phrase bank: {e}",
            )

    def _check_motif_consistency(self, context: RoleContext) -> RoleResult:
        """Validate motif usage across content."""
        system_prompt = self.build_system_prompt(context)

        motifs = context.additional_context.get("motifs", [])
        content = context.additional_context.get("content", [])

        user_prompt = f"""# Task: Check Motif Consistency

{self.format_artifacts(context.artifacts)}

## House Motifs
{self._format_list(motifs)}

## Content to Check
{self._format_list(content)}

Validate motif usage:
1. **Presence**: Are motifs appearing where expected?
2. **Frequency**: Overused or underused?
3. **Consistency**: Same phrasing/imagery used?
4. **Appropriateness**: Motifs fit the moment?

Respond in JSON format:
```json
{{
  "is_consistent": true|false,
  "motif_analysis": [
    {{
      "motif": "motif name",
      "presence": "appropriate|missing|overused",
      "consistency": "good|needs-work",
      "notes": "Observations"
    }}
  ],
  "recommendations": ["Action 1", "Action 2"]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "motif_check",
                    "is_consistent": data.get("is_consistent", False),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error checking motif consistency: {e}",
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
