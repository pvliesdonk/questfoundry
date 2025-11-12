"""Gatekeeper role implementation."""

import json
import logging

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class Gatekeeper(Role):
    """
    Gatekeeper: Quality validation specialist.

    The Gatekeeper checks artifacts against the 8 quality bars,
    protecting player surfaces while unblocking creators with specific,
    actionable feedback. They focus on lightweight, targeted checks
    rather than exhaustive reviews.

    Key responsibilities:
    - Validate artifacts against quality bars
    - Provide specific, actionable feedback
    - Block merges only on failing bars
    - Protect player-neutral boundaries
    - Enable rapid iteration with pre-gate checks
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/gatekeeper.md"""
        return "gatekeeper"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Gatekeeper"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a gatekeeper task.

        Supported tasks:
        - 'pre_gate': Quick check for likely issues
        - 'gate_check': Full quality bar validation
        - 'validate_bar': Check specific quality bar
        - 'export_check': Validate export/view formatting
        - 'evaluate_quality_bars': Evaluate all quality bars (alias for gate_check)
        - 'collect_findings': Collect quality findings (delegates to gate_check)
        - 'triage_blockers': Triage findings by severity (delegates to gate_check)
        - 'create_gatecheck_report': Create comprehensive report
          (delegates to gate_check)

        Args:
            context: Execution context

        Returns:
            Result with validation report
        """
        task = context.task.lower()
        logger.info("Gatekeeper executing task: %s", task)
        logger.trace("Number of artifacts to validate: %d", len(context.artifacts))

        if task == "pre_gate":
            logger.debug("Performing pre-gate check")
            return self._pre_gate(context)
        elif task == "gate_check":
            logger.debug("Performing full gate check")
            return self._gate_check(context)
        elif task == "validate_bar":
            logger.debug("Validating specific quality bar")
            return self._validate_bar(context)
        elif task == "export_check":
            logger.debug("Checking export formatting")
            return self._export_check(context)
        # New tasks for gatecheck loop
        elif task == "evaluate_quality_bars":
            logger.debug("Evaluating quality bars")
            return self._gate_check(context)
        elif task == "collect_findings":
            logger.debug("Collecting quality findings")
            return self._gate_check(context)
        elif task == "triage_blockers":
            logger.debug("Triaging blockers by severity")
            return self._gate_check(context)
        elif task == "create_gatecheck_report":
            logger.debug("Creating gatecheck report")
            return self._gate_check(context)
        # New tasks for post_mortem loop
        elif task == "final_validation":
            logger.debug("Performing final validation")
            return self._gate_check(context)
        elif task == "create_post_mortem_report":
            logger.debug("Creating post-mortem report")
            return self._gate_check(context)
        # New tasks for archive_snapshot loop
        elif task == "validate_snapshot":
            logger.debug("Validating snapshot")
            return self._gate_check(context)
        else:
            logger.warning("Unknown Gatekeeper task: %s", task)
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _pre_gate(self, context: RoleContext) -> RoleResult:
        """Quick pre-gate check for obvious issues."""
        logger.debug("Running pre-gate check")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Pre-Gate Check

{self.format_artifacts(context.artifacts)}

Perform a quick 5-10 minute pre-gate check. Identify:

1. **Obvious Blockers**: Clear violations of quality bars
2. **Quick Wins**: Easy fixes that improve quality
3. **Likely Issues**: Areas that need deeper review

Focus on:
- Player-Neutral boundary violations (spoilers in player-facing text)
- Structural problems (dead ends, railroading)
- Diegetic gate violations (meta/mechanical checks)

Provide brief, actionable feedback. Don't deep-dive; flag for later review.

Format as JSON:
{{
  "status": "pass|warning|fail",
  "blockers": ["Issue 1", "Issue 2"],
  "quick_wins": ["Suggestion 1", "Suggestion 2"],
  "review_needed": ["Area 1", "Area 2"]
}}
"""

        response = ""
        try:
            logger.trace("Calling LLM for pre-gate check")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            # Parse JSON from response (handles markdown code blocks)
            data = self._parse_json_from_response(response)
            status = data.get("status", "unknown")
            num_blockers = len(data.get("blockers", []))
            logger.info(
                "Pre-gate check completed with status: %s, blockers: %d",
                status,
                num_blockers,
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "check_type": "pre_gate",
                    "status": status,
                    "blockers": data.get("blockers", []),
                    "quick_wins": data.get("quick_wins", []),
                    "review_needed": data.get("review_needed", []),
                },
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON in pre-gate check: %s", e)
            return RoleResult(
                success=False,
                output=response,
                error=f"Failed to parse JSON response: {e}",
            )
        except Exception as e:
            logger.error("Error in pre-gate check: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error in pre-gate check: {e}",
            )

    def _gate_check(self, context: RoleContext) -> RoleResult:
        """Full gate check against all quality bars."""
        logger.debug("Running full gate check")

        system_prompt = self.build_system_prompt(context)

        # Get which bars to check (default: all 8)
        bars = context.additional_context.get(
            "bars",
            [
                "integrity",
                "reachability",
                "style_consistency",
                "gateway_design",
                "nonlinearity",
                "determinism",
                "presentation",
                "spoiler_hygiene",
            ],
        )
        logger.trace("Checking %d quality bars: %s", len(bars), ", ".join(bars))

        user_prompt = f"""# Task: Full Gate Check

{self.format_artifacts(context.artifacts)}

Validate these artifacts against the following quality bars:

{", ".join(f"**{bar}**" for bar in bars)}

For each bar, provide:
- **Status**: pass/fail
- **Issues**: Specific problems found (with line/section references)
- **Fixes**: Concrete suggestions to address issues

Quality Bar Definitions:
1. **Integrity**: Schema conformance, no missing required fields
2. **Reachability**: All choices lead somewhere, no dead ends
3. **Style Consistency**: Tone, voice, formatting match project style
4. **Gateway Design**: Checks are diegetic (world-based), not meta/mechanical
5. **Nonlinearity**: Multiple meaningful paths, not railroading
6. **Determinism**: Outcomes follow from choices, not random
7. **Presentation**: Formatting, readability, polish
8. **Spoiler Hygiene**: Player-neutral boundaries maintained

Format as JSON:
{{
  "overall_status": "pass|fail",
  "merge_safe": true|false,
  "bars": {{
    "bar_name": {{
      "status": "pass|fail",
      "issues": ["Issue 1", "Issue 2"],
      "fixes": ["Fix 1", "Fix 2"]
    }}
  }}
}}
"""

        response = ""
        try:
            logger.trace("Calling LLM for gate check")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)

            # Parse JSON from response (handles markdown code blocks)
            data = self._parse_json_from_response(response)
            overall_status = data.get("overall_status", "unknown")
            merge_safe = data.get("merge_safe", False)
            logger.info(
                "Gate check completed - overall status: %s, merge safe: %s",
                overall_status,
                merge_safe,
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "check_type": "gate_check",
                    "overall_status": overall_status,
                    "merge_safe": merge_safe,
                    "bars": data.get("bars", {}),
                },
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON in gate check: %s", e)
            return RoleResult(
                success=False,
                output=response,
                error=f"Failed to parse JSON response: {e}",
            )
        except Exception as e:
            logger.error("Error in gate check: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error in gate check: {e}",
            )

    def _validate_bar(self, context: RoleContext) -> RoleResult:
        """Validate a specific quality bar."""
        bar_name = context.additional_context.get("bar_name")
        if not bar_name:
            logger.warning(
                "bar_name not provided in additional_context for validate_bar"
            )
            return RoleResult(
                success=False,
                output="",
                error="bar_name required in additional_context",
            )

        logger.debug("Validating quality bar: %s", bar_name)
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Validate Quality Bar - {bar_name}

{self.format_artifacts(context.artifacts)}

Validate these artifacts against the **{bar_name}** quality bar only.

Provide:
- Specific issues found (with line/section references)
- Concrete fixes for each issue
- Overall pass/fail assessment

Focus deeply on this one bar; ignore other quality aspects.
"""

        try:
            logger.trace("Calling LLM to validate bar: %s", bar_name)
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)
            logger.info("Successfully validated quality bar: %s", bar_name)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "check_type": "validate_bar",
                    "bar_name": bar_name,
                },
            )

        except Exception as e:
            logger.error("Error validating bar %s: %s", bar_name, e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating bar {bar_name}: {e}",
            )

    def _export_check(self, context: RoleContext) -> RoleResult:
        """Check export/view formatting."""
        logger.debug("Checking export/view formatting")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Export/View Check

{self.format_artifacts(context.artifacts)}

Check the export/view formatting:

1. **Front Matter**: Title, author, metadata present and correct
2. **Navigation**: TOC, links, structure clear
3. **Labels**: Section headers, artifact IDs consistent
4. **Formatting**: Markdown/HTML valid, no broken elements
5. **Player Safety**: No spoilers in Cold view

Provide specific issues and fixes.
"""

        try:
            logger.trace("Calling LLM for export check")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)
            logger.info("Export check completed successfully")

            return RoleResult(
                success=True,
                output=response,
                metadata={"check_type": "export_check"},
            )

        except Exception as e:
            logger.error("Error in export check: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error in export check: {e}",
            )
