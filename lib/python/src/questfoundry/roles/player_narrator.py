"""Player Narrator role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class PlayerNarrator(Role):
    """
    Player Narrator: Test player experience flow and diegetic presentation.

    The Player Narrator validates player-facing surfaces, tests navigation flows,
    ensures gate enforcement stays diegetic, and provides experience reports
    from the player's perspective.

    Key responsibilities:
    - Test player experience flow through content
    - Validate diegetic gate phrasing (no plumbing exposed)
    - Check navigation clarity and affordances
    - Ensure choice labels are distinct and clear
    - Validate accessibility of player surfaces
    - Report experience issues from player perspective
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/player_narrator.md"""
        return "player_narrator"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Player Narrator"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a player narrator task.

        Supported tasks:
        - 'test_flow': Experience content flow as player
        - 'validate_gates': Check diegetic gate enforcement
        - 'assess_navigation': Evaluate wayfinding clarity
        - 'check_affordances': Validate interaction clarity
        - 'generate_experience_report': Create player perspective report
        - 'perform_narration': Perform narration/flow testing (alias for test_flow)
        - 'identify_issues': Identify issues in flow (alias for validate_gates)
        - 'create_report': Create experience report
          (alias for generate_experience_report)
        - 'collect_metrics': Collect experience metrics (alias for test_flow)
        - 'final_validation': Final validation of experience
          (alias for validate_gates)
        - 'create_post_mortem_report': Create post-mortem report
          (alias for generate_experience_report)

        Args:
            context: Execution context

        Returns:
            Result with experience reports and validation
        """
        task = context.task.lower()

        if task == "test_flow":
            return self._test_flow(context)
        elif task == "validate_gates":
            return self._validate_gates(context)
        elif task == "assess_navigation":
            return self._assess_navigation(context)
        elif task == "check_affordances":
            return self._check_affordances(context)
        elif task == "generate_experience_report":
            return self._generate_experience_report(context)
        # New tasks for loops
        elif task == "perform_narration":
            return self._test_flow(context)
        elif task == "identify_issues":
            return self._validate_gates(context)
        elif task == "create_report":
            return self._generate_experience_report(context)
        elif task == "collect_metrics":
            return self._test_flow(context)
        elif task == "final_validation":
            return self._validate_gates(context)
        elif task == "create_post_mortem_report":
            return self._generate_experience_report(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _test_flow(self, context: RoleContext) -> RoleResult:
        """Experience content flow as player."""
        system_prompt = self.build_system_prompt(context)

        sections = context.additional_context.get("sections", [])
        entry_point = context.additional_context.get("entry_point", "")

        user_prompt = f"""# Task: Test Player Flow

{self.format_artifacts(context.artifacts)}

## Sections to Test
{self._format_list(sections)}

## Entry Point
{entry_point}

Test player experience from entry point through sections:

1. **Navigation Flow**:
   - Can player find their way?
   - Are transitions clear?
   - Dead ends or confusion points?

2. **Choice Clarity**:
   - Are choice labels distinct?
   - Consequences telegraphed appropriately?
   - Options make sense in context?

3. **Pacing**:
   - Info density appropriate?
   - Rhythm varies or monotonous?
   - Stakes escalate naturally?

4. **Recall & Signposting**:
   - Can player remember key info?
   - Callbacks land effectively?
   - Landmarks aid wayfinding?

For each issue found:
- Location (section/choice)
- Problem description
- Player impact (confusion|frustration|lost)
- Severity (blocker|warning|minor)
- Suggestion

Respond in JSON format:
```json
{{
  "flow_test": {{
    "entry_point": "{entry_point}",
    "sections_tested": [],
    "overall_assessment": "smooth|bumpy|broken",
    "issues": [
      {{
        "location": "section-id or choice",
        "problem": "description",
        "player_impact": "confusion|frustration|lost|other",
        "severity": "blocker|warning|minor",
        "suggestion": "how to fix"
      }}
    ],
    "highlights": ["positive moments"]
  }}
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "flow_test",
                    "flow_test": data.get("flow_test", {}),
                    "issue_count": len(data.get("flow_test", {}).get("issues", [])),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error testing flow: {e}",
            )

    def _validate_gates(self, context: RoleContext) -> RoleResult:
        """Check diegetic gate enforcement."""
        system_prompt = self.build_system_prompt(context)

        gates = context.additional_context.get("gates", [])

        user_prompt = f"""# Task: Validate Gate Phrasing

{self.format_artifacts(context.artifacts)}

## Gates to Validate
{self._format_list([f"{g.get('id')}: {g.get('description')}" for g in gates])}

Check each gate for diegetic phrasing:

1. **Diegetic Check**:
   - Phrasing stays in-world?
   - No plumbing exposed (no "requirements", "locked", "stats")?
   - Uses natural language?

2. **Clarity**:
   - Player understands what's needed?
   - Feedback is helpful?
   - Mystery vs confusion balance?

3. **Style Consistency**:
   - Matches book's register?
   - Uses established phrase patterns?
   - Voice consistent?

For each gate:
- **Status**: diegetic|leaky|broken
- **Issues**: What exposes plumbing
- **Suggestion**: Diegetic alternative

Respond in JSON format:
```json
{{
  "gate_validation": {{
    "gates_checked": 0,
    "diegetic_count": 0,
    "issues": [
      {{
        "gate_id": "id",
        "status": "diegetic|leaky|broken",
        "problems": ["exposes X", "says Y"],
        "suggested_phrasing": "Diegetic alternative"
      }}
    ],
    "phrase_bank_additions": ["pattern 1", "pattern 2"]
  }}
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
                    "content_type": "gate_validation",
                    "validation": data.get("gate_validation", {}),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating gates: {e}",
            )

    def _assess_navigation(self, context: RoleContext) -> RoleResult:
        """Evaluate wayfinding clarity."""
        system_prompt = self.build_system_prompt(context)

        structure = context.additional_context.get("structure", {})

        user_prompt = f"""# Task: Assess Navigation

{self.format_artifacts(context.artifacts)}

## Structure
{self._format_dict(structure)}

Assess navigation from player perspective:

1. **Wayfinding**:
   - Can player orient themselves?
   - Landmarks clear?
   - Backtracking feasible?

2. **Signposting**:
   - Forward paths clear?
   - Dead ends marked?
   - Discovery vs confusion?

3. **Mental Model**:
   - Does topology make sense?
   - Hub/spoke clear?
   - Loops intuitive?

4. **Accessibility**:
   - Navigation aids adequate?
   - Choice labels informative?
   - State tracking clear?

Respond in JSON format:
```json
{{
  "navigation_assessment": {{
    "overall_clarity": "clear|confusing|broken",
    "strengths": ["strength 1", "strength 2"],
    "issues": [
      {{
        "area": "location or structure element",
        "problem": "description",
        "suggestion": "improvement"
      }}
    ],
    "recommendations": ["action 1", "action 2"]
  }}
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
                    "content_type": "navigation_assessment",
                    "assessment": data.get("navigation_assessment", {}),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error assessing navigation: {e}",
            )

    def _check_affordances(self, context: RoleContext) -> RoleResult:
        """Validate interaction clarity."""
        system_prompt = self.build_system_prompt(context)

        interactions = context.additional_context.get("interactions", [])

        user_prompt = f"""# Task: Check Affordances

{self.format_artifacts(context.artifacts)}

## Interactions
{self._format_list(interactions)}

Check affordance clarity:

1. **What's Interactive**:
   - Clear what player can interact with?
   - Appropriate signaling (without meta)?
   - Discoverable but not overwhelming?

2. **What's Blocked**:
   - Blocks communicated diegetically?
   - Rationale clear (when appropriate)?
   - Hints at unlock conditions (when relevant)?

3. **Feedback**:
   - Actions have appropriate response?
   - State changes clear?
   - Consequences telegraphed?

Respond in JSON format:
```json
{{
  "affordance_check": {{
    "clear_affordances": 0,
    "unclear_affordances": 0,
    "issues": [
      {{
        "interaction": "description",
        "problem": "what's unclear",
        "suggestion": "how to improve"
      }}
    ]
  }}
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
                    "content_type": "affordance_check",
                    "check": data.get("affordance_check", {}),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error checking affordances: {e}",
            )

    def _generate_experience_report(self, context: RoleContext) -> RoleResult:
        """Create player perspective report."""
        system_prompt = self.build_system_prompt(context)

        test_results = context.additional_context.get("test_results", {})

        user_prompt = f"""# Task: Generate Experience Report

{self.format_artifacts(context.artifacts)}

## Test Results
{self._format_dict(test_results)}

Create comprehensive experience report:

1. **Executive Summary**:
   - Overall player experience assessment
   - Major strengths
   - Critical issues

2. **Navigation & Flow**:
   - Wayfinding effectiveness
   - Pacing and rhythm
   - Signposting quality

3. **Diegetic Presentation**:
   - Gate enforcement quality
   - Affordance clarity
   - Voice consistency

4. **Player Impact**:
   - Moments of confusion
   - Moments of delight
   - Frustration points

5. **Recommendations**:
   - Priority fixes
   - Enhancement opportunities
   - Risk areas

Respond in JSON format with complete report.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "experience_report",
                    "report": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error generating report: {e}",
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
