"""Plotwright role implementation."""

import json
import logging

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class Plotwright(Role):
    """
    Plotwright: Story structure and narrative arc specialist.

    The Plotwright designs the topology of quests - hubs, loops, gateways,
    and keystones. They think in structure before prose, ensuring choices
    are contrastive, returns are meaningful, and gates are diegetic.

    Key responsibilities:
    - Design narrative topology (hubs, loops, gateways)
    - Create TU briefs and section briefs
    - Define choice intents and expected outcomes
    - Establish keystone moments and safe returns
    - Maintain narrative coherence and pacing
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/plotwright.md"""
        return "plotwright"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Plotwright"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a plotwright task.

        Supported tasks:
        - 'generate_hooks': Create quest hooks
        - 'create_tu_brief': Design TU structure
        - 'create_topology': Design narrative topology
        - 'create_section_briefs': Generate section briefs
        - 'review_structure': Review narrative structure

        Args:
            context: Execution context

        Returns:
            Result with generated content and artifacts
        """
        task = context.task.lower()
        logger.info("Plotwright executing task: %s", task)
        logger.trace("Number of artifacts provided: %d", len(context.artifacts))

        if task == "generate_hooks":
            logger.debug("Generating quest hooks")
            return self._generate_hooks(context)
        elif task == "create_tu_brief":
            logger.debug("Creating TU brief")
            return self._create_tu_brief(context)
        elif task == "create_topology":
            logger.debug("Creating narrative topology")
            return self._create_topology(context)
        elif task == "create_section_briefs":
            logger.debug("Creating section briefs")
            return self._create_section_briefs(context)
        elif task == "review_structure":
            logger.debug("Reviewing narrative structure")
            return self._review_structure(context)
        else:
            logger.warning("Unknown Plotwright task: %s", task)
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _generate_hooks(self, context: RoleContext) -> RoleResult:
        """Generate quest hooks based on project metadata."""
        logger.debug("Generating quest hooks")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Generate Quest Hooks

{self.format_artifacts(context.artifacts)}

Please generate 3-5 compelling quest hooks for this project. Each hook should:
- Be concise (1-2 sentences)
- Create intrigue without revealing the full plot
- Suggest meaningful choices
- Hint at stakes and consequences
- Be player-neutral (no spoilers)

Format your response as JSON:
{{
  "hooks": [
    {{"title": "Hook Title", "summary": "Hook description", "tags": ["tag1", "tag2"]}},
    ...
  ]
}}
"""

        response = ""
        try:
            logger.trace("Calling LLM to generate hooks")
            response = self._call_llm(system_prompt, user_prompt)

            # Parse JSON from response (handles markdown code blocks)
            data = self._parse_json_from_response(response)
            num_hooks = len(data.get("hooks", []))
            logger.info("Successfully generated %d quest hooks", num_hooks)

            return RoleResult(
                success=True,
                output=response,
                metadata={"hooks": data.get("hooks", [])},
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response when generating hooks: %s", e)
            return RoleResult(
                success=False,
                output=response,
                error=f"Failed to parse JSON response: {e}",
            )
        except Exception as e:
            logger.error("Error generating hooks: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error generating hooks: {e}",
            )

    def _create_tu_brief(self, context: RoleContext) -> RoleResult:
        """Create a TU (Transmedia Unit) brief."""
        logger.debug("Creating TU brief")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Create TU Brief

{self.format_artifacts(context.artifacts)}

Create a TU brief that defines the narrative structure for this quest. Include:

1. **Narrative Goal**: What is the quest trying to accomplish?
2. **Key Beats**: Major story moments
3. **Choice Structure**: How choices branch and reconverge
4. **Gateway Map**: What checks gate progress (diegetic only)
5. **Safe Returns**: How players can loop back without feeling stuck
6. **Keystones**: Critical moments that need redundancy

Format as structured text (not necessarily JSON). Be specific but concise.
"""

        try:
            logger.trace("Calling LLM to create TU brief")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)
            logger.info(
                "Successfully created TU brief, size: %d characters", len(response)
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "tu_brief"},
            )

        except Exception as e:
            logger.error("Error creating TU brief: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating TU brief: {e}",
            )

    def _create_topology(self, context: RoleContext) -> RoleResult:
        """Design narrative topology."""
        logger.debug("Creating narrative topology")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Design Narrative Topology

{self.format_artifacts(context.artifacts)}

Design the narrative topology for this quest. Define:

1. **Hubs**: Central nodes where multiple paths converge
2. **Loops**: Sections players can revisit with different context
3. **Gateways**: Progress checks (must be diegetic - based on world facts)
4. **Branches**: Where choices lead to different paths
5. **Convergence Points**: Where branches rejoin

Provide a clear structure showing how these elements connect.
"""

        try:
            logger.trace("Calling LLM to design topology")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
            logger.info(
                "Successfully created topology design, size: %d characters",
                len(response),
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "topology"},
            )

        except Exception as e:
            logger.error("Error creating topology: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating topology: {e}",
            )

    def _create_section_briefs(self, context: RoleContext) -> RoleResult:
        """Create section briefs for scene development."""
        logger.debug("Creating section briefs")
        section_count = context.additional_context.get("section_count", 4)
        logger.trace("Creating %d section briefs", section_count)

        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Create Section Briefs

{self.format_artifacts(context.artifacts)}

Create section briefs that the Scene Smith can use to draft individual scenes.
For each major section, provide:

1. **Section Goal**: What this section accomplishes
2. **Key Beats**: Specific moments within the section
3. **Choice Intents**: What each choice is meant to test/reveal
4. **Expected Outcomes**: Where choices lead
5. **Open Questions**: What the Scene Smith should decide

Number of sections: {section_count}
"""

        try:
            logger.trace("Calling LLM to create section briefs")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)
            logger.info(
                "Successfully created %d section briefs, size: %d characters",
                section_count,
                len(response),
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "section_briefs"},
            )

        except Exception as e:
            logger.error("Error creating section briefs: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating section briefs: {e}",
            )

    def _review_structure(self, context: RoleContext) -> RoleResult:
        """Review narrative structure for issues."""
        logger.debug("Reviewing narrative structure")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Review Narrative Structure

{self.format_artifacts(context.artifacts)}

Review the narrative structure and identify:

1. **Dead Ends**: Choices that don't lead anywhere meaningful
2. **Railroading**: Sections where choice is illusory
3. **Gateway Issues**: Non-diegetic checks or unfair requirements
4. **Keystone Risks**: Critical moments without backup paths
5. **Loop Problems**: Returns that don't alter context
6. **Structural Gaps**: Missing connections or unclear progressions

Provide specific feedback with line/section references where possible.
"""

        try:
            logger.trace("Calling LLM to review narrative structure")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
            logger.info(
                "Successfully completed structure review, size: %d characters",
                len(response),
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "structure_review"},
            )

        except Exception as e:
            logger.error("Error reviewing structure: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error reviewing structure: {e}",
            )
