"""Scene Smith role implementation."""

import logging

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class SceneSmith(Role):
    """
    Scene Smith: Individual scene content specialist.

    The Scene Smith drafts actual scene prose from Plotwright briefs,
    maintaining style register, creating contrastive choices, and keeping
    all gates diegetic. They show the world while hiding the gears.

    Key responsibilities:
    - Draft scene content from Plotwright briefs
    - Create contrastive, meaningful choices
    - Maintain style consistency
    - Ensure gates are diegetic (world-based)
    - Keep player-facing content spoiler-safe
    - File hooks for structural/canon needs
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/scene_smith.md"""
        return "scene_smith"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Scene Smith"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a scene smith task.

        Supported tasks:
        - 'draft_scene': Draft scene content from brief
        - 'draft_choices': Create choice options
        - 'rewrite_scene': Revise scene per feedback
        - 'write_gate_scene': Draft a gateway/check scene
        - 'polish_prose': Final polish pass

        Args:
            context: Execution context

        Returns:
            Result with scene content
        """
        task = context.task.lower()
        logger.info("SceneSmith executing task: %s", task)
        logger.trace("Number of artifacts provided: %d", len(context.artifacts))

        if task == "draft_scene":
            logger.debug("Drafting scene from brief")
            return self._draft_scene(context)
        elif task == "draft_choices":
            logger.debug("Drafting choice options")
            return self._draft_choices(context)
        elif task == "rewrite_scene":
            logger.debug("Rewriting scene with feedback")
            return self._rewrite_scene(context)
        elif task == "write_gate_scene":
            logger.debug("Writing gateway/check scene")
            return self._write_gate_scene(context)
        elif task == "polish_prose":
            logger.debug("Polishing scene prose")
            return self._polish_prose(context)
        else:
            logger.warning("Unknown SceneSmith task: %s", task)
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _draft_scene(self, context: RoleContext) -> RoleResult:
        """Draft a scene from Plotwright brief."""
        logger.debug("Drafting scene from brief")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Draft Scene

{self.format_artifacts(context.artifacts)}

Draft a scene based on the provided brief. Follow these principles:

1. **Honor the brief**: Respect goal, stakes, and choice intents
2. **Show, don't tell**: Use concrete details and action
3. **Contrastive choices**: Make options clearly different
4. **Diegetic gates**: Any checks must be world-based (badge, knowledge, item)
5. **Player-safe**: No spoilers or meta references
6. **Register**: Match the project's style and tone

Scene requirements:
- Opening: Set the scene and context
- Body: Progress toward the scene goal
- Choices: Present meaningful options (2-4)
- Each choice should:
  * Be clearly distinct from others
  * Suggest different consequences
  * Feel natural to the situation

Write in engaging, player-facing prose. Keep it tight and focused.
"""

        try:
            logger.trace("Calling LLM to draft scene")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
            logger.info(
                "Successfully drafted scene, size: %d characters", len(response)
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "scene_draft"},
            )

        except Exception as e:
            logger.error("Error drafting scene: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error drafting scene: {e}",
            )

    def _draft_choices(self, context: RoleContext) -> RoleResult:
        """Draft choice options for an existing scene."""
        logger.debug("Drafting choice options")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Draft Choices

{self.format_artifacts(context.artifacts)}

Draft choice options for the provided scene context. Create 2-4 choices that:

1. **Contrast clearly**: Each option feels meaningfully different
2. **Suggest consequences**: Players can anticipate what might happen
3. **Fit the moment**: Feel natural to the situation
4. **Respect agency**: No false choices or railroading
5. **Stay diegetic**: No meta references ("Try again", "Go back", etc.)

Format each choice as:
- **Choice text**: What the player sees (1-2 sentences max)
- **Intent**: What this choice tests or reveals
- **Likely outcome**: Where it leads (brief)

Focus on quality over quantity - 2 strong choices beat 4 weak ones.
"""

        try:
            logger.trace("Calling LLM to draft choices")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)
            logger.info(
                "Successfully drafted choices, size: %d characters", len(response)
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "choice_draft"},
            )

        except Exception as e:
            logger.error("Error drafting choices: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error drafting choices: {e}",
            )

    def _rewrite_scene(self, context: RoleContext) -> RoleResult:
        """Rewrite a scene based on feedback."""
        logger.debug("Rewriting scene based on feedback")
        feedback = context.additional_context.get("feedback", "")
        feedback_length = len(feedback) if feedback else 0
        logger.trace("Feedback provided: %d characters", feedback_length)

        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Rewrite Scene

{self.format_artifacts(context.artifacts)}

Revise the scene based on this feedback:

{feedback}

Apply the requested changes while maintaining:
- Overall scene structure and flow
- Style and register consistency
- Player-neutral boundaries
- Diegetic expression of gates/checks

Focus on the specific issues raised in the feedback.
"""

        try:
            logger.trace("Calling LLM to rewrite scene")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
            logger.info(
                "Successfully rewrote scene, size: %d characters", len(response)
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "scene_rewrite"},
            )

        except Exception as e:
            logger.error("Error rewriting scene: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error rewriting scene: {e}",
            )

    def _write_gate_scene(self, context: RoleContext) -> RoleResult:
        """Write a scene that implements a gateway/check."""
        logger.debug("Writing gateway/check scene")
        gate_type = context.additional_context.get("gate_type", "unknown")
        logger.trace("Gate type: %s", gate_type)

        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Write Gateway Scene

{self.format_artifacts(context.artifacts)}

Write a scene that implements a gateway check: {gate_type}

CRITICAL: The gate must be **diegetic** (world-based), not meta or mechanical.

Good gates:
- "You need the guard's badge to enter" (item check)
- "You recognize the ancient script from your studies" (knowledge check)
- "The merchant remembers you helped them" (reputation check)

Bad gates:
- "You need 3 strength to open this" (mechanical stat)
- "Come back when you're level 5" (meta progression)
- "You must complete quest X first" (meta dependency)

Write the gate as a natural part of the scene. Show what happens if the
player meets the requirement, and provide a diegetic reason if they don't.
"""

        try:
            logger.trace("Calling LLM to write gate scene")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)
            logger.info(
                "Successfully wrote gate scene for %s gate, size: %d characters",
                gate_type,
                len(response),
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "gate_scene", "gate_type": gate_type},
            )

        except Exception as e:
            logger.error(
                "Error writing gate scene (%s): %s", gate_type, e, exc_info=True
            )
            return RoleResult(
                success=False,
                output="",
                error=f"Error writing gate scene: {e}",
            )

    def _polish_prose(self, context: RoleContext) -> RoleResult:
        """Final polish pass on scene prose."""
        logger.debug("Polishing scene prose")
        system_prompt = self.build_system_prompt(context)

        user_prompt = f"""# Task: Polish Prose

{self.format_artifacts(context.artifacts)}

Perform a final polish pass on this scene. Focus on:

1. **Clarity**: Remove ambiguity, tighten phrasing
2. **Flow**: Smooth transitions between beats
3. **Register**: Consistent tone and voice
4. **Engagement**: Strong verbs, vivid details
5. **Accessibility**: Clear sentence structure
6. **Player safety**: No leaked spoilers

This is a polish pass, not a rewrite. Preserve the scene's structure
and choices while improving the prose quality.
"""

        try:
            logger.trace("Calling LLM to polish prose")
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
            logger.info(
                "Successfully polished prose, size: %d characters", len(response)
            )

            return RoleResult(
                success=True,
                output=response,
                metadata={"content_type": "scene_polish"},
            )

        except Exception as e:
            logger.error("Error polishing prose: %s", e, exc_info=True)
            return RoleResult(
                success=False,
                output="",
                error=f"Error polishing prose: {e}",
            )
