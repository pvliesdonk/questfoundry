"""DREAM stage implementation.

The DREAM stage establishes the creative vision for the story,
generating genre, tone, themes, and style direction.

Both interactive and direct modes use the same 3-phase pattern:
Discuss → Summarize → Serialize.

- Interactive: Multi-turn discussion with user, then summarize and serialize
- Direct: Single-turn discuss (no user), then summarize and serialize
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.conversation import ConversationRunner, ValidationResult
from questfoundry.observability.logging import get_logger
from questfoundry.tools import SubmitDreamTool

log = get_logger(__name__)

if TYPE_CHECKING:
    from questfoundry.prompts import PromptCompiler
    from questfoundry.providers import LLMProvider
    from questfoundry.providers.base import Message
    from questfoundry.tools import Tool


class DreamStage:
    """DREAM stage - establish creative vision.

    This stage takes a user's story idea and generates a creative
    vision artifact containing genre, tone, themes, and style direction.

    Both interactive and direct modes follow the same 3-phase pattern,
    differing only in configuration:
    - Interactive: max_discuss_turns=10, user_input_fn provided
    - Direct: max_discuss_turns=1, user_input_fn=None

    Attributes:
        name: Stage identifier ("dream").
    """

    name = "dream"

    async def execute(
        self,
        context: dict[str, Any],
        provider: LLMProvider,
        compiler: PromptCompiler,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the DREAM stage using the 3-phase pattern.

        Both interactive and direct modes use the same code path:
        Discuss → Summarize → Serialize.

        Args:
            context: Context with keys:
                - user_prompt: The user's story idea (required)
                - interactive: Enable multi-turn discussion (default: False)
                - user_input_fn: Async function to get user input (optional)
                - research_tools: Additional tools for context (optional)
                - on_assistant_message: Callback for assistant messages (optional)
                - max_turns: Max discussion turns (default: 10)
                - validation_retries: Max validation retries (default: 3)
            provider: LLM provider for completions.
            compiler: Prompt compiler for template assembly.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            ConversationError: If any phase fails.
        """
        interactive = context.get("interactive", False)
        user_prompt = context.get("user_prompt", "")
        user_input_fn = context.get("user_input_fn")
        on_assistant_message = context.get("on_assistant_message")
        research_tools: list[Tool] = context.get("research_tools") or []

        log.debug(
            "dream_execute_start",
            interactive=interactive,
            prompt_length=len(user_prompt),
            research_tools=len(research_tools),
        )

        # Build prompt context
        prompt_context = self._build_prompt_context(user_prompt, research_tools, interactive)

        # Compile prompt
        prompt = compiler.compile("dream", prompt_context)
        log.debug("prompt_compiled", template="dream")

        # Build initial messages
        initial_messages: list[Message] = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # Create conversation runner with unified configuration
        # Direct mode: max_discuss_turns=1, no user_input_fn
        # Interactive mode: max_discuss_turns from context, user_input_fn provided
        runner = ConversationRunner(
            provider=provider,
            research_tools=research_tools,
            finalization_tool=SubmitDreamTool(),
            max_discuss_turns=1 if not interactive else context.get("max_turns", 10),
            validation_retries=context.get("validation_retries", 3),
        )

        # Run the 3-phase conversation
        artifact_data, state = await runner.run(
            initial_messages=initial_messages,
            user_input_fn=user_input_fn if interactive else None,
            validator=self._validate_dream,
            summary_prompt=self._get_summary_prompt(),
            on_assistant_message=on_assistant_message,
        )

        # Add required fields
        artifact_data["type"] = "dream"
        artifact_data["version"] = artifact_data.get("version", 1)

        return artifact_data, state.llm_calls, state.tokens_used

    def _build_prompt_context(
        self,
        user_prompt: str,
        research_tools: list[Tool],
        interactive: bool,
    ) -> dict[str, str]:
        """Build the prompt context for the DREAM stage.

        Args:
            user_prompt: The user's story idea.
            research_tools: Available research tools.
            interactive: Whether interactive mode is enabled.

        Returns:
            Context dict for prompt compilation.
        """
        # Build research tools section if any are available
        research_tools_section = self._build_research_tools_section(research_tools)

        if interactive:
            mode_instructions = (
                "IMPORTANT: Engage in a creative discussion with the user to refine "
                "the creative vision. Ask clarifying questions about:\n"
                "- What genre and tone they envision\n"
                "- The target audience and themes\n"
                "- Any content to include or avoid\n"
                "- The desired scope and complexity\n\n"
                f"{research_tools_section}"
                "When the discussion is complete, call ready_to_summarize() to signal "
                "that you're ready to move to the summarization phase. The user can also "
                "type /done to signal they're ready."
            )
            mode_reminder = (
                "REMEMBER: Discuss the vision with the user first. Call ready_to_summarize() "
                "when you have refined the concept together and are ready to proceed."
            )
        else:
            mode_instructions = (
                "Generate a creative vision for the story idea provided. "
                f"{research_tools_section}"
                "Consider the key elements: genre, tone, themes, audience, scope, and style."
            )
            mode_reminder = ""

        return {
            "mode_instructions": mode_instructions,
            "mode_reminder": mode_reminder,
            "user_message": f"I'd like to create an interactive story. Here's my idea:\n\n{user_prompt}",
        }

    def _build_research_tools_section(self, tools: list[Tool]) -> str:
        """Build prompt section describing available research tools.

        Args:
            tools: List of research tools.

        Returns:
            Formatted string describing tools, or empty string if none.
        """
        if not tools:
            return ""

        # Build tool descriptions
        tool_lines = []
        for tool in tools:
            defn = tool.definition
            tool_lines.append(f"- {defn.name}: {defn.description}")

        return (
            "You have research tools available to gather information:\n"
            + "\n".join(tool_lines)
            + "\n\n"
            "Use these tools to research genre conventions, writing craft, or current trends "
            "that might inform the creative vision.\n\n"
        )

    def _get_summary_prompt(self) -> str:
        """Get the summary prompt for the DREAM stage.

        Returns:
            Prompt text for the Summarize phase.
        """
        return (
            "Summarize the creative vision we've developed. Include the key decisions about:\n"
            "- Genre and subgenre\n"
            "- Tone and atmosphere\n"
            "- Target audience\n"
            "- Core themes\n"
            "- Scope and complexity\n"
            "- Any content notes\n\n"
            "Be concise but capture the essential creative direction."
        )

    def _validate_dream(self, data: dict[str, Any]) -> ValidationResult:
        """Validate dream artifact data using Pydantic model.

        Uses the DreamArtifact model for full validation, returning
        structured error details for the LLM to correct.

        Args:
            data: Artifact data to validate.

        Returns:
            ValidationResult with valid=True if data passes validation,
            or valid=False with structured errors list and expected_fields.
        """
        from pydantic import ValidationError

        from questfoundry.artifacts import (
            DreamArtifact,
            get_all_field_paths,
            pydantic_errors_to_details,
        )

        # Get all expected field paths (including nested) for unknown field detection
        expected_fields = get_all_field_paths(DreamArtifact)

        try:
            # Validate using Pydantic model
            validated = DreamArtifact.model_validate(data)
            return ValidationResult(valid=True, data=validated.model_dump())
        except ValidationError as e:
            # Convert to structured error details
            error_details = pydantic_errors_to_details(e.errors(), data)

            # Also provide legacy error string for backwards compatibility
            error_strings = [f"{err.field}: {err.issue}" for err in error_details]

            return ValidationResult(
                valid=False,
                error=f"Validation errors: {'; '.join(error_strings)}",
                errors=error_details,
                expected_fields=expected_fields,
            )


# Create singleton instance for registration
dream_stage = DreamStage()
