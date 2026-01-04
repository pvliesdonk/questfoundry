"""DREAM stage implementation.

The DREAM stage establishes the creative vision for the story,
generating genre, tone, themes, and style direction.

Supports two execution modes:
- Interactive: Conversational refinement with user before finalization
- Direct: Single LLM call with tool-gated output (for testing/automation)

The caller (CLI) determines the mode via the `interactive` context flag.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import yaml

from questfoundry.conversation import ConversationRunner, ValidationResult
from questfoundry.tools import SubmitDreamTool

if TYPE_CHECKING:
    from questfoundry.prompts import PromptCompiler
    from questfoundry.providers import LLMProvider
    from questfoundry.providers.base import Message
    from questfoundry.tools import Tool


class DreamParseError(Exception):
    """Raised when DREAM stage response cannot be parsed."""

    def __init__(self, message: str, raw_content: str) -> None:
        self.raw_content = raw_content
        super().__init__(message)


class DreamStage:
    """DREAM stage - establish creative vision.

    This stage takes a user's story idea and generates a creative
    vision artifact containing genre, tone, themes, and style direction.

    Supports interactive mode (default) where the LLM engages in conversation
    with the user before finalizing the creative vision, and direct mode
    for automated/testing scenarios.

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
        """Execute the DREAM stage.

        Args:
            context: Context with keys:
                - user_prompt: The user's story idea (required)
                - interactive: Enable conversation mode (default: False)
                - user_input_fn: Async function to get user input (optional)
                - research_tools: Additional tools for context (optional)
            provider: LLM provider for completions.
            compiler: Prompt compiler for template assembly.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            DreamParseError: If response cannot be parsed (legacy mode only).
            ConversationError: If conversation fails (interactive mode).
        """
        # Determine interactive mode from context (CLI handles TTY detection)
        interactive = context.get("interactive", False)

        if interactive:
            return await self._execute_interactive(context, provider, compiler)
        else:
            return await self._execute_direct(context, provider, compiler)

    async def _execute_interactive(
        self,
        context: dict[str, Any],
        provider: LLMProvider,
        compiler: PromptCompiler,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute in interactive mode with conversation loop.

        Uses ConversationRunner to manage multi-turn dialogue with the user,
        finalizing via submit_dream tool call.
        """
        user_prompt = context.get("user_prompt", "")
        user_input_fn = context.get("user_input_fn")
        on_assistant_message = context.get("on_assistant_message")
        research_tools: list[Tool] = context.get("research_tools") or []

        # Build prompt context for interactive mode (sandwich pattern)
        prompt_context = {
            "mode_instructions": (
                "IMPORTANT: You have access to the submit_dream tool. Engage in conversation "
                "with the user to refine the creative vision. Ask clarifying questions about:\n"
                "- What genre and tone they envision\n"
                "- The target audience and themes\n"
                "- Any content to include or avoid\n"
                "- The desired scope and complexity\n\n"
                "When the user is satisfied with the direction, call submit_dream() with the "
                "finalized artifact data."
            ),
            "mode_reminder": (
                "REMEMBER: Discuss the vision with the user first. Only call submit_dream() "
                "when you have refined the concept together and the user is ready to proceed."
            ),
            "user_message": f"I'd like to create an interactive story. Here's my idea:\n\n{user_prompt}",
        }

        # Compile prompt for interactive mode
        prompt = compiler.compile("dream", prompt_context)

        # Build tool list: finalization + research tools
        tools: list[Tool] = [SubmitDreamTool(), *research_tools]

        # Create conversation runner
        runner = ConversationRunner(
            provider=provider,
            tools=tools,
            finalization_tool="submit_dream",
            max_turns=context.get("max_turns", 10),
            validation_retries=context.get("validation_retries", 3),
        )

        # Build initial messages
        initial_messages: list[Message] = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # Run conversation
        artifact_data, state = await runner.run(
            initial_messages=initial_messages,
            user_input_fn=user_input_fn,
            validator=self._validate_dream,
            on_assistant_message=on_assistant_message,
        )

        # Add required fields
        artifact_data["type"] = "dream"
        artifact_data["version"] = artifact_data.get("version", 1)

        return artifact_data, state.llm_calls, state.tokens_used

    async def _execute_direct(
        self,
        context: dict[str, Any],
        provider: LLMProvider,
        compiler: PromptCompiler,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute in direct mode with single LLM call.

        Uses tool_choice to force submit_dream call, falling back to
        YAML parsing for legacy provider compatibility.
        """
        user_prompt = context.get("user_prompt", "")

        # Build prompt context for direct mode
        prompt_context = {
            "mode_instructions": (
                "Generate a creative vision for the story idea provided. You have the "
                "submit_dream tool available - call it with the complete artifact data."
            ),
            "mode_reminder": "",
            "user_message": self._build_direct_user_message(user_prompt),
        }

        # Compile prompt for direct mode
        prompt = compiler.compile("dream", prompt_context)

        # Try tool-gated approach first
        submit_tool = SubmitDreamTool()
        response = await provider.complete(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            tools=[submit_tool.definition],
            tool_choice="submit_dream",
        )

        # Extract artifact data from tool call or fall back to YAML parsing
        artifact_data = None
        if response.has_tool_calls and response.tool_calls:
            # Prefer the finalization tool call if present
            for tc in response.tool_calls:
                if tc.name == "submit_dream":
                    artifact_data = tc.arguments
                    break

        # Fallback to YAML parsing for legacy providers or if tool call was missed
        if artifact_data is None:
            artifact_data = self._parse_response(response.content)

        # Validate
        result = self._validate_dream(artifact_data)
        if not result.valid:
            raise DreamParseError(f"Validation failed: {result.error}", str(artifact_data))

        # Add required fields
        artifact_data["type"] = "dream"
        artifact_data["version"] = artifact_data.get("version", 1)

        return artifact_data, 1, response.tokens_used

    def _build_direct_user_message(self, user_prompt: str) -> str:
        """Build user message for direct mode with YAML format spec.

        Includes output format for legacy providers that don't support tools.

        Args:
            user_prompt: The user's story idea.

        Returns:
            Formatted user message with output instructions.
        """
        return f"""Create a creative vision for this story idea:

{user_prompt}

Call submit_dream() with the artifact data, or output valid YAML with these fields:

type: dream
version: 1
genre: <primary genre>
subgenre: <optional refinement>
tone:
  - <tone descriptor>
  - <tone descriptor>
audience: <target audience>
themes:
  - <thematic element>
  - <thematic element>
style_notes: |
  <writing style guidance>
scope:
  target_word_count: <approximate length>
  estimated_passages: <scene count>
  branching_depth: <light, moderate, heavy, or extensive>
content_notes:
  includes:
    - <content to include>
  excludes:
    - <content to avoid>

Be creative but grounded. Make choices that serve the story."""

    def _validate_dream(self, data: dict[str, Any]) -> ValidationResult:
        """Validate dream artifact data using Pydantic model.

        Uses the DreamArtifact model for full validation, providing
        detailed error messages for the LLM to correct.

        Args:
            data: Artifact data to validate.

        Returns:
            ValidationResult with valid=True if data passes validation.
        """
        from pydantic import ValidationError

        from questfoundry.artifacts.models import DreamArtifact

        try:
            # Validate using Pydantic model
            validated = DreamArtifact.model_validate(data)
            return ValidationResult(valid=True, data=validated.model_dump())
        except ValidationError as e:
            # Format errors for LLM feedback
            errors = []
            for error in e.errors():
                loc = ".".join(str(part) for part in error["loc"])
                msg = error["msg"]
                if loc:
                    errors.append(f"{loc}: {msg}")
                else:
                    errors.append(msg)
            return ValidationResult(
                valid=False,
                error=f"Validation errors: {'; '.join(errors)}",
            )

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse YAML artifact from LLM response.

        Handles various response formats:
        - Raw YAML
        - YAML wrapped in ```yaml``` fences
        - YAML with leading/trailing text

        Args:
            content: Raw LLM response content.

        Returns:
            Parsed artifact dictionary.

        Raises:
            DreamParseError: If no valid YAML can be extracted.
        """
        # Try to extract YAML from markdown fences first
        fence_pattern = r"```(?:yaml|yml)?\s*\n(.*?)```"
        fence_match = re.search(fence_pattern, content, re.DOTALL | re.IGNORECASE)

        if fence_match:
            yaml_content = fence_match.group(1).strip()
        else:
            # Try to find YAML-like content (starts with key:)
            # Look for lines starting with common artifact keys
            yaml_content = self._extract_yaml_block(content)

        if not yaml_content:
            raise DreamParseError(
                "No valid YAML found in response",
                content,
            )

        try:
            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                raise DreamParseError(
                    f"Expected YAML dict, got {type(data).__name__}",
                    content,
                )
            return data
        except yaml.YAMLError as e:
            raise DreamParseError(
                f"YAML parse error: {e}",
                content,
            ) from e

    def _extract_yaml_block(self, content: str) -> str:
        """Extract YAML block from mixed content.

        Looks for content starting with common artifact keys like
        'type:', 'genre:', etc. and extracts until end or blank line.

        Args:
            content: Mixed content that may contain YAML.

        Returns:
            Extracted YAML string, or empty string if not found.
        """
        lines = content.split("\n")
        yaml_lines: list[str] = []
        in_yaml = False

        # Common starting keys for dream artifacts
        start_keys = ("type:", "genre:", "version:", "subgenre:", "tone:")

        for line in lines:
            stripped = line.strip()

            # Start capturing when we see a known key
            if not in_yaml and any(stripped.startswith(key) for key in start_keys):
                in_yaml = True

            if in_yaml:
                # Stop at blank lines after content, or obvious non-YAML
                if not stripped and yaml_lines:
                    # Check if next non-blank line is still YAML-like
                    continue
                if stripped.startswith("#") and not yaml_lines:
                    continue
                # Pass original line to preserve indentation for continuation check
                if stripped and not self._is_yaml_line(line):
                    break
                yaml_lines.append(line)

        return "\n".join(yaml_lines).strip()

    def _is_yaml_line(self, line: str) -> bool:
        """Check if a line looks like YAML content.

        Args:
            line: Line to check (may include leading whitespace).

        Returns:
            True if line appears to be YAML.
        """
        # Continuation: starts with whitespace (check first, before stripping)
        if line.startswith(" ") or line.startswith("\t"):
            return True
        # YAML patterns: key:, - item
        stripped = line.strip()
        # Key pattern: word characters followed by colon (not just ":" anywhere)
        if re.match(r"^\w[\w\s]*:", stripped):
            return True
        # List item: dash followed by space
        return stripped.startswith("- ")


# Create singleton instance for registration
dream_stage = DreamStage()
