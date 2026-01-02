"""DREAM stage implementation.

The DREAM stage establishes the creative vision for the story,
generating genre, tone, themes, and style direction.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from questfoundry.prompts import PromptCompiler
    from questfoundry.providers import LLMProvider


class DreamParseError(Exception):
    """Raised when DREAM stage response cannot be parsed."""

    def __init__(self, message: str, raw_content: str) -> None:
        self.raw_content = raw_content
        super().__init__(message)


class DreamStage:
    """DREAM stage - establish creative vision.

    This stage takes a user's story idea and generates a creative
    vision artifact containing genre, tone, themes, and style direction.
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
            context: Context with 'user_prompt' key containing the story idea.
            provider: LLM provider for completions.
            compiler: Prompt compiler for template assembly.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            DreamParseError: If the LLM response cannot be parsed as YAML.
        """
        # Get user prompt from context
        user_prompt = context.get("user_prompt", "")

        # Compile the dream prompt template
        prompt = compiler.compile("dream", {"user_prompt": user_prompt})

        # Call LLM
        response = await provider.complete(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )

        # Parse YAML from response
        artifact_data = self._parse_response(response.content)

        # Ensure required fields
        artifact_data["type"] = "dream"
        artifact_data["version"] = artifact_data.get("version", 1)

        return artifact_data, 1, response.tokens_used

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
                if stripped and not self._is_yaml_line(stripped):
                    break
                yaml_lines.append(line)

        return "\n".join(yaml_lines).strip()

    def _is_yaml_line(self, line: str) -> bool:
        """Check if a line looks like YAML content.

        Args:
            line: Stripped line to check.

        Returns:
            True if line appears to be YAML.
        """
        # YAML patterns: key:, - item, continuation indent
        # Key pattern: word characters followed by colon (not just ":" anywhere)
        if re.match(r"^\w[\w\s]*:", line):
            return True
        # List item: dash followed by space
        if line.startswith("- "):
            return True
        # Continuation: starts with whitespace
        return line.startswith(" ") or line.startswith("\t")


# Create singleton instance for registration
dream_stage = DreamStage()
