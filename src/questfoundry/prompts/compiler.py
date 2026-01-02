"""Prompt compiler for assembling stage prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import Any

from questfoundry.prompts.loader import PromptLoader, TemplateNotFoundError


@dataclass
class CompiledPrompt:
    """A compiled prompt ready for LLM submission."""

    system: str
    user: str
    token_count: int
    template_name: str
    included_components: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Alias for token_count for clarity."""
        return self.token_count


class PromptCompileError(Exception):
    """Raised when prompt compilation fails."""

    def __init__(self, template_name: str, message: str) -> None:
        self.template_name = template_name
        super().__init__(f"Failed to compile template '{template_name}': {message}")


class BudgetExceededError(Exception):
    """Raised when compiled prompt exceeds token budget."""

    def __init__(self, required: int, budget: int, template_name: str) -> None:
        self.required = required
        self.budget = budget
        self.template_name = template_name
        super().__init__(
            f"Prompt '{template_name}' requires {required} tokens but budget is {budget}"
        )


class PromptCompiler:
    """Compile prompts from templates with variable substitution.

    This is a simplified implementation for Slice 1:
    - Simple {{ variable }} substitution (no Jinja2)
    - No compression strategies (all content must fit)
    - Token counting with tiktoken, fallback to heuristic

    Attributes:
        prompts_path: Path to the prompts directory.
        token_budget: Maximum tokens for compiled prompts.
    """

    # Pattern for {{ variable }} substitution
    _VAR_PATTERN = re.compile(r"\{\{\s*(\w+(?:\.\w+)*)\s*\}\}")

    def __init__(
        self,
        prompts_path: Path,
        token_budget: int = 4000,
    ) -> None:
        """Initialize the prompt compiler.

        Args:
            prompts_path: Path to the prompts directory containing templates/.
            token_budget: Maximum tokens for compiled prompts.
        """
        self.prompts_path = prompts_path
        self.token_budget = token_budget
        self._loader = PromptLoader(prompts_path)
        self._tokenizer: Any | None = None

    def _get_tokenizer(self) -> Any:
        """Lazy-load tiktoken encoder."""
        if self._tokenizer is None:
            try:
                import tiktoken

                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._tokenizer = None
        return self._tokenizer

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses tiktoken if available, otherwise falls back to
        character heuristic (4 chars ≈ 1 token).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        # Fallback: ~4 characters per token
        return len(text) // 4

    def _resolve_variable(self, path: str, context: dict[str, Any]) -> str:
        """Resolve a dotted variable path from context.

        Args:
            path: Dotted path like 'user_prompt' or 'dream.genre'.
            context: Context dictionary.

        Returns:
            String representation of the value.

        Raises:
            KeyError: If the path cannot be resolved.
        """
        parts = path.split(".")
        value: Any = context

        for part in parts:
            if isinstance(value, dict):
                if part not in value:
                    raise KeyError(f"Key '{part}' not found in context path '{path}'")
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise KeyError(f"Cannot resolve '{part}' in context path '{path}'")

        # Convert to string
        if isinstance(value, (list, dict)):
            # Format lists/dicts nicely
            return json.dumps(value, indent=2)
        return str(value)

    def _substitute_variables(self, text: str, context: dict[str, Any]) -> str:
        """Substitute {{ variable }} placeholders with context values.

        Args:
            text: Text containing placeholders.
            context: Context dictionary for substitution.

        Returns:
            Text with placeholders replaced.
        """

        def replace_match(match: re.Match[str]) -> str:
            path = match.group(1)
            try:
                return self._resolve_variable(path, context)
            except KeyError:
                # Leave unresolved variables as-is
                return match.group(0)

        return self._VAR_PATTERN.sub(replace_match, text)

    def compile(
        self,
        template_name: str,
        context: dict[str, Any] | None = None,
    ) -> CompiledPrompt:
        """Compile a prompt from a template with context substitution.

        Args:
            template_name: Name of the template (e.g., 'dream').
            context: Context dictionary for variable substitution.

        Returns:
            CompiledPrompt ready for LLM submission.

        Raises:
            PromptCompileError: If template cannot be loaded or compiled.
            BudgetExceededError: If compiled prompt exceeds token budget.
        """
        context = context or {}

        # Load template
        try:
            template = self._loader.load(template_name)
        except TemplateNotFoundError as e:
            raise PromptCompileError(template_name, str(e)) from e

        # Substitute variables
        system = self._substitute_variables(template.system, context)
        user = self._substitute_variables(template.user, context)

        # Calculate token count
        token_count = self.estimate_tokens(system) + self.estimate_tokens(user)

        # Check budget
        if token_count > self.token_budget:
            raise BudgetExceededError(token_count, self.token_budget, template_name)

        return CompiledPrompt(
            system=system,
            user=user,
            token_count=token_count,
            template_name=template_name,
            included_components=template.components,
        )

    def list_templates(self) -> list[str]:
        """List available template names.

        Returns:
            List of template names that can be compiled.
        """
        return self._loader.list_templates()
