"""LLM call helper for the POLISH stage.

Provides _PolishLLMHelperMixin with _polish_llm_call(), adapting the
proven GROW pattern (template loading → context injection → structured
output → validation retry) for POLISH's phases.

PolishStage inherits this mixin so all LLM phase methods can call
``self._polish_llm_call(...)`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.stages.polish._helpers import (
    PolishStageError,
    log,
)
from questfoundry.prompts.compiler import safe_format
from questfoundry.providers.structured_output import (
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

T = TypeVar("T", bound=BaseModel)


def _get_prompts_path() -> Path:
    """Get the prompts directory path.

    Returns prompts from package first, then falls back to project root.
    """
    pkg_path = Path(__file__).parents[5] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


class _PolishLLMHelperMixin:
    """Mixin providing LLM call wrapper for POLISH phases.

    Expects the host class to set the following attributes in ``__init__``
    or ``execute()``:

    - ``_serialize_model``
    - ``_serialize_provider_name``
    - ``_provider_name``
    - ``_callbacks``
    """

    @traceable(name="POLISH LLM Call", run_type="llm", tags=["stage:polish"])
    async def _polish_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        Args:
            model: LangChain chat model.
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            PolishStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        # Build messages from template with context injection
        system_text = safe_format(template.system, context) if context else template.system
        user_text = (
            safe_format(template.user, context) if template.user and context else template.user
        )

        effective_model = self._serialize_model or model  # type: ignore[attr-defined]
        effective_provider = self._serialize_provider_name or self._provider_name  # type: ignore[attr-defined]
        structured_model = with_structured_output(
            effective_model, output_schema, provider_name=effective_provider
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        # Build config with callbacks for LLM call logging
        config = build_runnable_config(
            run_name=f"polish_{template_name}",
            metadata={"stage": "polish", "phase": template_name},
            callbacks=self._callbacks,  # type: ignore[attr-defined]
        )

        llm_calls = 0
        total_tokens = 0
        base_messages = list(messages)

        for attempt in range(max_retries):
            log.debug(
                "polish_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            try:
                raw_result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(raw_result)

                result = unwrap_structured_result(raw_result)
                validated = (
                    result
                    if isinstance(result, output_schema)
                    else output_schema.model_validate(result)
                )
                log.debug("polish_llm_validation_pass", template=template_name)

                return validated, llm_calls, total_tokens

            except (ValidationError, TypeError) as e:
                log.warning(
                    "polish_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    error_msg = _build_error_feedback(e, output_schema)
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise PolishStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )


def _build_error_feedback(error: Exception, output_schema: type[BaseModel]) -> str:
    """Build structured error feedback for LLM retry.

    Converts validation errors into field-level feedback the LLM can
    act on, including the list of required fields.

    Args:
        error: The validation or type error from parsing.
        output_schema: The Pydantic model class expected.

    Returns:
        Formatted error feedback string for the LLM.
    """
    if isinstance(error, ValidationError):
        lines: list[str] = []
        for e in error.errors():
            loc = ".".join(str(p) for p in e["loc"]) or "(root)"
            lines.append(f"  - {loc}: {e['msg']}")
        required_fields = ", ".join(sorted(get_all_field_paths(output_schema)))
        return (
            "Validation errors in your response:\n"
            + "\n".join(lines)
            + f"\n\nRequired fields: {required_fields}"
            + "\nEnsure all IDs are from the Valid IDs list."
        )
    return f"Error: {error}\n\nPlease produce valid output matching the expected schema."
