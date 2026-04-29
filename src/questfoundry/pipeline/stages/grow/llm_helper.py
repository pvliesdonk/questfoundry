"""LLM call helper and validation utilities for the GROW stage.

Provides _LLMHelperMixin with _grow_llm_call() and error feedback
formatting. GrowStage inherits this mixin so LLM phase methods can call
``self._grow_llm_call(...)`` directly.

Gap-insertion logic lives in ``questfoundry.graph.gap_insertion``. GROW
no longer calls gap-insertion directly (narrative_gaps and pacing_gaps
both moved to POLISH per issue #1368), but the
``_validate_and_insert_gaps`` delegation is retained here so existing
test call sites keep working without churn. ``GapInsertionReport`` is
also re-exported here for back-compat with code that imports it via
``grow.llm_helper``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.graph.gap_insertion import GapInsertionReport, validate_and_insert_gaps
from questfoundry.graph.mutations import GrowValidationError  # noqa: TC001 - used at runtime
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.stages.grow._helpers import (
    GrowStageError,
    T,
    _get_prompts_path,
    log,
)
from questfoundry.prompts.compiler import safe_format
from questfoundry.providers.base import ProviderRateLimitError
from questfoundry.providers.rate_limit import ainvoke_with_rate_limit_retry
from questfoundry.providers.structured_output import (
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph
    from questfoundry.models.grow import GapProposal

# Re-exported for back-compat with existing imports.
__all__ = ["GapInsertionReport", "_LLMHelperMixin"]


class _LLMHelperMixin:
    """Mixin providing LLM call wrapper and gap insertion for GROW phases.

    Expects the host class to set the following attributes in ``__init__``
    or ``execute()``:

    - ``_serialize_model``
    - ``_serialize_provider_name``
    - ``_provider_name``
    - ``_callbacks``
    """

    @traceable(name="GROW LLM Call", run_type="llm", tags=["stage:grow"])
    async def _grow_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
        semantic_validator: Callable[[T], list[GrowValidationError]] | None = None,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        If a semantic_validator is provided, it runs after Pydantic succeeds.
        When >50% of entries have semantic errors, retries the LLM call.
        Otherwise returns the result for the caller to filter.

        Args:
            model: LangChain chat model.
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.
            semantic_validator: Optional callable that checks ID validity.
                Should accept the validated result and return a list of errors.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            GrowStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        # Build system message from template with context injection
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
            run_name=f"grow_{template_name}",
            metadata={"stage": "grow", "phase": template_name},
            callbacks=self._callbacks,  # type: ignore[attr-defined]
        )

        llm_calls = 0
        total_tokens = 0
        base_messages = list(messages)  # Preserve original for retry resets

        for attempt in range(max_retries):
            log.debug(
                "grow_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            try:
                # Rate-limit-aware invocation: 429s back off transparently
                # without consuming this loop's semantic-retry budget (#1581).
                raw_result = await ainvoke_with_rate_limit_retry(
                    structured_model, messages, config=config
                )
                llm_calls += 1
                total_tokens += extract_tokens(raw_result)

                result = unwrap_structured_result(raw_result)
                # Defensive fallback for providers that return dicts instead.
                validated = (
                    result
                    if isinstance(result, output_schema)
                    else output_schema.model_validate(result)
                )
                log.debug("grow_llm_validation_pass", template=template_name)

                # Semantic validation: check IDs exist in graph
                if semantic_validator:
                    from questfoundry.graph.grow_validators import (
                        count_entries,
                        format_semantic_errors,
                    )

                    sem_errors = semantic_validator(validated)
                    if sem_errors:
                        entry_count = count_entries(validated)
                        error_ratio = len(sem_errors) / max(entry_count, 1)
                        log.info(
                            "grow_semantic_validation_fail",
                            template=template_name,
                            errors=len(sem_errors),
                            entries=entry_count,
                            ratio=f"{error_ratio:.0%}",
                        )
                        # Retry when >50% of entries have errors (majority invalid).
                        # Below threshold, return and let caller filter minor hallucinations.
                        if error_ratio > 0.5 and attempt < max_retries - 1:
                            feedback = format_semantic_errors(sem_errors)
                            messages = list(base_messages)
                            messages.append(HumanMessage(content=feedback))
                            continue  # retry
                        # Below threshold or last attempt: return for caller to filter

                return validated, llm_calls, total_tokens

            except ProviderRateLimitError:
                # Backoff exhausted — surface as a hard transport failure,
                # not a schema-repair attempt (#1581).
                raise

            except (ValidationError, TypeError) as e:
                log.info(
                    "grow_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    # Reset to base messages + error feedback to avoid
                    # unbounded message history growth across retries
                    error_msg = self._build_grow_error_feedback(e, output_schema)
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise GrowStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    def _build_grow_error_feedback(self, error: Exception, output_schema: type[BaseModel]) -> str:
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

    def _validate_and_insert_gaps(
        self,
        graph: Graph,
        gaps: list[GapProposal],
        valid_path_ids: set[str] | dict[str, Any],
        valid_beat_ids: set[str] | dict[str, Any],
        phase_name: str,
    ) -> GapInsertionReport:
        """Thin delegation to ``graph.gap_insertion.validate_and_insert_gaps``.

        Retained for back-compat with existing test call sites; GROW phase
        code itself no longer calls this since gap-insertion phases moved
        to POLISH (issue #1368).
        """
        return validate_and_insert_gaps(graph, gaps, valid_path_ids, valid_beat_ids, phase_name)
