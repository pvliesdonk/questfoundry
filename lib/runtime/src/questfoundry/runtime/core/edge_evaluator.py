"""
Edge Evaluator - evaluates conditional edges to determine routing.

Based on spec: components/edge_evaluator.md
STRICT component - condition evaluation correctness is critical.
Security is paramount - never execute untrusted code.
"""

import logging
from collections.abc import Callable
from typing import Any

from asteval import Interpreter
from json_logic import jsonLogic

from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


class EdgeEvaluator:
    """Evaluate conditional edges based on state to determine next node."""

    def evaluate_condition(self, condition: dict[str, Any], state: StudioState) -> Any:
        """Evaluate condition based on evaluator type.

        Returns raw evaluation output so routing logic can branch on
        booleans or explicit route values.

        Raises:
            ValueError: If evaluator type is unknown
        """
        evaluator_type = condition.get("evaluator")

        if evaluator_type == "python_expression":
            expr = condition.get("expression", "")
            return self.evaluate_python_expression(expr, state)

        if evaluator_type == "json_logic":
            rules = condition.get("rules", {})
            return self.evaluate_json_logic(rules, state)

        if evaluator_type == "bar_threshold":
            bars = condition.get("bars_checked", [])
            threshold = condition.get("threshold", "all_green")
            return self.evaluate_bar_threshold(bars, threshold, state)

        if evaluator_type == "state_key_match":
            # If an expression is provided, evaluate it directly (common in spec files)
            expr = condition.get("expression")
            if expr:
                return self.evaluate_python_expression(expr, state)

            state_key = condition.get("state_key")
            expected = condition.get("expected_value", True)
            actual = self._get_nested_value(state, state_key)
            return actual == expected

        if evaluator_type == "artifact_field_match":
            expr = condition.get("expression")
            if expr:
                return self.evaluate_python_expression(expr, state, coerce_bool=False)

            artifact_path = condition.get("artifact_path") or condition.get("state_key")
            expected = condition.get("expected_value")
            actual = self._get_nested_value(state.get("artifacts", {}), artifact_path)
            if expected is None:
                return actual
            return actual == expected

        if evaluator_type == "quality_bar_status":
            bar = condition.get("bar") or condition.get("quality_bar")
            expected_status = condition.get("status") or condition.get("expected_status")
            if not bar:
                logger.warning("quality_bar_status evaluator missing 'bar'")
                return False
            bar_status = state.get("quality_bars", {}).get(bar, {}).get("status")
            if expected_status:
                return bar_status == expected_status
            return bar_status

        raise ValueError(f"Unknown evaluator: {evaluator_type}")

    def evaluate_python_expression(
        self, expression: str, state: StudioState, coerce_bool: bool = True
    ) -> Any:
        """
        Safely evaluate Python expression against state using asteval.

        CRITICAL: Must use safe evaluation to prevent code injection.
        Uses asteval library which is safer than built-in eval().

        Args:
            expression: Python expression string
            state: Current state context

        Returns:
            Boolean result of expression evaluation

        Raises:
            Exception: If expression is unsafe or invalid
        """
        try:
            # Create safe interpreter with minimal builtins
            aeval = Interpreter(
                usersyms={},
                builtins_={"True": True, "False": False, "None": None},
            )

            # Wrap state to allow attribute-style access used in spec expressions
            state_proxy = self._wrap_state(state)

            # Add safe context variables
            safe_context = {
                "state": state_proxy,
                "artifacts": state_proxy.get("artifacts", {}),
                "quality_bars": state_proxy.get("quality_bars", {}),
                "tu_lifecycle": state_proxy.get("tu_lifecycle"),
                "error": state_proxy.get("error"),
                "retry_count": state_proxy.get("retry_count", 0),
                "messages": state_proxy.get("messages", []),
                "loop_context": state_proxy.get("loop_context", {}),
                "true": True,
                "false": False,
            }

            # Add to interpreter
            for key, value in safe_context.items():
                aeval.symtable[key] = value

            # Evaluate
            result = aeval(expression)

            # Check for errors
            if aeval.error:
                logger.warning(f"Evaluation error in expression: {expression}")
                logger.warning(f"Error details: {aeval.error}")
                return False

            return bool(result) if coerce_bool else result

        except Exception as e:
            logger.warning(f"Failed to evaluate expression: {expression}")
            logger.warning(f"Error: {e}")
            return False

    def evaluate_json_logic(self, rules: dict[str, Any], state: StudioState) -> bool:
        """
        Evaluate JSON Logic rules against state.

        Uses json-logic-py library for safe evaluation.

        Args:
            rules: JSON Logic rules object
            state: Current state context

        Returns:
            Boolean result of rules evaluation
        """
        try:
            # Prepare data context
            data = {
                "tu_lifecycle": state.get("tu_lifecycle"),
                "artifacts": state.get("artifacts", {}),
                "quality_bars": state.get("quality_bars", {}),
                "error": state.get("error"),
                "retry_count": state.get("retry_count", 0),
                "messages": state.get("messages", []),
                "loop_context": state.get("loop_context", {}),
                "artifacts_count": len(state.get("artifacts", {})),
                "message_count": len(state.get("messages", [])),
            }

            # Apply rules
            result = jsonLogic(rules, data)
            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate JSON logic rules: {rules}")
            logger.warning(f"Error: {e}")
            return False

    def evaluate_bar_threshold(
        self, bars_checked: list[str], threshold: str, state: StudioState
    ) -> bool:
        """
        Evaluate quality bars against threshold.

        Thresholds:
        - all_green: All checked bars must be green
        - mostly_green: ≥75% green, rest yellow (no red)
        - no_red: No red bars allowed
        - any_progress: At least one bar checked

        Args:
            bars_checked: List of bar names to check
            threshold: Threshold type
            state: Current state with quality_bars

        Returns:
            True if threshold met, False otherwise

        Raises:
            ValueError: If threshold is unknown
        """
        # Get statuses for checked bars
        statuses = []
        for bar_name in bars_checked:
            if bar_name in state.get("quality_bars", {}):
                bar_status = state["quality_bars"][bar_name].get("status", "not_checked")
                statuses.append(bar_status)

        if not statuses:
            logger.warning(f"No bars found for checking: {bars_checked}")
            return False

        # Evaluate based on threshold
        if threshold == "all_green":
            result = all(s == "green" for s in statuses)

        elif threshold == "mostly_green":
            green_count = sum(1 for s in statuses if s == "green")
            red_count = sum(1 for s in statuses if s == "red")
            result = red_count == 0 and green_count >= len(statuses) * 0.75

        elif threshold == "no_red":
            result = all(s != "red" for s in statuses)

        elif threshold == "any_progress":
            result = any(s != "not_checked" for s in statuses)

        else:
            raise ValueError(f"Unknown threshold: {threshold}")

        logger.debug(
            f"Bar threshold check ({threshold}): {bars_checked} → {result} (statuses: {statuses})"
        )
        return result

    @staticmethod
    def _get_nested_value(data: Any, path: str | None) -> Any:
        """Safely navigate dotted paths in nested dicts."""

        if not path:
            return None

        current: Any = data
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _wrap_state(self, data: Any) -> Any:
        """Recursively wrap mappings to allow attribute-style access in expressions."""

        if isinstance(data, dict):
            return _AttrDict({k: self._wrap_state(v) for k, v in data.items()})
        if isinstance(data, list):
            return [self._wrap_state(v) for v in data]
        return data

    def create_routing_function(
        self, edge: dict[str, Any], max_retries: int = 3
    ) -> Callable[[StudioState], str]:
        """
        Create routing function for LangGraph conditional edge.

        Returns a function that:
        1. Evaluates condition
        2. Returns next node ID based on result

        Args:
            edge: Edge definition with source, target, condition
            max_retries: Maximum retry attempts

        Returns:
            Routing function that takes state and returns next node ID
        """

        routes = edge.get("condition", {}).get("routes", {}) if edge.get("condition") else {}
        default_route = (
            edge.get("condition", {}).get("default_route") if edge.get("condition") else None
        )

        def normalize_target(target_name: str | Any) -> str:
            if target_name == END:
                return END
            if isinstance(target_name, str) and target_name.upper() == "END":
                return END
            return target_name

        def routing_function(state: StudioState) -> str:
            condition = edge.get("condition", {})
            source = edge.get("source")
            target = edge.get("target")

            try:
                result = self.evaluate_condition(condition, state)

                if routes:
                    route_key = result
                    # Normalise booleans to strings to match YAML keys like "true"/"false"
                    if isinstance(route_key, bool):
                        route_key = str(route_key).lower()

                    mapped_target = routes.get(route_key) or routes.get(str(route_key))
                    if mapped_target:
                        return normalize_target(mapped_target)
                    if default_route:
                        return normalize_target(default_route)

                if bool(result):
                    return target

                retry_count = state.get("retry_count", 0)
                if retry_count < max_retries:
                    return source

                return "__end__"

            except Exception as e:
                logger.error(f"Routing function error: {e}")
                return "__end__"

        return routing_function


class _AttrDict(dict):
    """Dict that supports attribute-style access and recursive wrapping."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple passthrough
        return self.get(item)

    def __getitem__(self, key: Any) -> Any:  # pragma: no cover - simple passthrough
        value = super().__getitem__(key)
        return value
