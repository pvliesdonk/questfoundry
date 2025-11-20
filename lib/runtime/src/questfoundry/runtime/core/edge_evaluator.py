"""
Edge Evaluator - evaluates conditional edges to determine routing.

Based on spec: components/edge_evaluator.md
STRICT component - condition evaluation correctness is critical.
Security is paramount - never execute untrusted code.
"""

import logging
from typing import Any, Callable, Dict, Optional

from asteval import Interpreter
from json_logic import jsonLogic

from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


class EdgeEvaluator:
    """Evaluate conditional edges based on state to determine next node."""

    def evaluate_condition(
        self,
        condition: Dict[str, Any],
        state: StudioState
    ) -> bool:
        """
        Evaluate condition based on evaluator type.

        Args:
            condition: Condition object from edge
            state: Current loop state

        Returns:
            True if condition met, False otherwise

        Raises:
            ValueError: If evaluator type is unknown
        """
        evaluator_type = condition.get("evaluator")

        if evaluator_type == "python_expression":
            expr = condition.get("expression", "")
            return self.evaluate_python_expression(expr, state)

        elif evaluator_type == "json_logic":
            rules = condition.get("rules", {})
            return self.evaluate_json_logic(rules, state)

        elif evaluator_type == "bar_threshold":
            bars = condition.get("bars_checked", [])
            threshold = condition.get("threshold", "all_green")
            return self.evaluate_bar_threshold(bars, threshold, state)

        else:
            raise ValueError(f"Unknown evaluator: {evaluator_type}")

    def evaluate_python_expression(
        self,
        expression: str,
        state: StudioState
    ) -> bool:
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
            # Create safe interpreter
            aeval = Interpreter(
                usersyms=None,  # No user symbols
                builtins_={}  # No builtins by default
            )

            # Add safe context variables
            safe_context = {
                "state": state,
                "artifacts": state.get("artifacts", {}),
                "quality_bars": state.get("quality_bars", {}),
                "tu_lifecycle": state.get("tu_lifecycle"),
                "error": state.get("error"),
                "retry_count": state.get("retry_count", 0),
                "messages": state.get("messages", []),
                "loop_context": state.get("loop_context", {})
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

            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate expression: {expression}")
            logger.warning(f"Error: {e}")
            return False

    def evaluate_json_logic(
        self,
        rules: Dict[str, Any],
        state: StudioState
    ) -> bool:
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
                "message_count": len(state.get("messages", []))
            }

            # Apply rules
            result = jsonLogic(rules, data)
            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate JSON logic rules: {rules}")
            logger.warning(f"Error: {e}")
            return False

    def evaluate_bar_threshold(
        self,
        bars_checked: list[str],
        threshold: str,
        state: StudioState
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
            f"Bar threshold check ({threshold}): {bars_checked} → {result} "
            f"(statuses: {statuses})"
        )
        return result

    def create_routing_function(
        self,
        edge: Dict[str, Any],
        max_retries: int = 3
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
        def routing_function(state: StudioState) -> str:
            condition = edge.get("condition", {})
            source = edge.get("source")
            target = edge.get("target")

            try:
                # Evaluate condition
                condition_met = self.evaluate_condition(condition, state)

                if condition_met:
                    # Condition passed, go to target
                    return target

                else:
                    # Condition failed, check retry count
                    retry_count = state.get("retry_count", 0)
                    if retry_count < max_retries:
                        # Loop back to source for rework
                        return source
                    else:
                        # Max retries exceeded, exit
                        return "__end__"

            except Exception as e:
                logger.error(f"Routing function error: {e}")
                # On error, exit to prevent infinite loops
                return "__end__"

        return routing_function
