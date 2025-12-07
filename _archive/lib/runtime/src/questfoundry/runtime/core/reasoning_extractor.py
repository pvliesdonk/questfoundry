"""
Reasoning Extractor - captures agent reasoning from LLM responses.

Extracts and classifies reasoning text from AIMessage content that precedes tool calls.
Supports structured logging to qf.reasoning domain.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ReasoningExtractor:
    """Extract and classify agent reasoning from LLM responses."""

    # Reasoning type classification patterns
    REASONING_PATTERNS = {
        "tool_decision": [
            r"(?i)(I (?:will|need to|should|must|can)) (?:use|call|invoke|execute|run)",
            r"(?i)(?:let me|I'll|I'm going to) (?:use|call|invoke|execute|run)",
            r"(?i)(?:using|calling|invoking) the .+ tool",
            r"(?i)(?:first|next|then|now),? (?:I will|I'll|let me)",
        ],
        "state_interpretation": [
            r"(?i)(?:the current state|hot_sot|cold_sot) (?:shows|indicates|contains|has)",
            r"(?i)(?:based on|looking at|examining|checking) (?:the )?(?:state|hot_sot|cold_sot)",
            r"(?i)(?:I see|I notice|I observe) (?:that |the )?(?:state|hot_sot)",
        ],
        "error_recovery": [
            r"(?i)(?:validation failed|error|issue|problem) .+ (?:need to|should|must) (?:fix|correct|retry)",
            r"(?i)(?:let me|I'll|I need to) (?:fix|correct|retry|address) .+ (?:error|issue|validation)",
            r"(?i)(?:the|this) (?:error|validation|issue) .+ (?:suggests|indicates)",
        ],
        "task_decomposition": [
            r"(?i)(?:I will|let me|I'll) (?:break|split|divide) .+ into",
            r"(?i)(?:first|second|third|next|then|finally),? (?:I will|I'll|let me)",
            r"(?i)(?:step \d+|phase \d+|part \d+):",
            r"(?i)(?:the|this) (?:task|goal|objective) (?:requires|needs)",
        ],
        "decision_point": [
            r"(?i)(?:I (?:will|should|need to|must)) (?:decide|choose|determine|consider)",
            r"(?i)(?:alternatives?|options?) (?:are|include|would be)",
            r"(?i)(?:weighing|considering|evaluating) .+ (?:vs|versus|against)",
            r"(?i)(?:given|based on) .+ (?:I will|I'll|let me)",
        ],
    }

    def __init__(self):
        """Initialize reasoning extractor with compiled patterns."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for reasoning_type, patterns in self.REASONING_PATTERNS.items():
            self._compiled_patterns[reasoning_type] = [re.compile(p) for p in patterns]

    def extract_reasoning(
        self, message_content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> dict[str, Any] | None:
        """
        Extract reasoning from message content.

        Args:
            message_content: The AIMessage text content
            tool_calls: Optional list of tool calls following the reasoning

        Returns:
            Dict with:
                - reasoning_text: The extracted reasoning text
                - reasoning_type: Classification of reasoning
                - tool_calls: Names of tools being called
                - confidence: Confidence in classification (0.0-1.0)
            Or None if no reasoning detected
        """
        if not message_content or not isinstance(message_content, str):
            return None

        # Extract reasoning text (text before tool calls or entire message)
        reasoning_text = self._extract_reasoning_text(message_content)
        if not reasoning_text or len(reasoning_text.strip()) < 10:
            # Too short to be meaningful reasoning
            return None

        # Classify reasoning type
        reasoning_type, confidence = self._classify_reasoning(reasoning_text)

        # Extract tool call names
        tool_names = []
        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict) and "name" in call:
                    tool_names.append(call["name"])
                elif hasattr(call, "name"):
                    tool_names.append(call.name)

        return {
            "reasoning_text": reasoning_text.strip(),
            "reasoning_type": reasoning_type,
            "tool_calls": tool_names,
            "confidence": confidence,
        }

    def _extract_reasoning_text(self, content: str) -> str:
        """
        Extract reasoning text from message content.

        Heuristics:
        - Text before explicit tool call markers (e.g., "Using tool X:")
        - Text before code blocks or JSON structures
        - Entire message if no obvious boundary

        Args:
            content: Message content

        Returns:
            Extracted reasoning text
        """
        # Look for explicit tool call markers
        tool_markers = [
            r"<tool[s]?[_-]?call[s]?>",  # <tool_call>, <tools_call>, etc.
            r"\[tool[s]?[_-]?call[s]?\]",  # [tool_call], etc.
            r"(?i)using (?:the )?(?:\w+\s+)?tool:",
            r"(?i)calling (?:the )?(?:\w+\s+)?tool:",
        ]

        earliest_marker = len(content)
        for pattern in tool_markers:
            match = re.search(pattern, content)
            if match:
                earliest_marker = min(earliest_marker, match.start())

        # Look for code blocks (```...```)
        code_block_match = re.search(r"```", content)
        if code_block_match:
            earliest_marker = min(earliest_marker, code_block_match.start())

        # Extract text before marker
        if earliest_marker < len(content):
            reasoning = content[:earliest_marker]
        else:
            reasoning = content

        return reasoning.strip()

    def _classify_reasoning(self, reasoning_text: str) -> tuple[str, float]:
        """
        Classify reasoning text into a category.

        Args:
            reasoning_text: The reasoning text to classify

        Returns:
            Tuple of (reasoning_type, confidence)
        """
        # Count matches per category
        category_scores: dict[str, int] = {}

        for reasoning_type, patterns in self._compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(reasoning_text):
                    score += 1
            if score > 0:
                category_scores[reasoning_type] = score

        if not category_scores:
            # No pattern matches - generic reasoning
            return "general", 0.3

        # Find category with highest score
        max_category = max(category_scores, key=category_scores.get)  # type: ignore
        max_score = category_scores[max_category]

        # Calculate confidence based on:
        # - Number of pattern matches (more = higher confidence)
        # - Whether multiple categories matched (fewer = higher confidence)
        num_categories = len(category_scores)
        confidence = min(1.0, (max_score * 0.3) + (1.0 - (num_categories * 0.1)))

        return max_category, confidence

    def format_for_logging(
        self,
        reasoning: dict[str, Any],
        role_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Format extracted reasoning for structured logging.

        Args:
            reasoning: Extracted reasoning dict
            role_id: ID of the role that produced the reasoning
            context: Additional context (tu_id, loop_id, etc.)

        Returns:
            Formatted log entry
        """
        log_entry = {
            "event": "reasoning.captured",
            "role": role_id,
            "reasoning_type": reasoning["reasoning_type"],
            "reasoning_text": reasoning["reasoning_text"],
            "tool_calls": reasoning["tool_calls"],
            "confidence": reasoning["confidence"],
        }

        if context:
            log_entry["context"] = context

        return log_entry
