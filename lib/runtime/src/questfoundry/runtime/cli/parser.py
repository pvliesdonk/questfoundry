"""
CLI Parser - DEPRECATED

This module is deprecated as of Phase 6 (CLI/Runtime Architecture Fix).

The old architecture used deterministic command→loop mappings:
  qf write "scene" → story_spark loop
  qf review story → hook_harvest loop

The new architecture uses natural language interpretation by the Showrunner:
  qf ask "create a scene about X" → Showrunner interprets and decides loops

For backward compatibility, the legacy classes are kept below but should not
be used in new code. Use ShowrunnerInterface.interpret_and_execute() instead.

TODO: Remove in Phase 6C after all references are updated.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """
    DEPRECATED: Legacy class for backward compatibility.

    Use ShowrunnerInterface.interpret_and_execute() instead.
    """
    action: str
    loop_id: str
    context: Dict[str, Any]
    raw: str


class CLIParser:
    """
    DEPRECATED: Legacy parser class.

    The old architecture used this parser to map commands to loops.
    The new architecture uses ShowrunnerInterface for natural language interpretation.
    """

    def __init__(self):
        """Initialize with deprecation warning."""
        logger.warning(
            "CLIParser is deprecated. Use ShowrunnerInterface.interpret_and_execute() instead."
        )

    def parse(self, command_text: str) -> Optional[Command]:
        """
        DEPRECATED: Parse command (no longer used).

        Raises:
            NotImplementedError: This method is no longer supported.
        """
        raise NotImplementedError(
            "CLIParser.parse() is deprecated. "
            "Use ShowrunnerInterface.interpret_and_execute() instead."
        )

    def get_help(self) -> str:
        """Get help text."""
        return """
QuestFoundry CLI - Natural Language Interface

Primary workflow:
  qf ask "<your request in natural language>"

Examples:
  qf ask "Can you create a mystery story about a space station?"
  qf ask "Review the story and identify interesting hooks"
  qf ask "Add backstory about the Foreman character"
  qf ask "Export the manuscript"

Debug workflow (for testing/auditing):
  qf loop <loop_id> --context "key=value"

For more information:
  qf --help
  qf version
"""
