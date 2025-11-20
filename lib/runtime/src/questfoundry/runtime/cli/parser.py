"""
CLI Parser - parse natural language commands into studio intents.

Based on spec: components/cli.md
FLEXIBLE component - creative interpretation permitted for UX.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Represents a parsed CLI command."""
    action: str  # write, review, add, tune, export, narrate, etc.
    loop_id: str  # Target loop ID
    context: Dict[str, Any]  # Command parameters
    raw: str  # Original command text


class CLIParser:
    """Parse natural language commands into studio intents."""

    # Command mappings: (keywords) → (loop_id, action)
    COMMAND_MAP = {
        # Story creation
        ("write", "scene"): ("story_spark", "write_scene"),
        ("add", "scene"): ("story_spark", "write_scene"),
        ("create", "scene"): ("story_spark", "write_scene"),
        ("draft", "section"): ("story_spark", "write_section"),

        # Hook harvesting / Story review
        ("review", "story"): ("hook_harvest", "review_story"),
        ("review", "hooks"): ("hook_harvest", "review_hooks"),
        ("harvest", "hooks"): ("hook_harvest", "harvest_hooks"),
        ("triage", "hooks"): ("hook_harvest", "triage_hooks"),

        # Lore expansion
        ("add", "lore"): ("lore_deepening", "add_lore"),
        ("expand", "lore"): ("lore_deepening", "expand_lore"),
        ("deepen", "lore"): ("lore_deepening", "deepen_lore"),
        ("canon", "update"): ("lore_deepening", "update_canon"),

        # Code/Codex expansion
        ("expand", "codex"): ("codex_expansion", "expand_codex"),
        ("add", "codex"): ("codex_expansion", "add_codex"),
        ("define", "term"): ("codex_expansion", "define_term"),

        # Audio production
        ("add", "audio"): ("audio_pass", "add_audio"),
        ("generate", "audio"): ("audio_pass", "generate_audio"),
        ("narrate", "audio"): ("audio_pass", "narrate_audio"),

        # Narration dry-run
        ("narrate", "story"): ("narration_dry_run", "narrate_story"),
        ("narrate", "scene"): ("narration_dry_run", "narrate_scene"),
        ("read", "aloud"): ("narration_dry_run", "read_aloud"),

        # Style tuning
        ("tune", "style"): ("style_tune_up", "tune_style"),
        ("style", "check"): ("style_tune_up", "style_check"),
        ("check", "tone"): ("style_tune_up", "check_tone"),

        # Book binding / Export
        ("export", "book"): ("binding_run", "export_book"),
        ("bind", "book"): ("binding_run", "bind_book"),
        ("export", "story"): ("binding_run", "export_story"),
    }

    def __init__(self):
        """Initialize CLI parser."""
        pass

    def parse(self, command_text: str) -> Optional[Command]:
        """
        Parse natural language command.

        Args:
            command_text: Raw command text from user

        Returns:
            Parsed Command object or None if unrecognized

        Example:
            parse("write a tense cargo bay scene") →
            Command(
                action="write_scene",
                loop_id="story_spark",
                context={"scene_text": "a tense cargo bay scene"},
                raw="write a tense cargo bay scene"
            )
        """
        # Normalize input
        text = command_text.strip().lower()

        if not text:
            return None

        # Extract keywords
        words = text.split()

        # Try to match against command map
        best_match = None
        best_score = 0

        for keywords, (loop_id, action) in self.COMMAND_MAP.items():
            # Check if all keywords appear in command
            if all(kw in text for kw in keywords):
                score = len(keywords)  # Prefer longer matches
                if score > best_score:
                    best_match = (loop_id, action)
                    best_score = score

        if not best_match:
            logger.warning(f"Unrecognized command: {command_text}")
            return None

        loop_id, action = best_match

        # Extract context (remaining text after keywords)
        # Simple heuristic: remove command keywords, use rest as context
        remaining = text
        for cmd_map in self.COMMAND_MAP:
            for keyword in cmd_map:
                if keyword in remaining:
                    remaining = remaining.replace(keyword, " ").strip()

        context = {
            "scene_text": remaining,
            "loop_id": loop_id,
            "action": action
        }

        # Clean up empty context
        if not remaining:
            context.pop("scene_text", None)

        return Command(
            action=action,
            loop_id=loop_id,
            context=context,
            raw=command_text
        )

    def get_help(self) -> str:
        """Get help text for CLI commands."""
        help_text = """
QuestFoundry CLI Commands
========================

Story Creation:
  qf write <scene description>     - Create a new scene with story_spark loop
  qf review story                  - Review and harvest hooks from existing story
  qf narrate <scene>               - Do a dry-run narration with player-narrator

Lore & Canon:
  qf add lore <topic>              - Expand lore/canon with lore_deepening loop
  qf expand codex <term>           - Add/expand codex entries

Style & Audio:
  qf tune style                    - Check and tune style consistency
  qf add audio <description>       - Generate audio for sections

Export:
  qf export <format>               - Export/bind manuscript to format
                                     (binding_run loop)

Examples:
  qf write "a tense cargo bay confrontation"
  qf review story
  qf add lore "The Foreman's background"
  qf export epub
"""
        return help_text
