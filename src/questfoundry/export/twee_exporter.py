"""Twee/SugarCube export format.

Generates a Twee 3 file compatible with SugarCube 2. The output can be
imported into Twine or compiled directly with Tweego.

Format reference: https://twinery.org/cookbook/terms/terms_twee.html
SugarCube: https://www.motoslave.net/sugarcube/2/docs/
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import (
        ExportChoice,
        ExportContext,
        ExportIllustration,
        ExportPassage,
    )

log = get_logger(__name__)


class TweeExporter:
    """Export story as Twee 3 / SugarCube 2 format."""

    format_name = "twee"

    def export(self, context: ExportContext, output_dir: Path) -> Path:
        """Write story as a .twee file.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.

        Returns:
            Path to the generated .twee file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "story.twee"

        lines: list[str] = []
        lines.extend(_story_header(context.title))
        lines.append("")

        # Build lookup structures
        choices_by_passage = _group_choices_by_passage(context.choices)
        illustrations_by_passage: dict[str, ExportIllustration] = {
            ill.passage_id: ill for ill in context.illustrations
        }

        # Find start passage
        start_passage = next((p for p in context.passages if p.is_start), context.passages[0])

        # Emit start passage first
        lines.extend(
            _render_passage(
                start_passage,
                choices_by_passage.get(start_passage.id, []),
                illustrations_by_passage.get(start_passage.id),
                is_start=True,
            )
        )
        lines.append("")

        # Emit remaining passages
        for passage in context.passages:
            if passage.id == start_passage.id:
                continue
            lines.extend(
                _render_passage(
                    passage,
                    choices_by_passage.get(passage.id, []),
                    illustrations_by_passage.get(passage.id),
                )
            )
            lines.append("")

        content = "\n".join(lines)
        output_file.write_text(content, encoding="utf-8")

        log.info(
            "twee_export_complete",
            passages=len(context.passages),
            choices=len(context.choices),
            output=str(output_file),
        )

        return output_file


def _story_header(title: str) -> list[str]:
    """Generate Twee 3 story header passages."""
    ifid = str(uuid.uuid4()).upper()
    return [
        f":: StoryTitle\n{title}",
        "",
        f':: StoryData\n{{"ifid": "{ifid}", "format": "SugarCube", "format-version": "2.37.3"}}',
    ]


def _passage_name(passage_id: str) -> str:
    """Convert a passage ID to a Twee passage name.

    Strips the ``passage::`` prefix for cleaner output.
    """
    if passage_id.startswith("passage::"):
        return passage_id[len("passage::") :]
    return passage_id


def _codeword_var(codeword_id: str) -> str:
    """Convert a codeword ID to a SugarCube variable name.

    Strips the ``codeword::`` prefix and uses ``$`` notation.
    """
    name = codeword_id
    if name.startswith("codeword::"):
        name = name[len("codeword::") :]
    return f"${name}"


def _group_choices_by_passage(
    choices: list[ExportChoice],
) -> dict[str, list[ExportChoice]]:
    """Group choices by their source passage."""
    result: dict[str, list[ExportChoice]] = {}
    for choice in choices:
        result.setdefault(choice.from_passage, []).append(choice)
    return result


def _render_passage(
    passage: ExportPassage,
    choices: list[ExportChoice],
    illustration: ExportIllustration | None = None,
    *,
    is_start: bool = False,
) -> list[str]:
    """Render a single passage as Twee markup."""
    name = "Start" if is_start else _passage_name(passage.id)
    tags = " [start]" if is_start else ""
    lines = [f":: {name}{tags}"]

    # Illustration
    if illustration is not None:
        lines.append(f"[img[{illustration.asset_path}]]")
        if illustration.caption:
            lines.append(f"//{illustration.caption}//")

    # Prose
    lines.append(passage.prose)

    # Choices
    for choice in choices:
        lines.append(_render_choice(choice))

    return lines


def _render_choice(choice: ExportChoice) -> str:
    """Render a single choice as SugarCube markup.

    If the choice has grants, uses ``<<link>>`` with ``<<set>>`` to
    assign codewords before navigating. If it also has requires, wraps
    the link in an ``<<if>>`` conditional.

    If no grants, uses simple ``[[label->target]]`` syntax.
    """
    target = _passage_name(choice.to_passage)

    if choice.grants:
        # Use <<link>> macro to set codewords before navigating
        sets = "".join(f"<<set {_codeword_var(cw)} to true>>" for cw in choice.grants)
        link = f'<<link "{choice.label}">>{sets}<<goto "{target}">><</link>>'
    else:
        link = f"[[{choice.label}->{target}]]"

    if choice.requires:
        conditions = " and ".join(_codeword_var(cw) for cw in choice.requires)
        return f"<<if {conditions}>>{link}<</if>>"
    return link
