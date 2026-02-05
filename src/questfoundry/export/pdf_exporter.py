"""Gamebook-style PDF export format.

Generates a classic gamebook (Choose Your Own Adventure / Fighting Fantasy style)
PDF with numbered passages, "go to section N" choice references, codeword
checklist, and codex appendix.

Requires the optional `pdf` dependency: `uv pip install questfoundry[pdf]`
"""

from __future__ import annotations

import hashlib
import html
import random
from typing import TYPE_CHECKING

from questfoundry.export.i18n import get_ui_strings
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import (
        ExportChoice,
        ExportCodeword,
        ExportCodexEntry,
        ExportContext,
        ExportIllustration,
        ExportPassage,
    )

log = get_logger(__name__)

# CSS for A5 gamebook layout
_CSS_STYLES = """
@page {
    size: A5;
    margin: 15mm 12mm;
}

body {
    font-family: "Palatino Linotype", "Palatino", "Georgia", serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
}

h1 {
    font-size: 24pt;
    font-weight: normal;
    text-align: center;
    margin-bottom: 2em;
}

/* Title page */
.title-page {
    page-break-after: always;
    text-align: center;
    padding-top: 30mm;
}

.title-page h1 {
    font-size: 28pt;
    margin-bottom: 1em;
}

.title-page .subtitle {
    font-style: italic;
    margin-bottom: 3em;
}

.cover-image {
    max-width: 80%;
    max-height: 100mm;
    margin: 2em auto;
    display: block;
}

.cover-caption {
    font-size: 9pt;
    color: #666;
    text-align: center;
    margin-top: 0.5em;
}

/* Instructions page */
.instructions {
    page-break-after: always;
}

.instructions h2 {
    text-align: center;
    font-size: 14pt;
    margin-bottom: 1em;
}

.instructions p {
    margin-bottom: 1em;
}

/* Passages */
.passage {
    page-break-inside: avoid;
    margin-bottom: 2.5em;
}

.passage-number {
    text-align: center;
    font-size: 14pt;
    font-weight: normal;
    margin-bottom: 0.5em;
    letter-spacing: 0.1em;
}

.passage-illustration {
    max-width: 100%;
    max-height: 60mm;
    display: block;
    margin: 1em auto;
}

.passage-illustration-caption {
    font-size: 9pt;
    color: #666;
    text-align: center;
    margin-top: 0.3em;
    margin-bottom: 1em;
}

.prose {
    text-align: justify;
    text-indent: 1.5em;
}

.prose p:first-child {
    text-indent: 0;
}

.choices {
    margin-top: 1.5em;
    font-style: italic;
}

.choice {
    margin-bottom: 0.5em;
}

.choice-requires {
    font-size: 10pt;
    color: #555;
}

.ending {
    text-align: center;
    font-weight: bold;
    margin-top: 2em;
    font-size: 12pt;
}

/* Appendix sections */
.appendix {
    page-break-before: always;
}

.appendix h2 {
    text-align: center;
    font-size: 16pt;
    margin-bottom: 1.5em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.5em;
}

/* Codeword checklist */
.codeword-checklist ul {
    list-style: none;
    padding: 0;
    columns: 2;
    column-gap: 2em;
}

.codeword-checklist li {
    margin-bottom: 0.8em;
    break-inside: avoid;
}

.codeword-checkbox {
    display: inline-block;
    width: 12pt;
    height: 12pt;
    border: 1px solid #333;
    margin-right: 0.5em;
    vertical-align: middle;
}

.codeword-name {
    font-variant: small-caps;
    letter-spacing: 0.05em;
}

/* Codex - two-column, compact layout */
.codex {
    columns: 2;
    column-gap: 1.5em;
}

.codex-entry {
    margin-bottom: 1em;
    break-inside: avoid;
}

.codex-entry h3 {
    font-size: 10pt;
    font-weight: bold;
    margin-bottom: 0.3em;
}

.codex-entry p {
    font-size: 9pt;
    text-align: justify;
    line-height: 1.4;
}
"""


class PdfExporter:
    """Export story as a gamebook-style PDF."""

    format_name = "pdf"

    def export(self, context: ExportContext, output_dir: Path) -> Path:
        """Write story as a gamebook PDF with numbered passages.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.

        Returns:
            Path to the generated PDF file.
        """
        try:
            from weasyprint import HTML
        except ImportError as e:
            msg = (
                "WeasyPrint is required for PDF export. "
                "Install with: uv pip install questfoundry[pdf]"
            )
            raise ImportError(msg) from e

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "story.pdf"

        # Build passage number mapping (randomized for spoiler prevention)
        numbering = _build_passage_numbering(context.passages)

        # Build data structures
        choices_by_passage: dict[str, list[ExportChoice]] = {}
        for choice in context.choices:
            choices_by_passage.setdefault(choice.from_passage, []).append(choice)

        illustrations_by_passage = {ill.passage_id: ill for ill in context.illustrations}

        # Generate HTML content
        html_content = _render_html(
            context=context,
            numbering=numbering,
            choices_by_passage=choices_by_passage,
            illustrations_by_passage=illustrations_by_passage,
        )

        # Convert to PDF
        # Set base_url to project root so relative asset paths resolve correctly
        # output_dir is typically PROJECT/exports/pdf/, so parent.parent = PROJECT/
        project_root = output_dir.parent.parent
        html_doc = HTML(string=html_content, base_url=str(project_root))
        html_doc.write_pdf(output_file)

        log.info(
            "pdf_export_complete",
            passages=len(context.passages),
            codewords=len(context.codewords),
            output=str(output_file),
        )

        return output_file


def _build_passage_numbering(passages: list[ExportPassage]) -> dict[str, int]:
    """Map passage IDs to randomized section numbers.

    Uses sorted IDs as seed for reproducible shuffling.
    Same story = same numbers across exports.

    Start passage always gets number 1 for reader convenience.

    Args:
        passages: List of passages to number.

    Returns:
        Mapping from passage ID to section number.
    """
    if not passages:
        return {}

    # Find start passage
    start_id = next((p.id for p in passages if p.is_start), passages[0].id)

    # Get other passage IDs (excluding start)
    other_ids = sorted(p.id for p in passages if p.id != start_id)

    # Create reproducible random seed from passage IDs using hashlib for cross-session determinism
    # (Python's hash() is randomized per process via PYTHONHASHSEED)
    all_ids = sorted(p.id for p in passages)
    id_string = "|".join(all_ids)
    seed = int(hashlib.md5(id_string.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    # Assign numbers 2..N to other passages randomly
    numbers = list(range(2, len(passages) + 1))
    rng.shuffle(numbers)

    # Build mapping: start=1, others=randomized
    numbering: dict[str, int] = {start_id: 1}
    for pid, num in zip(other_ids, numbers, strict=True):
        numbering[pid] = num

    return numbering


def _render_html(
    context: ExportContext,
    numbering: dict[str, int],
    choices_by_passage: dict[str, list[ExportChoice]],
    illustrations_by_passage: dict[str, ExportIllustration],
) -> str:
    """Render the complete HTML document for PDF conversion.

    Args:
        context: Export context with all story data.
        numbering: Passage ID to section number mapping.
        choices_by_passage: Choices grouped by source passage.
        illustrations_by_passage: Illustrations indexed by passage ID.

    Returns:
        Complete HTML document string.
    """
    ui = get_ui_strings(context.language)

    parts: list[str] = []

    # HTML head
    parts.append(f"""<!DOCTYPE html>
<html lang="{html.escape(context.language)}">
<head>
<meta charset="UTF-8">
<title>{html.escape(context.title)}</title>
<style>
{_CSS_STYLES}
</style>
</head>
<body>
""")

    # Title page
    parts.append(_render_title_page(context, ui))

    # Instructions page
    parts.append(_render_instructions(ui))

    # Passages (sorted by section number)
    sorted_passages = sorted(context.passages, key=lambda p: numbering.get(p.id, 999))
    for passage in sorted_passages:
        choices = choices_by_passage.get(passage.id, [])
        illustration = illustrations_by_passage.get(passage.id)
        parts.append(_render_passage(passage, numbering, choices, illustration, ui))

    # Codeword checklist (appendix)
    if context.codewords:
        parts.append(_render_codeword_checklist(context.codewords, ui))

    # Codex (appendix)
    if context.codex_entries:
        parts.append(_render_codex(context.codex_entries, ui))

    parts.append("</body>\n</html>")

    return "\n".join(parts)


def _render_title_page(context: ExportContext, ui: dict[str, str]) -> str:
    """Render the title page with optional cover image."""
    parts: list[str] = ['<section class="title-page">']
    parts.append(f"<h1>{html.escape(context.title)}</h1>")

    # Subtitle with genre hint from art direction
    if context.art_direction:
        genre = context.art_direction.get("genre", "")
        if genre:
            parts.append(f'<p class="subtitle">{html.escape(genre)}</p>')

    # Cover image
    if context.cover and context.cover.asset_path:
        parts.append(
            f'<img class="cover-image" src="{html.escape(context.cover.asset_path)}" '
            f'alt="{html.escape(context.cover.caption or ui.get("cover_alt", "Cover"))}">'
        )
        if context.cover.caption:
            parts.append(f'<p class="cover-caption">{html.escape(context.cover.caption)}</p>')

    parts.append("</section>")
    return "\n".join(parts)


def _render_instructions(ui: dict[str, str]) -> str:
    """Render the how-to-play instructions page."""
    title = ui.get("instructions_title", "How to Play")
    # Default instructions text
    instructions = ui.get(
        "instructions_text",
        (
            "This is an interactive story. At the end of each section, you will be "
            "presented with choices. Each choice tells you which section to turn to "
            "next. Simply find that numbered section and continue reading.\n\n"
            "Some choices may only be available if you have collected certain "
            "codewords during your adventure. Keep track of your codewords using "
            "the checklist at the back of this book.\n\n"
            "Begin your adventure at section 1."
        ),
    )

    parts = ['<section class="instructions">']
    parts.append(f"<h2>{html.escape(title)}</h2>")
    for paragraph in instructions.split("\n\n"):
        if paragraph.strip():
            parts.append(f"<p>{html.escape(paragraph.strip())}</p>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_passage(
    passage: ExportPassage,
    numbering: dict[str, int],
    choices: list[ExportChoice],
    illustration: ExportIllustration | None,
    ui: dict[str, str],
) -> str:
    """Render a single passage section."""
    section_num = numbering.get(passage.id, 0)
    parts: list[str] = [f'<section class="passage" id="section-{section_num}">']

    # Section number header
    parts.append(f'<p class="passage-number">— {section_num} —</p>')

    # Illustration (if any)
    if illustration and illustration.asset_path:
        parts.append(
            f'<img class="passage-illustration" src="{html.escape(illustration.asset_path)}" '
            f'alt="{html.escape(illustration.caption or "")}">'
        )
        if illustration.caption:
            parts.append(
                f'<p class="passage-illustration-caption">{html.escape(illustration.caption)}</p>'
            )

    # Prose content
    prose_html = _format_prose(passage.prose)
    parts.append(f'<div class="prose">{prose_html}</div>')

    # Choices or ending
    if passage.is_ending:
        the_end = ui.get("the_end", "THE END")
        parts.append(f'<p class="ending">{html.escape(the_end)}</p>')
    elif choices:
        parts.append(_render_choices(choices, numbering, ui))

    parts.append("</section>")
    return "\n".join(parts)


def _format_prose(prose: str) -> str:
    """Format prose text as HTML paragraphs.

    Double newlines (\\n\\n) split paragraphs into separate <p> tags.
    Single newlines within a paragraph become <br> tags.
    Empty paragraphs (whitespace-only) are filtered out.

    Args:
        prose: Raw prose text with newlines.

    Returns:
        HTML string with <p> tags for each paragraph.
    """
    paragraphs = prose.strip().split("\n\n")
    formatted = []
    for p in paragraphs:
        # Replace single newlines with <br> within paragraphs
        p_html = html.escape(p.strip()).replace("\n", "<br>")
        if p_html:
            formatted.append(f"<p>{p_html}</p>")
    return "\n".join(formatted)


def _render_choices(
    choices: list[ExportChoice],
    numbering: dict[str, int],
    ui: dict[str, str],
) -> str:
    """Render passage choices as gamebook-style references."""
    parts: list[str] = ['<div class="choices">']

    go_to = ui.get("go_to", "go to")
    if_you_have = ui.get("if_you_have", "If you have")

    for choice in choices:
        target_num = numbering.get(choice.to_passage, 0)
        label = html.escape(choice.label)

        parts.append('<p class="choice">')

        # Handle conditional choices (requires codewords)
        if choice.requires:
            codeword_names = ", ".join(
                f'<span class="codeword-name">{html.escape(_format_codeword_name(cw))}</span>'
                for cw in choice.requires
            )
            parts.append(f'<span class="choice-requires">{if_you_have} {codeword_names}: </span>')

        # Choice text with target section
        parts.append(f"{label}, {go_to} <strong>{target_num}</strong>.")
        parts.append("</p>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_codeword_checklist(codewords: list[ExportCodeword], ui: dict[str, str]) -> str:
    """Render the codeword checklist appendix."""
    title = ui.get("codeword_checklist", "Codeword Checklist")

    parts: list[str] = ['<section class="appendix codeword-checklist">']
    parts.append(f"<h2>{html.escape(title)}</h2>")
    parts.append("<ul>")

    # Sort codewords alphabetically by display name
    sorted_codewords = sorted(codewords, key=lambda cw: _format_codeword_name(cw.id))

    for codeword in sorted_codewords:
        name = _format_codeword_name(codeword.id)
        parts.append(
            f'<li><span class="codeword-checkbox"></span>'
            f'<span class="codeword-name">{html.escape(name)}</span></li>'
        )

    parts.append("</ul>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_codex(codex_entries: list[ExportCodexEntry], ui: dict[str, str]) -> str:
    """Render the codex appendix with only spoiler-free entries.

    For print media, we only include entries that have no visibility restrictions
    (visible_when is empty). Spoiler entries that require specific codewords are
    omitted since the reader can't dynamically unlock them in a physical book.
    """
    # Filter to only spoiler-free entries (no visibility restrictions)
    spoiler_free = [e for e in codex_entries if not e.visible_when]

    if not spoiler_free:
        return ""

    title = ui.get("codex", "Codex")

    parts: list[str] = ['<section class="appendix codex">']
    parts.append(f"<h2>{html.escape(title)}</h2>")

    # Sort by rank, then title
    sorted_entries = sorted(spoiler_free, key=lambda e: (e.rank, e.title))

    for entry in sorted_entries:
        parts.append('<article class="codex-entry">')
        parts.append(f"<h3>{html.escape(entry.title)}</h3>")

        # Content
        content_html = _format_prose(entry.content)
        parts.append(content_html)

        parts.append("</article>")

    parts.append("</section>")
    return "\n".join(parts)


def _format_codeword_name(codeword_id: str) -> str:
    """Format a codeword ID for display.

    Strips the 'codeword::' prefix and converts to title case.

    Args:
        codeword_id: Full codeword ID (e.g., 'codeword::golden_key').

    Returns:
        Display name (e.g., 'Golden Key').
    """
    # Strip prefix and convert snake_case to Title Case
    name = codeword_id.rsplit("::", 1)[-1] if "::" in codeword_id else codeword_id
    return name.replace("_", " ").title()
