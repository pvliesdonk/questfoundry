"""Standalone HTML export format.

Generates a single self-contained HTML file with embedded CSS and
JavaScript. The story is playable in any modern browser with no
external dependencies.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import (
        ExportChoice,
        ExportCodexEntry,
        ExportContext,
        ExportIllustration,
        ExportPassage,
    )

log = get_logger(__name__)


class HtmlExporter:
    """Export story as a standalone HTML file."""

    format_name = "html"

    def export(self, context: ExportContext, output_dir: Path) -> Path:
        """Write story as a single playable HTML file.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.

        Returns:
            Path to the generated HTML file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "story.html"

        # Find start passage
        start_id = next(
            (p.id for p in context.passages if p.is_start),
            context.passages[0].id if context.passages else "",
        )

        # Build data structures for JavaScript
        choices_by_passage: dict[str, list[ExportChoice]] = {}
        for choice in context.choices:
            choices_by_passage.setdefault(choice.from_passage, []).append(choice)

        illustrations_by_passage = {ill.passage_id: ill for ill in context.illustrations}

        passages_html = []
        for passage in context.passages:
            choices = choices_by_passage.get(passage.id, [])
            illustration = illustrations_by_passage.get(passage.id)
            passages_html.append(_render_passage_div(passage, choices, illustration))

        # Build codex HTML
        codex_html = ""
        if context.codex_entries:
            codex_html = _render_codex_panel(context.codex_entries)

        # Build art direction meta tag.
        # json.dumps() serializes to JSON, then html.escape() escapes any
        # HTML-special chars (<, >, &, ", ') for safe embedding in an attribute.
        art_direction_meta = ""
        if context.art_direction:
            art_direction_meta = f'<meta name="art-direction" content="{html.escape(json.dumps(context.art_direction))}">'

        # Build the complete HTML document
        content = _build_html_document(
            title=context.title,
            passages_html="\n".join(passages_html),
            start_id=_safe_id(start_id),
            codex_html=codex_html,
            art_direction_meta=art_direction_meta,
        )

        output_file.write_text(content, encoding="utf-8")

        log.info(
            "html_export_complete",
            passages=len(context.passages),
            choices=len(context.choices),
            codex_entries=len(context.codex_entries),
            has_art_direction=context.art_direction is not None,
            output=str(output_file),
        )

        return output_file


def _safe_id(passage_id: str) -> str:
    """Convert a passage ID to a safe HTML id attribute."""
    return passage_id.replace("::", "--").replace(" ", "_")


def _render_passage_div(
    passage: ExportPassage,
    choices: list[ExportChoice],
    illustration: ExportIllustration | None = None,
) -> str:
    """Render a passage as an HTML div element."""
    pid = _safe_id(passage.id)
    prose_escaped = html.escape(passage.prose).replace("\n", "<br>\n")

    parts = [f'<div class="passage" id="{pid}">']

    # Illustration
    if illustration is not None:
        parts.append("  <figure>")
        parts.append(
            f'    <img src="{html.escape(illustration.asset_path)}" alt="{html.escape(illustration.caption)}" />'
        )
        parts.append(f"    <figcaption>{html.escape(illustration.caption)}</figcaption>")
        parts.append("  </figure>")

    # Prose
    parts.append(f'  <div class="prose">{prose_escaped}</div>')

    # Choices
    if choices:
        parts.append('  <div class="choices">')
        for choice in choices:
            target = _safe_id(choice.to_passage)
            label = html.escape(choice.label)
            requires_attr = ""
            grants_attr = ""
            if choice.requires:
                requires_attr = f' data-requires="{html.escape(json.dumps(choice.requires))}"'
            if choice.grants:
                grants_attr = f' data-grants="{html.escape(json.dumps(choice.grants))}"'
            parts.append(
                f'    <a class="choice" href="#" data-target="{target}"{requires_attr}{grants_attr}>{label}</a>'
            )
        parts.append("  </div>")

    # Ending marker
    if passage.is_ending:
        parts.append('  <div class="ending">The End</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _render_codex_panel(codex_entries: list[ExportCodexEntry]) -> str:
    """Render a toggleable codex panel with conditionally visible entries."""
    sorted_entries = sorted(codex_entries, key=lambda e: e.rank)
    parts = ['<div id="codex" class="codex-panel">']
    parts.append("  <h2>Codex</h2>")
    for entry in sorted_entries:
        # json.dumps() handles quoting/escaping of codeword values,
        # html.escape() makes the JSON safe inside an HTML attribute.
        visible_attr = ""
        if entry.visible_when:
            visible_attr = f' data-visible-when="{html.escape(json.dumps(entry.visible_when))}"'
        parts.append(f'  <div class="codex-entry"{visible_attr}>')
        parts.append(f"    <h3>{html.escape(entry.entity_id)}</h3>")
        parts.append(f"    <p>{html.escape(entry.content)}</p>")
        parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def _build_html_document(
    title: str,
    passages_html: str,
    start_id: str,
    codex_html: str = "",
    art_direction_meta: str = "",
) -> str:
    """Build the complete HTML document."""
    extra_meta = f"\n{art_direction_meta}" if art_direction_meta else ""
    codex_button = (
        '<button class="codex-toggle" id="codex-toggle">Codex</button>' if codex_html else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">{extra_meta}
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: Georgia, 'Times New Roman', serif;
  max-width: 700px;
  margin: 2em auto;
  padding: 0 1em;
  background: #1a1a2e;
  color: #e0e0e0;
  line-height: 1.7;
}}
.passage {{
  display: none;
}}
.passage.active {{
  display: block;
  animation: fadeIn 0.3s ease-in;
}}
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to {{ opacity: 1; }}
}}
.prose {{
  margin-bottom: 1.5em;
}}
.choices {{
  display: flex;
  flex-direction: column;
  gap: 0.5em;
}}
.choice {{
  display: block;
  padding: 0.7em 1em;
  background: #16213e;
  color: #a8d8ea;
  text-decoration: none;
  border-radius: 4px;
  border: 1px solid #0f3460;
  cursor: pointer;
  transition: background 0.2s;
}}
.choice:hover {{
  background: #0f3460;
}}
.choice.hidden {{
  display: none;
}}
.ending {{
  text-align: center;
  font-style: italic;
  margin-top: 2em;
  color: #888;
}}
figure {{
  margin: 1em 0;
  text-align: center;
}}
figure img {{
  max-width: 100%;
  border-radius: 4px;
}}
figcaption {{
  font-style: italic;
  color: #888;
  margin-top: 0.5em;
}}
h1 {{
  text-align: center;
  color: #a8d8ea;
}}
.codex-toggle {{
  position: fixed;
  top: 1em;
  right: 1em;
  padding: 0.5em 1em;
  background: #16213e;
  color: #a8d8ea;
  border: 1px solid #0f3460;
  border-radius: 4px;
  cursor: pointer;
  z-index: 10;
}}
.codex-toggle:hover {{
  background: #0f3460;
}}
.codex-panel {{
  display: none;
  position: fixed;
  top: 4em;
  right: 1em;
  width: 300px;
  max-height: 80vh;
  overflow-y: auto;
  background: #16213e;
  border: 1px solid #0f3460;
  border-radius: 4px;
  padding: 1em;
  z-index: 10;
}}
.codex-panel.visible {{
  display: block;
}}
.codex-panel h2 {{
  margin-top: 0;
  color: #a8d8ea;
}}
.codex-entry {{
  margin-bottom: 1em;
}}
.codex-entry.hidden {{
  display: none;
}}
.codex-entry h3 {{
  color: #a8d8ea;
  margin-bottom: 0.3em;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{codex_button}
{passages_html}

{codex_html}

<script>
(function() {{
  const codewords = new Set();
  const startId = "{start_id}";

  function showPassage(id) {{
    document.querySelectorAll('.passage').forEach(p => p.classList.remove('active'));
    const el = document.getElementById(id);
    if (el) {{
      el.classList.add('active');
      updateChoiceVisibility(el);
      window.scrollTo(0, 0);
    }}
  }}

  function updateChoiceVisibility(passageEl) {{
    passageEl.querySelectorAll('.choice').forEach(link => {{
      const requires = link.dataset.requires;
      if (requires) {{
        const needed = JSON.parse(requires);
        const visible = needed.every(cw => codewords.has(cw));
        link.classList.toggle('hidden', !visible);
      }}
    }});
  }}

  document.addEventListener('click', function(e) {{
    const link = e.target.closest('.choice');
    if (!link) return;
    e.preventDefault();
    const grants = link.dataset.grants;
    if (grants) {{
      JSON.parse(grants).forEach(cw => codewords.add(cw));
    }}
    updateCodexVisibility();
    const target = link.dataset.target;
    if (target) showPassage(target);
  }});

  function updateCodexVisibility() {{
    document.querySelectorAll('.codex-entry').forEach(entry => {{
      const visibleWhen = entry.dataset.visibleWhen;
      if (visibleWhen) {{
        const needed = JSON.parse(visibleWhen);
        const visible = needed.every(cw => codewords.has(cw));
        entry.classList.toggle('hidden', !visible);
      }}
    }});
  }}

  const codexToggle = document.getElementById('codex-toggle');
  if (codexToggle) {{
    codexToggle.addEventListener('click', function() {{
      const panel = document.getElementById('codex');
      if (panel) {{
        panel.classList.toggle('visible');
        updateCodexVisibility();
      }}
    }});
  }}

  showPassage(startId);
}})();
</script>
</body>
</html>"""
