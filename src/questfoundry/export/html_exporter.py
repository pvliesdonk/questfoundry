"""Standalone HTML export format.

Generates a single self-contained HTML file with embedded CSS and
JavaScript. The story is playable in any modern browser with no
external dependencies.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

from questfoundry.export.i18n import get_ui_strings
from questfoundry.export.metadata import ExportMetadata, build_export_metadata
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

# Backward-compatible additive changes only; bump on shape changes
# (new <head> meta block, breaking layout reshuffle, etc.).
HTML_FORMAT_VERSION = "1.0.0"

# R-3.3 voice-document-driven CSS. Each register/rhythm value maps to a
# body class plus a scoped rule block. Falls back to baseline styling
# (no class added) when FILL hasn't produced a voice document.
#
# The CSS dicts below are the SINGLE SOURCE OF TRUTH for the valid
# register/rhythm values: _voice_body_class and _voice_css_block both
# validate against dict membership so a class can never be emitted
# without a matching scoped rule (or vice versa).
_VOICE_REGISTER_CSS = {
    "formal": """body.register-formal .prose {
  font-family: 'EB Garamond', Garamond, Georgia, serif;
  text-align: justify;
}""",
    "conversational": """body.register-conversational .prose {
  font-family: 'Georgia', 'Times New Roman', serif;
  text-align: left;
}""",
    "literary": """body.register-literary .prose {
  font-family: 'Iowan Old Style', 'Palatino Linotype', Palatino, serif;
  text-align: justify;
  font-feature-settings: "liga", "dlig";
}""",
    "sparse": """body.register-sparse .prose {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  text-align: left;
  letter-spacing: 0.01em;
}""",
}

_VOICE_RHYTHM_CSS = {
    "varied": """body.rhythm-varied .prose {
  line-height: 1.7;
  margin-bottom: 1.5em;
}""",
    "punchy": """body.rhythm-punchy .prose {
  line-height: 1.45;
  margin-bottom: 1em;
}""",
    "flowing": """body.rhythm-flowing .prose {
  line-height: 1.85;
  margin-bottom: 1.8em;
}""",
}


class HtmlExporter:
    """Export story as a standalone HTML file."""

    format_name = "html"
    format_version = HTML_FORMAT_VERSION

    def export(
        self,
        context: ExportContext,
        output_dir: Path,
        *,
        timestamp: str | None = None,
    ) -> Path:
        """Write story as a single playable HTML file.

        Args:
            context: Extracted story data.
            output_dir: Directory to write output files.
            timestamp: Optional override for the metadata generation
                timestamp (test seam for deterministic assertions).

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

        ui = get_ui_strings(context.language)

        passages_html = []
        for passage in context.passages:
            choices = choices_by_passage.get(passage.id, [])
            illustration = illustrations_by_passage.get(passage.id)
            passages_html.append(_render_passage_div(passage, choices, illustration, ui=ui))

        # Build codex HTML
        codex_html = ""
        if context.codex_entries:
            codex_html = _render_codex_panel(context.codex_entries, ui=ui)

        # Build art direction meta tag.
        # json.dumps() serializes to JSON, then html.escape() escapes any
        # HTML-special chars (<, >, &, ", ') for safe embedding in an attribute.
        art_direction_meta = ""
        if context.art_direction:
            art_direction_meta = f'<meta name="art-direction" content="{html.escape(json.dumps(context.art_direction))}">'

        # Build cover HTML
        cover_html = ""
        if context.cover and context.cover.asset_path:
            cap = html.escape(context.cover.caption) if context.cover.caption else ""
            caption_tag = f"\n  <figcaption>{cap}</figcaption>" if cap else ""
            # When figcaption is present, use empty alt to avoid screen reader redundancy
            alt_attr = "" if cap else html.escape(ui["cover_alt"])
            cover_html = f'<figure class="cover">\n  <img src="{html.escape(context.cover.asset_path)}" alt="{alt_attr}">{caption_tag}\n</figure>'

        # R-3.6 metadata block — emitted as <meta name="qf-..."> tags
        # in <head> for in-band provenance without affecting the page body.
        metadata = build_export_metadata(context, HTML_FORMAT_VERSION, timestamp=timestamp)

        # Build the complete HTML document
        content = _build_html_document(
            title=context.title,
            passages_html="\n".join(passages_html),
            start_id=_safe_id(start_id),
            codex_html=codex_html,
            art_direction_meta=art_direction_meta,
            metadata=metadata,
            voice=context.voice,
            cover_html=cover_html,
            language=context.language,
            ui=ui,
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


def _render_metadata_meta_tags(metadata: ExportMetadata) -> str:
    """Render the R-3.6 metadata block as HTML <meta> tags.

    One tag per field, prefixed with ``qf-`` so they don't collide
    with any spec-defined name attributes.
    """
    return "\n".join(
        f'<meta name="qf-{key.replace("_", "-")}" content="{html.escape(value)}">'
        for key, value in sorted(metadata.to_dict().items())
    )


def _voice_body_class(voice: dict[str, Any] | None) -> str:
    """Return the ``class="..."`` attribute fragment for ``<body>`` (R-3.3).

    Empty string when no voice document is present, so the page falls
    back to baseline styling. Otherwise returns one or both of
    ``register-<name>`` and ``rhythm-<name>``, validated against the
    VoiceDocument literal sets so unknown values silently degrade
    rather than emit unscoped classes.
    """
    if not voice:
        return ""
    classes: list[str] = []
    register = voice.get("voice_register")
    if register in _VOICE_REGISTER_CSS:
        classes.append(f"register-{register}")
    rhythm = voice.get("sentence_rhythm")
    if rhythm in _VOICE_RHYTHM_CSS:
        classes.append(f"rhythm-{rhythm}")
    if not classes:
        return ""
    return f' class="{" ".join(classes)}"'


def _voice_css_block(voice: dict[str, Any] | None) -> str:
    """Return the voice-scoped CSS rule block to append to ``<style>``.

    The rules are body-scoped (``body.register-formal .prose {...}``),
    so they only activate when the matching class is present on
    ``<body>`` — cheap to ship, zero impact when voice is absent or
    its values fall outside the known set.
    """
    if not voice:
        return ""
    blocks: list[str] = []
    register = voice.get("voice_register")
    if register in _VOICE_REGISTER_CSS:
        blocks.append(_VOICE_REGISTER_CSS[register])
    rhythm = voice.get("sentence_rhythm")
    if rhythm in _VOICE_RHYTHM_CSS:
        blocks.append(_VOICE_RHYTHM_CSS[rhythm])
    return "\n".join(blocks)


def _render_passage_div(
    passage: ExportPassage,
    choices: list[ExportChoice],
    illustration: ExportIllustration | None = None,
    *,
    ui: dict[str, str] | None = None,
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
        the_end = (ui or {}).get("the_end", "The End")
        parts.append(f'  <div class="ending">{html.escape(the_end)}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def _render_codex_panel(
    codex_entries: list[ExportCodexEntry],
    *,
    ui: dict[str, str] | None = None,
) -> str:
    """Render a toggleable codex panel with conditionally visible entries."""
    codex_label = html.escape((ui or {}).get("codex", "Codex"))
    sorted_entries = sorted(codex_entries, key=lambda e: e.rank)
    parts = ['<div id="codex" class="codex-panel">']
    parts.append(f"  <h2>{codex_label}</h2>")
    for entry in sorted_entries:
        # json.dumps() handles quoting/escaping of codeword values,
        # html.escape() makes the JSON safe inside an HTML attribute.
        visible_attr = ""
        if entry.visible_when:
            visible_attr = f' data-visible-when="{html.escape(json.dumps(entry.visible_when))}"'
        parts.append(f'  <div class="codex-entry"{visible_attr}>')
        parts.append(f"    <h3>{html.escape(entry.title)}</h3>")
        parts.append(f"    <p>{html.escape(entry.content)}</p>")
        parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def _build_html_document(
    title: str,
    passages_html: str,
    start_id: str,
    metadata: ExportMetadata,
    codex_html: str = "",
    art_direction_meta: str = "",
    voice: dict[str, Any] | None = None,
    cover_html: str = "",
    language: str = "en",
    ui: dict[str, str] | None = None,
) -> str:
    """Build the complete HTML document.

    ``metadata`` is required: R-3.6 mandates a metadata header on every
    export, so an HTML build without it would silently violate the spec.
    Make the parameter mandatory rather than defaulting to ``None`` so a
    forgotten argument fails at call time, not at audit time.

    ``voice`` is optional (R-3.3): when present, voice-driven CSS
    classes are added to ``<body>`` and matching scoped rule blocks
    are appended to the embedded stylesheet. When absent, the page
    falls back to baseline styling — no class added, no extra rules.
    """
    extra_meta_parts = [_render_metadata_meta_tags(metadata)]
    if art_direction_meta:
        extra_meta_parts.append(art_direction_meta)
    extra_meta = "\n" + "\n".join(extra_meta_parts)
    body_class = _voice_body_class(voice)
    voice_css = _voice_css_block(voice)
    codex_label = html.escape((ui or {}).get("codex", "Codex"))
    codex_button = (
        f'<button class="codex-toggle" id="codex-toggle">{codex_label}</button>'
        if codex_html
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="{html.escape(language)}">
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
{voice_css}
</style>
</head>
<body{body_class}>
<h1>{html.escape(title)}</h1>
{cover_html}
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
