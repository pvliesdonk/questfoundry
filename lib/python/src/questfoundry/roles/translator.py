"""Translator role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class Translator(Role):
    """
    Translator: Create player-safe translations preserving narrative intent.

    The Translator produces language packs with glossaries, style transfer notes,
    and localized surfaces while preserving PN boundaries, style intent, and
    navigation structure.

    Key responsibilities:
    - Create/refresh glossaries with register decisions
    - Translate player surfaces (no internal labels, no spoilers)
    - Preserve PN diegesis and hyperlink structure
    - Solve idioms with functionally equivalent phrases
    - Flag untranslatables and cultural risks
    - Validate cross-references and link resolution
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/translator.md"""
        return "translator"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Translator"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a translator task.

        Supported tasks:
        - 'create_glossary': Build terminology dictionary
        - 'translate_content': Localize player surfaces
        - 'solve_idiom': Find functional equivalents for idioms
        - 'validate_links': Check cross-reference resolution
        - 'assess_coverage': Calculate translation completeness
        - 'extract_strings': Extract strings for translation (alias for create_glossary)
        - 'translate': Translate content (alias for translate_content)
        - 'validate_translation': Validate translation (alias for validate_links)
        - 'check_style': Check style in translation (alias for assess_coverage)

        Args:
            context: Execution context

        Returns:
            Result with translation artifacts and metadata
        """
        task = context.task.lower()

        if task == "create_glossary":
            return self._create_glossary(context)
        elif task == "translate_content":
            return self._translate_content(context)
        elif task == "solve_idiom":
            return self._solve_idiom(context)
        elif task == "validate_links":
            return self._validate_links(context)
        elif task == "assess_coverage":
            return self._assess_coverage(context)
        # New tasks for loops
        elif task == "extract_strings":
            return self._create_glossary(context)
        elif task == "translate":
            return self._translate_content(context)
        elif task == "validate_translation":
            return self._validate_links(context)
        elif task == "check_style":
            return self._assess_coverage(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _create_glossary(self, context: RoleContext) -> RoleResult:
        """Build terminology dictionary."""
        system_prompt = self.build_system_prompt(context)

        source_content = context.additional_context.get("source_content", [])
        target_language = context.additional_context.get("target_language", "")

        user_prompt = f"""# Task: Create Glossary

{self.format_artifacts(context.artifacts)}

## Source Content
{self._format_list(source_content[:10])}  # Sample

## Target Language
{target_language}

Create glossary with:

1. **Terms**: Key terms needing consistent translation
2. **Register Decisions**:
   - T/V distinction (formal/informal "you")
   - Dialect choices
   - Tone equivalents
3. **Do-Not-Translate**: Terms to preserve (proper names, technical terms)
4. **Usage Examples**: Context for each term

For each term:
- Source term
- Target translation
- Part of speech
- Register notes
- Usage context
- Alternatives (if any)

Respond in JSON format:
```json
{{
  "glossary": [
    {{
      "source_term": "term",
      "target_translation": "translation",
      "part_of_speech": "noun|verb|adj|etc",
      "register": "formal|informal|neutral",
      "usage_notes": "When/how to use",
      "examples": ["example 1", "example 2"],
      "do_not_translate": false
    }}
  ],
  "register_map": {{
    "pronoun_choice": "T or V form",
    "formality": "formal|informal|mixed",
    "tone_notes": "Overall register guidance"
  }}
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "glossary",
                    "glossary": data.get("glossary", []),
                    "term_count": len(data.get("glossary", [])),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating glossary: {e}",
            )

    def _translate_content(self, context: RoleContext) -> RoleResult:
        """Localize player surfaces."""
        system_prompt = self.build_system_prompt(context)

        source_text = context.additional_context.get("source_text", "")
        glossary = context.additional_context.get("glossary", [])
        target_language = context.additional_context.get("target_language", "")

        user_prompt = f"""# Task: Translate Content

{self.format_artifacts(context.artifacts)}

## Source Text
{source_text}

## Glossary
{self._format_glossary(glossary)}

## Target Language
{target_language}

Translate while:
- Using glossary terms consistently
- Preserving PN diegesis (no internal labels)
- Keeping hyperlinks and anchors intact
- Maintaining choice label distinctiveness
- Staying spoiler-safe
- Matching register from glossary

Ensure:
- Natural phrasing in target language
- Motifs resonate appropriately
- No literalism that breaks tone
- Navigation structure preserved

Respond in JSON format:
```json
{{
  "translated_text": "Full translation",
  "glossary_terms_used": ["term1", "term2"],
  "anchors_preserved": ["anchor1", "anchor2"],
  "notes": "Any translation decisions or challenges"
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=data.get("translated_text", response),
                metadata={
                    "content_type": "translation",
                    "translated_text": data.get("translated_text", ""),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error translating content: {e}",
            )

    def _solve_idiom(self, context: RoleContext) -> RoleResult:
        """Find functional equivalents for idioms."""
        system_prompt = self.build_system_prompt(context)

        idiom = context.additional_context.get("idiom", "")
        idiom_context = context.additional_context.get("context", "")
        target_language = context.additional_context.get("target_language", "")

        user_prompt = f"""# Task: Solve Idiom

{self.format_artifacts(context.artifacts)}

## Idiom
"{idiom}"

## Context
{idiom_context}

## Target Language
{target_language}

Provide solutions:
1. **Literal Translation**: Direct translation (if sensible)
2. **Functional Equivalent**: Target language idiom with similar meaning
3. **Rewrite**: Neutral phrasing that conveys intent
4. **Recommendation**: Which approach to use and why

Consider:
- Does literal translation work?
- Is there a natural equivalent idiom?
- Would neutral phrasing be better?
- Cultural appropriateness

Respond in JSON format:
```json
{{
  "solutions": {{
    "literal": "literal translation",
    "functional_equivalent": "equivalent idiom if exists",
    "rewrite": "neutral phrasing alternative"
  }},
  "recommendation": "which solution to use",
  "rationale": "why this solution works best",
  "untranslatable": false
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "idiom_solution",
                    "solution": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error solving idiom: {e}",
            )

    def _validate_links(self, context: RoleContext) -> RoleResult:
        """Check cross-reference resolution."""
        system_prompt = self.build_system_prompt(context)

        translated_content = context.additional_context.get("translated_content", {})
        link_map = context.additional_context.get("link_map", {})

        user_prompt = f"""# Task: Validate Links

{self.format_artifacts(context.artifacts)}

## Translated Content
{self._format_dict(translated_content)}

## Expected Links
{self._format_dict(link_map)}

Validate:
1. **Anchors Preserved**: All hyperlinks/anchors intact?
2. **Target Resolution**: Links point to correct targets?
3. **Choice Labels**: Distinct and clear?
4. **Cross-References**: Codex entries resolve?

For each issue:
- Link/anchor ID
- Problem description
- Severity (broken|missing|ambiguous)
- Suggested fix

Respond in JSON format:
```json
{{
  "validation": {{
    "all_links_valid": true|false,
    "issues": [
      {{
        "link_id": "id",
        "problem": "description",
        "severity": "broken|missing|ambiguous",
        "suggestion": "how to fix"
      }}
    ]
  }},
  "links_checked": 0,
  "links_valid": 0
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)
            validation = data.get("validation", {})
            all_valid = validation.get("all_links_valid", False)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "link_validation",
                    "all_valid": all_valid,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating links: {e}",
            )

    def _assess_coverage(self, context: RoleContext) -> RoleResult:
        """Calculate translation completeness."""
        system_prompt = self.build_system_prompt(context)

        source_sections = context.additional_context.get("source_sections", [])
        translated_sections = context.additional_context.get("translated_sections", [])

        user_prompt = f"""# Task: Assess Translation Coverage

{self.format_artifacts(context.artifacts)}

## Source Sections
Total: {len(source_sections)}

## Translated Sections
Total: {len(translated_sections)}

Calculate coverage:
1. **Section Coverage**: % of sections translated
2. **Codex Coverage**: % of codex entries translated
3. **Choice Coverage**: % of choice labels translated
4. **Status by Section**: complete|partial|missing

Respond in JSON format:
```json
{{
  "coverage": {{
    "section_percentage": 0.0,
    "codex_percentage": 0.0,
    "choice_percentage": 0.0,
    "overall_status": "complete|incomplete"
  }},
  "by_section": [
    {{
      "section_id": "id",
      "status": "complete|partial|missing",
      "completion_percentage": 0.0
    }}
  ],
  "open_issues": ["issue 1", "issue 2"]
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "coverage_assessment",
                    "coverage": data.get("coverage", {}),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error assessing coverage: {e}",
            )

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dictionary as bullet list."""
        if not d:
            return "(empty)"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())

    def _format_list(self, items: list[str]) -> str:
        """Format list as bullet list."""
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)

    def _format_glossary(self, glossary: list[dict[str, Any]]) -> str:
        """Format glossary for prompt."""
        if not glossary:
            return "(no glossary provided)"

        formatted = []
        for entry in glossary:
            formatted.append(
                f"- {entry.get('source_term')}: {entry.get('target_translation')} "
                f"({entry.get('part_of_speech')})"
            )
        return "\n".join(formatted)
