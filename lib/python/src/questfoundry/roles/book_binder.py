"""Book Binder role implementation."""

import logging
from typing import Any

from .base import Role, RoleContext, RoleResult

logger = logging.getLogger(__name__)


class BookBinder(Role):
    """
    Book Binder: Create final exports from canonical content.

    The Book Binder assembles exports using configured formats and options,
    includes requested assets (art, audio, translations), and generates
    front matter with snapshot metadata and coverage information.

    Key responsibilities:
    - Create exports using Epic 10 export module
    - Assemble content with requested assets
    - Generate front matter with snapshot IDs and coverage
    - Include/exclude content based on export options
    - Validate export completeness and structure
    - Package language packs with coverage flags
    """

    @property
    def role_name(self) -> str:
        """Role identifier matching spec/01-roles/briefs/book_binder.md"""
        return "book_binder"

    @property
    def display_name(self) -> str:
        """Human-readable role name"""
        return "Book Binder"

    def execute_task(self, context: RoleContext) -> RoleResult:
        """
        Execute a book binder task.

        Supported tasks:
        - 'create_export': Generate export from snapshot
        - 'generate_front_matter': Create metadata and coverage info
        - 'validate_export': Check export completeness
        - 'package_assets': Bundle art/audio/translations
        - 'generate_view': Generate view export (alias for create_export)
        - 'export_files': Export and bundle files (alias for package_assets)

        Args:
            context: Execution context

        Returns:
            Result with export artifacts and metadata
        """
        task = context.task.lower()

        if task == "create_export":
            return self._create_export(context)
        elif task == "generate_front_matter":
            return self._generate_front_matter(context)
        elif task == "validate_export":
            return self._validate_export(context)
        elif task == "package_assets":
            return self._package_assets(context)
        # New tasks for loops
        elif task == "generate_view":
            return self._create_export(context)
        elif task == "export_files":
            return self._package_assets(context)
        else:
            return RoleResult(
                success=False,
                output="",
                error=f"Unknown task: {task}",
            )

    def _create_export(self, context: RoleContext) -> RoleResult:
        """Generate export from snapshot."""
        system_prompt = self.build_system_prompt(context)

        export_options = context.additional_context.get("export_options", {})
        snapshot_id = context.additional_context.get("snapshot_id", "")

        user_prompt = f"""# Task: Create Export

{self.format_artifacts(context.artifacts)}

## Export Options
{self._format_dict(export_options)}

## Snapshot ID
{snapshot_id}

Create export with:
1. **Format**: Based on export_options (Twine, JSON, HTML, etc.)
2. **Content Inclusion**:
   - Manuscript sections (based on scope)
   - Codex entries (if requested)
   - Art plans/renders (based on options)
   - Audio plans/assets (based on options)
   - Language packs (with coverage flags)
3. **Front Matter**:
   - Snapshot ID
   - Export date
   - Content coverage
   - Languages included
   - Asset inclusion flags
4. **Structure Validation**:
   - All links resolve
   - Required sections present
   - Assets referenced correctly

Note: In a real implementation, this would call the Epic 10 export module.
For now, provide export specification.

Respond in JSON format:
```json
{{
  "export_spec": {{
    "format": "format name",
    "snapshot_id": "{snapshot_id}",
    "content_summary": {{
      "sections": 0,
      "codex_entries": 0,
      "art_assets": 0,
      "audio_assets": 0,
      "languages": []
    }},
    "front_matter": {{}},
    "validation": {{
      "complete": true|false,
      "issues": []
    }}
  }}
}}
```
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "export",
                    "export_spec": data.get("export_spec", {}),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error creating export: {e}",
            )

    def _generate_front_matter(self, context: RoleContext) -> RoleResult:
        """Create metadata and coverage info."""
        system_prompt = self.build_system_prompt(context)

        snapshot_info = context.additional_context.get("snapshot_info", {})
        content_summary = context.additional_context.get("content_summary", {})

        user_prompt = f"""# Task: Generate Front Matter

{self.format_artifacts(context.artifacts)}

## Snapshot Info
{self._format_dict(snapshot_info)}

## Content Summary
{self._format_dict(content_summary)}

Create front matter with:
1. **Snapshot Metadata**:
   - Snapshot ID
   - Creation date
   - Source commit/version
2. **Content Coverage**:
   - Sections: count and completeness
   - Codex: count and coverage %
   - Choices: count
3. **Assets Included**:
   - Art: plan-only|renders|both|none
   - Audio: plan-only|assets|both|none
4. **Languages**:
   - Languages included
   - Coverage % per language
   - Translation status
5. **Quality Bars**:
   - Which bars were checked
   - Pass/fail status
   - Open issues

Respond in JSON format with complete front matter.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "front_matter",
                    "front_matter": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error generating front matter: {e}",
            )

    def _validate_export(self, context: RoleContext) -> RoleResult:
        """Check export completeness."""
        system_prompt = self.build_system_prompt(context)

        export_spec = context.additional_context.get("export_spec", {})

        user_prompt = f"""# Task: Validate Export

{self.format_artifacts(context.artifacts)}

## Export Specification
{self._format_dict(export_spec)}

Validate export:
1. **Completeness**:
   - All requested sections present
   - Assets included as specified
   - Front matter complete
2. **Structure**:
   - All internal links resolve
   - Choice labels distinct
   - Navigation flows work
3. **Assets**:
   - Referenced assets exist
   - Asset metadata present
   - Determinism logs (if promised)
4. **Languages**:
   - Language packs complete (or flagged incomplete)
   - Links resolve in all languages
   - Glossaries present

Respond in JSON format:
```json
{{
  "validation": {{
    "is_complete": true|false,
    "issues": [
      {{
        "category": "completeness|structure|assets|languages",
        "description": "Issue description",
        "severity": "blocker|warning|info",
        "affected_items": []
      }}
    ],
    "recommendations": ["Fix 1", "Fix 2"]
  }},
  "ready_for_delivery": true|false
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
                    "content_type": "export_validation",
                    "is_complete": data.get("validation", {}).get("is_complete", False),
                    "ready": data.get("ready_for_delivery", False),
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error validating export: {e}",
            )

    def _package_assets(self, context: RoleContext) -> RoleResult:
        """Bundle art/audio/translations."""
        system_prompt = self.build_system_prompt(context)

        assets = context.additional_context.get("assets", [])
        options = context.additional_context.get("options", {})

        user_prompt = f"""# Task: Package Assets

{self.format_artifacts(context.artifacts)}

## Assets to Package
{self._format_list([f"{a.get('type')}: {a.get('name')}" for a in assets])}

## Packaging Options
{self._format_dict(options)}

Package assets with:
1. **Organization**:
   - Group by type (art/audio/i18n)
   - Maintain references
   - Include metadata files
2. **Inclusion Rules**:
   - Art: plans, renders, or both
   - Audio: plans, assets, or both
   - Translations: with coverage flags
3. **Metadata**:
   - Asset manifests
   - Determinism logs
   - Attribution/licensing
4. **Validation**:
   - All referenced assets present
   - Metadata complete
   - File integrity

Respond in JSON format with packaging specification.
"""

        try:
            response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            data = self._parse_json_from_response(response)

            return RoleResult(
                success=True,
                output=response,
                metadata={
                    "content_type": "asset_package",
                    "package": data,
                },
            )

        except Exception as e:
            return RoleResult(
                success=False,
                output="",
                error=f"Error packaging assets: {e}",
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
