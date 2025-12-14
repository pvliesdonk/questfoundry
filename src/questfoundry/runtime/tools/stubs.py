"""
Stub implementations for tools not yet implemented.

These tools return "not implemented" responses and are placeholders
for future implementation phases.

Tools:
- generate_image: Image generation (Phase N)
- generate_audio: Audio generation (Phase N)
- assemble_export: Export assembly (Phase N)
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool


@register_tool("generate_image")
class GenerateImageTool(BaseTool):
    """
    Generate an image based on a prompt.

    STUB IMPLEMENTATION: Not yet implemented.
    Will integrate with image generation API in future phase.
    """

    def check_availability(self) -> bool:
        return False

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        prompt = args.get("prompt", "")

        return ToolResult(
            success=False,
            data={
                "prompt": prompt,
                "placeholder": True,
            },
            error=(
                "Image generation not implemented. "
                "This feature requires integration with an image generation service."
            ),
        )


@register_tool("generate_audio")
class GenerateAudioTool(BaseTool):
    """
    Generate audio (music or sound effects) based on a prompt.

    STUB IMPLEMENTATION: Not yet implemented.
    Will integrate with audio generation API in future phase.
    """

    def check_availability(self) -> bool:
        return False

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        prompt = args.get("prompt", "")
        audio_type = args.get("type", "music")  # music or sfx

        return ToolResult(
            success=False,
            data={
                "prompt": prompt,
                "type": audio_type,
                "placeholder": True,
            },
            error=(
                "Audio generation not implemented. "
                "This feature requires integration with an audio generation service."
            ),
        )


@register_tool("assemble_export")
class AssembleExportTool(BaseTool):
    """
    Assemble project artifacts into an export package.

    STUB IMPLEMENTATION: Not yet implemented.
    Will create static exports (HTML, ePub, etc.) in future phase.
    """

    def check_availability(self) -> bool:
        return False

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        export_format = args.get("format", "html")
        include_assets = args.get("include_assets", True)

        return ToolResult(
            success=False,
            data={
                "format": export_format,
                "include_assets": include_assets,
                "placeholder": True,
            },
            error=(
                "Export assembly not implemented. This feature will be available in a future phase."
            ),
        )
