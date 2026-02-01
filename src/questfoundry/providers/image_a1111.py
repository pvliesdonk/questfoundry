"""Automatic1111 (Stable Diffusion WebUI) image provider.

Generates images via the A1111 REST API (``/sdapi/v1/txt2img``).
Requires ``A1111_HOST`` environment variable pointing to a running
WebUI instance (e.g., ``http://athena:7860``).

The provider spec string selects the SD checkpoint::

    a1111                   # Use whatever checkpoint is loaded
    a1111/dreamshaper_8     # Override to dreamshaper_8
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from questfoundry.observability.logging import get_logger
from questfoundry.providers.image import (
    ImageProviderConnectionError,
    ImageProviderError,
    ImageResult,
)

if TYPE_CHECKING:
    from questfoundry.providers.image_brief import ImageBrief

log = get_logger(__name__)

_DEFAULT_TIMEOUT = 180.0  # SDXL at high res can be slow on consumer GPUs


# -- Model-aware generation presets ----------------------------------------


@dataclass(frozen=True)
class _A1111Preset:
    """Generation parameters tuned for a specific SD architecture.

    Note: A1111 splits sampler and scheduler into separate fields.
    ``sampler`` is the algorithm (e.g., "DPM++ 2M"), ``scheduler``
    controls the noise schedule (e.g., "karras").
    """

    sizes: dict[str, tuple[int, int]] = field(repr=False)
    steps: int = 30
    sampler: str = "DPM++ 2M"
    scheduler: str = "karras"
    cfg_scale: float = 7.0
    quality_tier: str = "medium"


_SD15_PRESET = _A1111Preset(
    sizes={
        "1:1": (768, 768),
        "16:9": (1024, 768),
        "9:16": (768, 1024),
        "3:2": (768, 512),
        "2:3": (512, 768),
    },
    steps=30,
    sampler="DPM++ 2M",
    scheduler="karras",
    cfg_scale=7.0,
    quality_tier="medium",
)

_SDXL_PRESET = _A1111Preset(
    sizes={
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
    },
    steps=35,
    sampler="DPM++ 2M",
    scheduler="karras",
    cfg_scale=7.5,
    quality_tier="high",
)

_XL_TAGS = ("sdxl", "xl_", "_xl", "-xl")


def _resolve_preset(model: str | None) -> _A1111Preset:
    """Choose generation preset based on checkpoint name.

    Matches models containing "sdxl", "xl_", "_xl", or "-xl"
    (case-insensitive). Examples: ``sdxl_base``, ``realvisxl_v40``,
    ``dreamshaperXL``.
    """
    if model and any(tag in model.lower() for tag in _XL_TAGS):
        return _SDXL_PRESET
    return _SD15_PRESET


class A1111ImageProvider:
    """Image provider using Automatic1111 Stable Diffusion WebUI.

    Args:
        model: Optional SD checkpoint name (e.g., ``dreamshaper_8``).
            When set, the request includes ``override_settings.sd_model_checkpoint``.
        host: WebUI base URL. Falls back to ``A1111_HOST`` env var.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        self._host = host or os.environ.get("A1111_HOST")
        if not self._host:
            raise ImageProviderError(
                "a1111",
                "A1111_HOST environment variable is required. "
                "Set it to the WebUI URL (e.g., http://localhost:7860).",
            )
        # Strip trailing slash for consistent URL construction
        self._host = self._host.rstrip("/")
        self._model = model
        self._preset = _resolve_preset(model)
        self._llm = llm
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -- PromptDistiller implementation ------------------------------------

    async def distill_prompt(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Transform a structured brief into SD-optimised prompts.

        Uses LLM distillation when an ``llm`` was provided at construction,
        otherwise falls back to rule-based tag extraction.
        """
        if self._llm is not None:
            return await self._distill_with_llm(brief)
        return self._distill_rule_based(brief)

    @staticmethod
    def _distill_rule_based(brief: ImageBrief) -> tuple[str, str | None]:
        """Extract SD-optimised tags ordered by CLIP priority.

        CLIP processes tokens left-to-right with a ~77-token window for
        SD 1.5 (SDXL extends to ~150). Tags near the front matter most.

        Priority order:
            1. Style / medium  (anchors the look)
            2. Entity fragments  (character consistency)
            3. Subject  (what the image depicts)
            4. Mood  (emotional tone)
            5. Style overrides
            6. Composition  (camera / framing — often truncated)
            7. Palette
        """
        parts: list[str] = []

        # Style anchors first — these define the visual language
        if brief.art_style:
            parts.append(brief.art_style)
        if brief.art_medium:
            parts.append(brief.art_medium)
        if brief.style_overrides:
            parts.append(brief.style_overrides)

        # Entity consistency fragments
        parts.extend(brief.entity_fragments)

        # Core content
        parts.append(brief.subject)
        if brief.mood:
            parts.append(brief.mood)

        # Lower priority — likely to be truncated on SD 1.5
        if brief.composition:
            parts.append(brief.composition)
        if brief.palette:
            parts.append(", ".join(brief.palette))

        positive = ", ".join(p for p in parts if p)

        # Truncate to ~60 words (~75 CLIP tokens)
        words = positive.split()
        if len(words) > 60:
            positive = " ".join(words[:60])

        # Negative prompt
        negative_parts = [brief.negative or "", brief.negative_defaults or ""]
        negative = ", ".join(n for n in negative_parts if n)
        return positive, negative or None

    async def _distill_with_llm(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Use an LLM to condense the brief into SD-optimised tags."""
        assert self._llm is not None  # guaranteed by caller
        brief_text = (
            f"Style: {brief.art_style or 'not specified'}\n"
            f"Medium: {brief.art_medium or 'not specified'}\n"
            f"Subject: {brief.subject}\n"
            f"Composition: {brief.composition}\n"
            f"Mood: {brief.mood}\n"
            f"Entities: {'; '.join(brief.entity_fragments) if brief.entity_fragments else 'none'}\n"
            f"Palette: {', '.join(brief.palette) if brief.palette else 'not specified'}\n"
        )
        if brief.style_overrides:
            brief_text += f"Style overrides: {brief.style_overrides}\n"

        system_msg = (
            "You are a Stable Diffusion prompt engineer. Condense the "
            "following image brief into comma-separated SD tags. "
            "Output ONLY the tags, no explanation. Maximum 60 words. "
            "Put style/medium first, then subject, then details."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self._llm.ainvoke(
            [SystemMessage(content=system_msg), HumanMessage(content=brief_text)]
        )
        positive = (
            response.content.strip() if hasattr(response, "content") else str(response).strip()
        )

        # Negative prompt — pass through as-is (SD native)
        negative_parts = [brief.negative or "", brief.negative_defaults or ""]
        negative = ", ".join(n for n in negative_parts if n)
        return positive, negative or None

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        aspect_ratio: str = "1:1",
        quality: str = "standard",  # noqa: ARG002
    ) -> ImageResult:
        """Generate an image via A1111 txt2img API.

        Args:
            prompt: Positive text prompt.
            negative_prompt: Negative prompt (natively supported by SD).
            aspect_ratio: Desired aspect ratio (e.g., "16:9").
            quality: Ignored — SD quality is controlled by steps/cfg.

        Returns:
            ImageResult with PNG data and provider metadata.

        Raises:
            ImageProviderConnectionError: If A1111 is unreachable.
            ImageProviderError: On API errors or unexpected responses.
        """
        default_size = self._preset.sizes["1:1"]
        width, height = self._preset.sizes.get(aspect_ratio, default_size)

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": self._preset.steps,
            "cfg_scale": self._preset.cfg_scale,
            "sampler_name": self._preset.sampler,
            "scheduler": self._preset.scheduler,
        }

        if self._model:
            payload["override_settings"] = {"sd_model_checkpoint": self._model}

        url = f"{self._host}/sdapi/v1/txt2img"

        log.debug(
            "a1111_generate_start",
            host=self._host,
            model=self._model,
            size=f"{width}x{height}",
            prompt_preview=prompt[:80],
        )

        try:
            response = await self._client.post(url, json=payload)
        except httpx.ConnectError as e:
            log.error("a1111_connect_error", host=self._host, error=str(e))
            raise ImageProviderConnectionError(
                "a1111", f"Cannot connect to A1111 at {self._host}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            log.error("a1111_timeout", host=self._host, timeout=_DEFAULT_TIMEOUT)
            raise ImageProviderConnectionError(
                "a1111",
                f"Request to A1111 timed out after {_DEFAULT_TIMEOUT}s: {e}",
            ) from e

        if response.status_code != 200:
            body_preview = response.text[:200]
            log.error(
                "a1111_http_error",
                status_code=response.status_code,
                body_preview=body_preview,
            )
            raise ImageProviderError(
                "a1111",
                f"A1111 returned HTTP {response.status_code}: {body_preview}",
            )

        data = response.json()
        images = data.get("images")
        if not images:
            raise ImageProviderError(
                "a1111",
                "A1111 response missing 'images' field or returned empty list",
            )

        # Extract seed and model name from response info if available
        seed = None
        active_model = self._model
        info_str = data.get("info")
        if info_str:
            try:
                info = json.loads(info_str) if isinstance(info_str, str) else info_str
                seed = info.get("seed")
                # When no checkpoint override was requested, capture the
                # active model from the response so metadata is accurate.
                if not self._model:
                    active_model = info.get("sd_model_name")
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                log.warning(
                    "a1111_info_parse_failed",
                    error=str(e),
                    info_preview=str(info_str)[:100],
                )

        metadata: dict[str, Any] = {
            "quality": self._preset.quality_tier,
            "model": active_model,
            "size": f"{width}x{height}",
            "steps": self._preset.steps,
        }
        if seed is not None:
            metadata["seed"] = seed

        log.info(
            "a1111_generate_complete",
            model=active_model,
            size=f"{width}x{height}",
            seed=seed,
        )

        return ImageResult.from_base64(
            images[0],
            content_type="image/png",
            **metadata,
        )
