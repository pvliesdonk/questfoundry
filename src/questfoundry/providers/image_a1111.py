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


def _truncate_tags(text: str, limit: int) -> str:
    """Truncate a comma-separated tag string to at most *limit* tags."""
    tags = [t.strip() for t in text.split(",") if t.strip()]
    if len(tags) <= limit:
        return text
    return ", ".join(tags[:limit])


class A1111ImageProvider:
    """Image provider using Automatic1111 Stable Diffusion WebUI.

    Requires an LLM for prompt distillation — structured briefs contain
    prose-heavy descriptions that must be condensed into SD-optimised tags.

    Args:
        model: Optional SD checkpoint name (e.g., ``dreamshaper_8``).
            When set, the request includes ``override_settings.sd_model_checkpoint``.
        host: WebUI base URL. Falls back to ``A1111_HOST`` env var.
        llm: LangChain chat model for prompt distillation. Required for
            the ``PromptDistiller`` protocol; without it, ``distill_prompt``
            raises ``ImageProviderError``.
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
        """Transform a structured brief into SD-optimised prompts via LLM.

        Raises:
            ImageProviderError: If no LLM was provided at construction.
        """
        if self._llm is None:
            raise ImageProviderError(
                "a1111",
                "A1111 prompt distillation requires an LLM. "
                "Pass --provider to generate-images or set QF_PROVIDER.",
            )
        return await self._distill_with_llm(brief)

    async def _distill_with_llm(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Use an LLM to condense the brief into SD-optimised tags.

        SD CLIP has a hard token window — anything beyond it is silently
        ignored. SD 1.5: ~77 tokens (~40 tags). SDXL: ~150 tokens (~75 tags
        across two chunks separated by BREAK).
        """
        assert self._llm is not None  # guaranteed by caller
        is_xl = self._preset is _SDXL_PRESET
        entity_cap = 3 if is_xl else 2
        capped_entities = brief.entity_fragments[:entity_cap]

        brief_text = (
            f"Subject: {brief.subject}\n"
            f"Composition: {brief.composition}\n"
            f"Mood: {brief.mood}\n"
            f"Entities: {'; '.join(capped_entities) if capped_entities else 'none'}\n"
            f"Style: {brief.art_style or 'not specified'}\n"
            f"Medium: {brief.art_medium or 'not specified'}\n"
            f"Palette: {', '.join(brief.palette) if brief.palette else 'not specified'}\n"
        )
        if brief.style_overrides:
            brief_text += f"Style overrides: {brief.style_overrides}\n"

        negative_raw = ", ".join(
            n for n in [brief.negative or "", brief.negative_defaults or ""] if n
        )
        if negative_raw:
            brief_text += f"Negative: {negative_raw}\n"

        tag_limit = 75 if is_xl else 40
        if is_xl:
            format_instruction = (
                "FORMAT: <scene tags> BREAK <style tags>\n"
                "Scene chunk: subject + key entities + one composition tag + mood.\n"
                "Style chunk: art style, medium, palette, quality boosters.\n"
                "Scene chunk: ~50 tags. Style chunk: ~20 tags."
            )
            example = (
                "EXAMPLE (13 tags total — this is the right length):\n"
                "warrior on bridge, scarred face, jade pendant, wide shot, "
                "golden hour, epic BREAK watercolor, traditional paper, "
                "crimson gold palette, masterpiece, best quality\n"
                "blurry, text, watermark, deformed hands"
            )
        else:
            format_instruction = (
                "FORMAT: Single flat line of tags.\n"
                "Order: subject, entities, composition, style, mood, palette."
            )
            example = (
                "EXAMPLE (8 tags total — this is the right length):\n"
                "warrior on bridge, scarred face, wide shot, watercolor, "
                "epic mood, crimson gold palette, masterpiece, best quality\n"
                "blurry, text, watermark, deformed hands"
            )

        system_msg = (
            f"TAG BUDGET: {tag_limit} tags. You MUST use FEWER than {tag_limit}. "
            "Anything beyond this is silently discarded by SD CLIP.\n\n"
            "You are a Stable Diffusion prompt distiller. The brief below is "
            "REFERENCE MATERIAL, not a checklist. Most of it must be discarded. "
            "Your job is to extract only the visually essential elements.\n\n"
            "PRIORITY TIERS (spend your tag budget here):\n"
            "1. Subject — what is in the image (5-8 tags)\n"
            "2. Key entities — most important 1-2 characters/objects (3-5 tags)\n"
            "3. Composition — ONE camera/framing tag only (1 tag)\n"
            "4. Style/medium — art style and medium (2-4 tags)\n"
            "5. Mood/palette — only if budget remains (1-3 tags)\n"
            "6. Quality boosters — masterpiece, best quality (1-2 tags)\n\n"
            "DROP EVERYTHING ELSE. No backstory. No lighting details beyond one "
            "word. No spatial relationships. No abstract concepts.\n\n"
            "RULES:\n"
            "- Each tag is 1-4 words, comma-separated.\n"
            "- No prose, no articles, no prepositions, no sentences.\n"
            "- Output EXACTLY two lines. Line 1: positive. Line 2: negative.\n"
            "- No labels, no explanation, no commentary.\n"
            f"- {format_instruction}\n\n"
            f"{example}\n\n"
            f"REMINDER: {tag_limit} tag budget. Count before you answer. "
            "Shorter is better. Aim for 60-70% of budget, not 100%."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self._llm.ainvoke(
            [SystemMessage(content=system_msg), HumanMessage(content=brief_text)]
        )
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Parse two-line output: line 1 = positive, line 2 = negative
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        positive = lines[0] if lines else raw
        negative = lines[1] if len(lines) > 1 else None

        # Strip any accidental labels the LLM may prepend
        for prefix in ("Positive:", "positive:", "Negative:", "negative:"):
            if positive.startswith(prefix):
                positive = positive[len(prefix) :].strip()
            if negative and negative.startswith(prefix):
                negative = negative[len(prefix) :].strip()

        # Hard-truncate to CLIP token limits — LLMs routinely overshoot.
        positive = _truncate_tags(positive, tag_limit)
        if negative:
            negative = _truncate_tags(negative, 30)

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
            positive_prompt=prompt,
            negative_prompt=negative_prompt or "",
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
