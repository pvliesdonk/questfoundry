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
import re
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


_STRIP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "with",
        "from",
        "into",
        "by",
        "for",
        "of",
        "to",
        "and",
        "or",
        "but",
        "as",
        "while",
        "going",
        "being",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
    ]
)

_PUNCT_SPLIT = re.compile(r"[;.]+")


def _condense_to_tags(text: str) -> str:
    """Strip articles, prepositions and filler from prose, return comma-tags.

    Normalises semicolons/periods to commas, then removes filler words
    and collapses whitespace within each tag.
    """
    if not text or not text.strip():
        return ""
    # Normalise all separators to commas, then process each tag
    normalised = _PUNCT_SPLIT.sub(",", text)
    tags = (
        " ".join(w for w in tag.strip().split() if w.lower() not in _STRIP_WORDS)
        for tag in normalised.split(",")
    )
    return ", ".join(filter(None, tags))


def _truncate_words(text: str, limit: int) -> str:
    """Truncate text to at most *limit* words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


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

    def _distill_rule_based(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Extract SD-optimised tags ordered by CLIP priority.

        CLIP processes tokens left-to-right with a ~77-token window for
        SD 1.5 (SDXL extends to ~150 via dual encoder with BREAK).
        Tags near the front matter most, so subject goes first.

        SD 1.5 (~60 words, flat):
            subject → entities → composition → style → mood → palette

        SDXL (~110 words, BREAK-separated):
            Chunk 1 (scene): subject → entities → composition → mood
            BREAK
            Chunk 2 (style): art_style → art_medium → overrides → palette → quality
        """
        is_xl = self._preset is _SDXL_PRESET
        entity_cap = 3 if is_xl else 2
        entities = [_condense_to_tags(e) for e in brief.entity_fragments[:entity_cap]]
        entities = [e for e in entities if e]

        if is_xl:
            positive = self._build_sdxl_prompt(brief, entities)
        else:
            positive = self._build_sd15_prompt(brief, entities)

        # Negative prompt — pass through as-is
        negative_parts = [brief.negative or "", brief.negative_defaults or ""]
        negative = ", ".join(n for n in negative_parts if n)
        return positive, negative or None

    @staticmethod
    def _build_sd15_prompt(brief: ImageBrief, entities: list[str]) -> str:
        """Build a flat SD 1.5 prompt (~60 words), subject-first.

        SD 1.5 CLIP has a 77-token window; ~1.25 tokens per word gives
        ~60 words.  SDXL uses 55 words per chunk (2 x 77-token encoders).
        """
        parts: list[str] = []

        # Subject first — most important for CLIP
        parts.append(_condense_to_tags(brief.subject))

        # Entity fragments (max 2, already capped)
        parts.extend(entities)

        # Composition
        if brief.composition:
            parts.append(_condense_to_tags(brief.composition))

        # Style / medium
        if brief.art_style:
            parts.append(brief.art_style)
        if brief.art_medium:
            parts.append(_condense_to_tags(brief.art_medium))

        # Mood
        if brief.mood:
            parts.append(brief.mood)

        # Style overrides
        if brief.style_overrides:
            parts.append(brief.style_overrides)

        # Palette
        if brief.palette:
            parts.append(", ".join(brief.palette))

        positive = ", ".join(p for p in parts if p)
        return _truncate_words(positive, 60)

    @staticmethod
    def _build_sdxl_prompt(brief: ImageBrief, entities: list[str]) -> str:
        """Build a BREAK-separated SDXL prompt (~110 words)."""
        # Chunk 1: scene content
        scene_parts: list[str] = []
        scene_parts.append(_condense_to_tags(brief.subject))
        scene_parts.extend(entities)
        if brief.composition:
            scene_parts.append(_condense_to_tags(brief.composition))
        if brief.mood:
            scene_parts.append(brief.mood)
        scene = ", ".join(p for p in scene_parts if p)
        scene = _truncate_words(scene, 55)

        # Chunk 2: style + quality
        style_parts: list[str] = []
        if brief.art_style:
            style_parts.append(brief.art_style)
        if brief.art_medium:
            style_parts.append(_condense_to_tags(brief.art_medium))
        if brief.style_overrides:
            style_parts.append(brief.style_overrides)
        if brief.palette:
            style_parts.append(", ".join(brief.palette))
        style_parts.append("masterpiece, best quality, highly detailed")
        style = ", ".join(p for p in style_parts if p)
        style = _truncate_words(style, 55)

        return f"{scene} BREAK {style}"

    async def _distill_with_llm(self, brief: ImageBrief) -> tuple[str, str | None]:
        """Use an LLM to condense the brief into SD-optimised tags."""
        assert self._llm is not None  # guaranteed by caller
        is_xl = self._preset is _SDXL_PRESET
        entity_cap = 3 if is_xl else 2
        capped_entities = brief.entity_fragments[:entity_cap]

        brief_text = (
            f"Style: {brief.art_style or 'not specified'}\n"
            f"Medium: {brief.art_medium or 'not specified'}\n"
            f"Subject: {brief.subject}\n"
            f"Composition: {brief.composition}\n"
            f"Mood: {brief.mood}\n"
            f"Entities: {'; '.join(capped_entities) if capped_entities else 'none'}\n"
            f"Palette: {', '.join(brief.palette) if brief.palette else 'not specified'}\n"
        )
        if brief.style_overrides:
            brief_text += f"Style overrides: {brief.style_overrides}\n"

        word_limit = 110 if is_xl else 60
        break_instruction = (
            " For SDXL, separate scene content from style with BREAK." if is_xl else ""
        )
        system_msg = (
            "You are a Stable Diffusion prompt engineer. Condense the "
            "following image brief into comma-separated SD tags. "
            "Output ONLY the tags, no explanation. "
            f"Maximum {word_limit} words. "
            "Put subject first, then entities, then composition. "
            f"Style/medium goes last.{break_instruction}"
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
