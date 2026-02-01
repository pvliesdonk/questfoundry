"""Placeholder image provider for testing.

Generates minimal solid-color PNG images with no external dependencies.
Zero cost, instant generation — ideal for development and CI.
"""

from __future__ import annotations

import hashlib
import struct
import zlib

from questfoundry.providers.image import ImageResult

# Aspect ratio → (width, height) for placeholder images.
# Kept small to minimize memory/disk usage during testing.
_ASPECT_RATIO_TO_SIZE: dict[str, tuple[int, int]] = {
    "1:1": (256, 256),
    "16:9": (640, 360),
    "9:16": (360, 640),
    "3:2": (480, 320),
    "2:3": (320, 480),
}

# Rotate through muted colors for visual distinction between placeholders.
_PALETTE: list[tuple[int, int, int]] = [
    (88, 101, 130),  # slate blue
    (130, 88, 101),  # dusty rose
    (101, 130, 88),  # sage green
    (130, 118, 88),  # warm sand
    (88, 130, 125),  # teal
    (118, 88, 130),  # muted purple
]


def _make_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Generate a minimal solid-color PNG in pure Python.

    Creates an uncompressed RGB PNG with no filtering.
    Not optimized for size — these are throwaway test images.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        r: Red channel (0-255).
        g: Green channel (0-255).
        b: Blue channel (0-255).

    Returns:
        Raw PNG bytes.
    """

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        """Build a PNG chunk with CRC."""
        payload = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + payload + crc

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR: width, height, 8-bit depth, RGB (color type 2)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: raw pixel data (filter byte 0 + RGB triplets per row)
    row = bytes([0]) + bytes([r, g, b]) * width  # filter=None + pixels
    raw_data = row * height
    compressed = zlib.compress(raw_data)
    idat = _chunk(b"IDAT", compressed)

    # IEND
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


class PlaceholderImageProvider:
    """Zero-cost image provider that generates solid-color PNGs.

    Each call produces a minimal PNG with a color selected from a
    rotating palette (based on prompt hash). The ``quality`` metadata
    is always ``"placeholder"``.

    Useful for:
    - Testing the image generation pipeline without API costs
    - CI/CD runs where real images are unnecessary
    - Validating asset storage and graph mutations
    """

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,  # noqa: ARG002
        aspect_ratio: str = "1:1",
        quality: str = "standard",  # noqa: ARG002
    ) -> ImageResult:
        """Generate a placeholder PNG.

        The color is deterministically chosen from the prompt hash,
        so the same prompt always produces the same color.

        Args:
            prompt: Text prompt (used only for color selection).
            negative_prompt: Ignored.
            aspect_ratio: Determines image dimensions.
            quality: Ignored (always produces minimal quality).

        Returns:
            ImageResult with a solid-color PNG and ``quality="placeholder"``
            in provider_metadata.
        """
        size = _ASPECT_RATIO_TO_SIZE.get(aspect_ratio, _ASPECT_RATIO_TO_SIZE["1:1"])
        width, height = size

        # Deterministic color from prompt hash
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(_PALETTE)
        r, g, b = _PALETTE[idx]

        image_data = _make_png(width, height, r, g, b)

        return ImageResult(
            image_data=image_data,
            content_type="image/png",
            provider_metadata={
                "quality": "placeholder",
                "size": f"{width}x{height}",
                "color": f"#{r:02x}{g:02x}{b:02x}",
                "prompt_preview": prompt[:80],
            },
        )
