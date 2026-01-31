"""Asset storage for generated images and other binary content.

Provides hash-based deduplication and organized storage under the
project's ``assets/`` directory.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

# Extension mapping from MIME content types
_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


class AssetManager:
    """Manage binary assets (images) for a project.

    Stores files in ``{project_path}/assets/`` with hash-based naming
    for automatic deduplication.

    Args:
        project_path: Root path of the project directory.
    """

    def __init__(self, project_path: Path) -> None:
        self.assets_dir = project_path / "assets"

    def store(self, data: bytes, content_type: str = "image/png") -> str:
        """Store binary data and return relative path.

        Uses SHA-256 hash prefix for filename deduplication — storing
        the same image twice returns the same path without writing.

        Args:
            data: Raw binary data to store.
            content_type: MIME type for file extension mapping.

        Returns:
            Relative path from project root (e.g., ``assets/a1b2c3d4e5f6.png``).
        """
        ext = _CONTENT_TYPE_TO_EXT.get(content_type, ".png")
        digest = hashlib.sha256(data).hexdigest()[:16]
        filename = f"{digest}{ext}"
        path = self.assets_dir / filename

        if path.exists():
            log.debug("asset_deduplicated", filename=filename)
            return f"assets/{filename}"

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        log.debug("asset_stored", filename=filename, size_bytes=len(data))
        return f"assets/{filename}"
