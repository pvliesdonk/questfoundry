"""Asset bundling for SHIP exports.

Copies illustration image files from the project directory to the
export output directory so that relative paths in the exported
formats (Twee, HTML) resolve correctly. Also supports embedding
assets as base64 data URLs for standalone HTML exports.
"""

from __future__ import annotations

import base64
import shutil
from io import BytesIO
from typing import TYPE_CHECKING

from questfoundry.export.base import ExportIllustration
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

_ASSET_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def bundle_assets(
    illustrations: list[ExportIllustration],
    project_path: Path,
    output_dir: Path,
) -> int:
    """Copy illustration assets from project to export directory.

    Args:
        illustrations: Illustrations with relative asset paths.
        project_path: Root project directory containing source assets.
        output_dir: Export output directory to copy assets into.

    Returns:
        Number of assets successfully copied.
    """
    copied = 0
    project_resolved = project_path.resolve()
    for ill in illustrations:
        src = project_path / ill.asset_path
        src_resolved = src.resolve()
        if not src_resolved.is_relative_to(project_resolved):
            log.warning("asset_outside_project", path=str(src), passage=ill.passage_id)
            continue
        if not src_resolved.exists():
            log.warning("asset_missing", path=str(src), passage=ill.passage_id)
            continue
        dest = output_dir / ill.asset_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_resolved, dest)
        copied += 1

    if copied:
        log.info("assets_bundled", count=copied, output_dir=str(output_dir))

    return copied


def embed_assets(
    illustrations: list[ExportIllustration],
    cover: ExportIllustration | None,
    project_path: Path,
) -> tuple[list[ExportIllustration], ExportIllustration | None]:
    """Embed illustration assets as base64 data URLs.

    Args:
        illustrations: Illustrations with relative asset paths.
        cover: Optional cover illustration.
        project_path: Root project directory containing source assets.

    Returns:
        Tuple of (updated_illustrations, updated_cover).
    """
    project_resolved = project_path.resolve()
    updated_illustrations = [_embed_illustration(ill, project_resolved) for ill in illustrations]
    updated_cover = _embed_illustration(cover, project_resolved) if cover else None
    return updated_illustrations, updated_cover


def _embed_illustration(
    illustration: ExportIllustration,
    project_resolved: Path,
) -> ExportIllustration:
    if not illustration.asset_path:
        return illustration
    src = (project_resolved / illustration.asset_path).resolve()
    if not src.is_relative_to(project_resolved):
        log.warning("asset_outside_project", path=str(src), passage=illustration.passage_id)
        return illustration
    if not src.exists() or not src.is_file():
        log.warning("asset_missing", path=str(src), passage=illustration.passage_id)
        return illustration

    content_type = _guess_mime_type(src)
    if content_type is None:
        log.warning("asset_unsupported", path=str(src), passage=illustration.passage_id)
        return illustration

    data = src.read_bytes()
    if content_type == "image/png":
        data = _compress_png(data, path=str(src))

    data_url = _to_data_url(data, content_type)
    return ExportIllustration(
        passage_id=illustration.passage_id,
        asset_path=data_url,
        caption=illustration.caption,
        category=illustration.category,
    )


def _guess_mime_type(path: Path) -> str | None:
    return _ASSET_MIME_BY_EXT.get(path.suffix.lower())


def _compress_png(data: bytes, *, path: str) -> bytes:
    try:
        from PIL import Image

        buffer = BytesIO()
        with Image.open(BytesIO(data)) as img:
            img.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue()
    except Exception as exc:  # pragma: no cover - best-effort fallback
        log.warning("asset_png_compress_failed", path=path, error=str(exc))
        return data


def _to_data_url(data: bytes, content_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
