"""Asset bundling for SHIP exports.

Copies illustration image files from the project directory to the
export output directory so that relative paths in the exported
formats (Twee, HTML) resolve correctly.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from questfoundry.export.base import ExportIllustration

log = get_logger(__name__)


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
    for ill in illustrations:
        src = project_path / ill.asset_path
        if not src.exists():
            log.warning("asset_missing", path=str(src), passage=ill.passage_id)
            continue
        dest = output_dir / ill.asset_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1

    if copied:
        log.info("assets_bundled", count=copied, output_dir=str(output_dir))

    return copied
