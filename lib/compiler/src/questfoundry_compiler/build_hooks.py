from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SpecCopyHook(BuildHookInterface):
    """Ensure the spec tree is available inside the package before building."""

    PLUGIN_NAME = "spec-copy"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        project_root = Path(self.root)
        bundled_dir = project_root / ".spec_cache"

        repo_spec = project_root.parent.parent / "spec"
        if repo_spec.exists():
            self._copy_tree(repo_spec, bundled_dir)
            return

        sdist_spec = project_root / "spec"
        if sdist_spec.exists():
            self._copy_tree(sdist_spec, bundled_dir)
            return

        raise FileNotFoundError("spec/ directory not found for bundling")

    @staticmethod
    def _copy_tree(src: Path, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
