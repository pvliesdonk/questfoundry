from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SPEC_DIRECTORIES = (
    ("05-behavior", "05-behavior"),
    ("03-schemas", "03-schemas"),
    ("01-roles/charters", "01-roles/charters"),
)


class SpecCopyHook(BuildHookInterface):
    """Ensure the spec tree is available inside the package before building."""

    PLUGIN_NAME = "spec-copy"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        project_root = Path(self.root)
        bundled_root = project_root / "src" / "questfoundry_compiler" / "_bundled_spec"

        spec_source = self._find_spec_source(project_root)
        if spec_source is None:
            if bundled_root.exists():
                # Already bundled (e.g., when building from an sdist).
                return
            raise FileNotFoundError("spec/ directory not found for bundling")

        for source_rel, target_rel in SPEC_DIRECTORIES:
            src = spec_source / source_rel
            dest = bundled_root / target_rel
            self._copy_tree(src, dest)

    @staticmethod
    def _copy_tree(src: Path, dest: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"Missing required spec directory: {src}")

        if dest.exists():
            shutil.rmtree(dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)

    @staticmethod
    def _find_spec_source(project_root: Path) -> Path | None:
        repo_spec = project_root.parent.parent / "spec"
        if repo_spec.exists():
            return repo_spec

        sdist_spec = project_root / "spec"
        if sdist_spec.exists():
            return sdist_spec

        return None
