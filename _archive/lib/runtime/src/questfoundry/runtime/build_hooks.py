from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class BundleResourcesHook(BuildHookInterface):
    """Ensure spec resources are bundled into the runtime package before building."""

    PLUGIN_NAME = "bundle-resources"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        project_root = Path(self.root)
        resources_dir = project_root / "src" / "questfoundry" / "runtime" / "resources"

        # If resources already exist (e.g., when building from an sdist), do nothing.
        if resources_dir.exists():
            return

        script_path = project_root / "scripts" / "bundle_resources.py"
        if not script_path.is_file():
            raise FileNotFoundError(
                f"Spec bundling script not found at {script_path}. "
                "Builds must run from the QuestFoundry monorepo with scripts/bundle_resources.py "
                "present or include pre-bundled resources."
            )

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Failed to bundle spec resources from `spec/` into "
                "`src/questfoundry/runtime/resources` during build.\n\n"
                "You can run the bundling step manually with:\n"
                "  uv run hatch run bundle\n\n"
                f"Bundler stdout:\n{result.stdout}\n"
                f"Bundler stderr:\n{result.stderr}"
            )
