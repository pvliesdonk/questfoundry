"""Resource loader abstraction for runtime resources.

Provides a unified interface for loading compiled resources (roles, loops, schemas).
Currently reads from the `generated/` directory, but the abstraction allows
future migration to bundled resources or downloaded content.

Usage
-----
    from questfoundry.runtime.resources import ResourceLoader

    loader = ResourceLoader()
    role = loader.load_role("showrunner")
    loop = loader.load_loop("story_spark")
    schema = loader.load_schema("hook_card")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


def _find_generated_root() -> Path | None:
    """Find the generated/ directory by searching from this file."""
    current = Path(__file__).parent
    for _ in range(10):
        candidate = current / "generated"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    return None


class ResourceLoader:
    """Load compiled resources from generated/ directory.

    This class provides a clean abstraction over resource loading.
    Currently reads from filesystem, but can be extended to support:
    - Bundled resources (importlib.resources)
    - Downloaded/cached resources
    - In-memory resources for testing

    Parameters
    ----------
    root : Path | None
        Root directory for resources. If None, auto-discovers generated/.

    Examples
    --------
    Load a role definition::

        loader = ResourceLoader()
        role = loader.load_role("showrunner")
        print(role["meta"]["mandate"])

    Load a loop definition::

        loop = loader.load_loop("story_spark")
        print(loop["nodes"])
    """

    def __init__(self, root: Path | None = None):
        if root is not None:
            self.root = root
        else:
            discovered = _find_generated_root()
            if discovered is None:
                logger.warning("Could not find generated/ directory")
                self.root = Path("generated")  # Fallback, will fail gracefully
            else:
                self.root = discovered

    def load_role(self, role_id: str) -> dict[str, Any] | None:
        """Load a role definition by ID.

        Parameters
        ----------
        role_id : str
            Role identifier (e.g., "showrunner", "plotwright").

        Returns
        -------
        dict | None
            Role definition dict with meta, tools, constraints, prompt_template.
            Returns None if role not found.
        """
        # Try generated/roles/{role_id}.json
        role_path = self.root / "roles" / f"{role_id}.json"
        if role_path.exists():
            try:
                with open(role_path, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load role {role_id}: {e}")
                return None

        # Try importing from generated.roles module
        try:
            from questfoundry.generated import roles as roles_module

            # Look for ALL_ROLES dict (the actual export name)
            if hasattr(roles_module, "ALL_ROLES"):
                roles_dict = roles_module.ALL_ROLES
                if role_id in roles_dict:
                    role_ir = roles_dict[role_id]
                    return self._role_ir_to_dict(role_ir)
        except ImportError:
            pass

        logger.warning(f"Role '{role_id}' not found")
        return None

    def _role_ir_to_dict(self, role_ir: Any) -> dict[str, Any]:
        """Convert RoleIR object to dict for consistency."""
        if hasattr(role_ir, "model_dump"):
            return cast(dict[str, Any], role_ir.model_dump())
        if hasattr(role_ir, "__dict__"):
            return dict(role_ir.__dict__)
        return {"raw": str(role_ir)}

    def load_loop(self, loop_id: str) -> dict[str, Any] | None:
        """Load a loop definition by ID.

        Loop definitions are guidance for SR, not executable graphs.

        Parameters
        ----------
        loop_id : str
            Loop identifier (e.g., "story_spark").

        Returns
        -------
        dict | None
            Loop definition dict with nodes, edges, gates.
            Returns None if loop not found.
        """
        # Try generated/loops/{loop_id}.json (if we add JSON export later)
        loop_path = self.root / "loops" / f"{loop_id}.json"
        if loop_path.exists():
            try:
                with open(loop_path, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load loop {loop_id}: {e}")
                return None

        # Try importing from generated.loops module
        try:
            from questfoundry.generated import loops as loops_module

            if hasattr(loops_module, "ALL_LOOPS"):
                loops_dict = loops_module.ALL_LOOPS
                if loop_id in loops_dict:
                    loop_ir = loops_dict[loop_id]
                    return self._loop_ir_to_dict(loop_ir)
        except ImportError:
            pass

        logger.debug(f"Loop '{loop_id}' not found")
        return None

    def _loop_ir_to_dict(self, loop_ir: Any) -> dict[str, Any]:
        """Convert LoopIR object to dict for consistency."""
        if hasattr(loop_ir, "model_dump"):
            return cast(dict[str, Any], loop_ir.model_dump())
        if hasattr(loop_ir, "__dict__"):
            return dict(loop_ir.__dict__)
        return {"raw": str(loop_ir)}

    def load_schema(self, artifact_type: str) -> dict[str, Any] | None:
        """Load a JSON schema for an artifact type.

        Parameters
        ----------
        artifact_type : str
            Artifact type (e.g., "hook_card", "scene", "act").

        Returns
        -------
        dict | None
            JSON schema dict.
            Returns None if schema not found.
        """
        # Normalize filename
        schema_name = artifact_type
        if not schema_name.endswith(".schema.json"):
            schema_name = f"{artifact_type}.schema.json"

        # Try generated/schemas/{name}
        schema_path = self.root / "schemas" / schema_name
        if schema_path.exists():
            try:
                with open(schema_path, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load schema {artifact_type}: {e}")
                return None

        # Try extracting from Pydantic models
        try:
            from questfoundry.generated import models as models_module

            # Convert artifact_type to PascalCase class name
            # e.g., "hook_card" -> "HookCard", "act" -> "Act"
            class_name = "".join(word.capitalize() for word in artifact_type.split("_"))

            if hasattr(models_module, class_name):
                model_class = getattr(models_module, class_name)
                if hasattr(model_class, "model_json_schema"):
                    return cast(dict[str, Any], model_class.model_json_schema())
        except ImportError:
            pass

        logger.warning(f"Schema '{artifact_type}' not found")
        return None

    def list_roles(self) -> list[str]:
        """List all available role IDs.

        Returns
        -------
        list[str]
            List of role IDs.
        """
        roles_dir = self.root / "roles"
        if not roles_dir.exists():
            # Fallback: try to get from generated.roles module
            try:
                from questfoundry.generated import roles as roles_module

                if hasattr(roles_module, "ALL_ROLES"):
                    return list(roles_module.ALL_ROLES.keys())
            except ImportError:
                pass
            return []

        return [f.stem for f in roles_dir.glob("*.json")]

    def list_loops(self) -> list[str]:
        """List all available loop IDs.

        Returns
        -------
        list[str]
            List of loop IDs.
        """
        loops_dir = self.root / "loops"
        if not loops_dir.exists():
            return []
        return [f.stem for f in loops_dir.glob("*.json")]

    def list_schemas(self) -> list[str]:
        """List all available schema names.

        Returns
        -------
        list[str]
            List of artifact type names (without .schema.json suffix).
        """
        schemas_dir = self.root / "schemas"
        if not schemas_dir.exists():
            return []
        return [f.stem.replace(".schema", "") for f in schemas_dir.glob("*.schema.json")]


# Default singleton instance
_default_loader: ResourceLoader | None = None


def get_resource_loader() -> ResourceLoader:
    """Get the default resource loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ResourceLoader()
    return _default_loader
