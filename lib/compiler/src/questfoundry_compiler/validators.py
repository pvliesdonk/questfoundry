"""Validators for cross-references and integrity checks."""

import logging
from pathlib import Path
from typing import Any

from questfoundry_compiler.types import BehaviorPrimitive

logger = logging.getLogger(__name__)


class ReferenceValidator:
    """Validate cross-references between primitives."""

    def __init__(self, primitives: dict[str, BehaviorPrimitive], spec_root: Path):
        """Initialize validator.

        Args:
            primitives: Dictionary of loaded primitives
            spec_root: Root directory of spec/
        """
        self.primitives = primitives
        self.spec_root = Path(spec_root)
        logger.debug(
            f"Initialized ReferenceValidator with {len(primitives)} primitives"
        )

    def validate_all(self) -> list[str]:
        """Run all validation checks.

        Returns:
            List of error messages (empty if all valid)
        """
        logger.info("Running all validation checks...")
        errors = []

        logger.debug("Validating expertise references...")
        errors.extend(self.validate_expertise_refs())

        logger.debug("Validating procedure references...")
        errors.extend(self.validate_procedure_refs())

        logger.debug("Validating schema references...")
        errors.extend(self.validate_schema_refs())

        logger.debug("Validating role references...")
        errors.extend(self.validate_role_refs())

        logger.debug("Detecting circular dependencies...")
        errors.extend(self.detect_circular_deps())

        logger.debug("Checking for orphans...")
        errors.extend(self.check_orphans())

        if errors:
            logger.warning(f"Validation completed with {len(errors)} issues")
        else:
            logger.info("All validation checks passed")

        return errors

    def validate_expertise_refs(self) -> list[str]:
        """Check all expertise references resolve.

        Returns:
            List of error messages
        """
        errors = []
        checked_count = 0
        for prim_key, primitive in self.primitives.items():
            if "expertise" in primitive.references:
                for expertise_id in primitive.references["expertise"]:
                    checked_count += 1
                    ref_key = f"expertise:{expertise_id}"
                    if ref_key not in self.primitives:
                        error_msg = f"{prim_key}: Expertise '{expertise_id}' not found"
                        errors.append(error_msg)
                        logger.debug(f"  ✗ {error_msg}")

        logger.debug(
            f"Checked {checked_count} expertise references, found {len(errors)} errors"
        )
        return errors

    def validate_procedure_refs(self) -> list[str]:
        """Check all procedure references resolve.

        Returns:
            List of error messages
        """
        errors = []
        checked_count = 0
        for prim_key, primitive in self.primitives.items():
            if "procedure" in primitive.references:
                for procedure_id in primitive.references["procedure"]:
                    checked_count += 1
                    ref_key = f"procedure:{procedure_id}"
                    if ref_key not in self.primitives:
                        error_msg = f"{prim_key}: Procedure '{procedure_id}' not found"
                        errors.append(error_msg)
                        logger.debug(f"  ✗ {error_msg}")

        logger.debug(
            f"Checked {checked_count} procedure references, found {len(errors)} errors"
        )
        return errors

    def validate_schema_refs(self) -> list[str]:
        """Check all schema references point to valid L3 schemas.

        Returns:
            List of error messages
        """
        errors = []
        schema_dir = self.spec_root / "03-schemas"
        checked_count = 0

        for prim_key, primitive in self.primitives.items():
            if "schema" in primitive.references:
                for schema_id in primitive.references["schema"]:
                    checked_count += 1
                    schema_path = schema_dir / schema_id
                    if not schema_path.exists():
                        error_msg = (
                            f"{prim_key}: Schema '{schema_id}' "
                            f"not found at {schema_path}"
                        )
                        errors.append(error_msg)
                        logger.debug(f"  ✗ {error_msg}")

        logger.debug(
            f"Checked {checked_count} schema references, found {len(errors)} errors"
        )
        return errors

    def validate_role_refs(self) -> list[str]:
        """Check all role references match L1 role definitions.

        Returns:
            List of error messages
        """
        errors = []
        roles_dir = self.spec_root / "01-roles" / "charters"

        for prim_key, primitive in self.primitives.items():
            if "role" in primitive.references:
                for role_id in primitive.references["role"]:
                    role_path = roles_dir / f"{role_id}.md"
                    if not role_path.exists():
                        errors.append(
                            f"{prim_key}: Role '{role_id}' not found at {role_path}"
                        )
        return errors

    def detect_circular_deps(self) -> list[str]:
        """Detect circular dependencies in references.

        Playbook <-> Adapter cycles are allowed (adapters participate in playbooks,
        playbooks reference adapters), so we exclude those from cycle detection.

        Returns:
            List of error messages describing circular dependencies
        """
        errors = []

        # Build dependency graph, excluding playbook <-> adapter edges
        dep_graph: dict[str, set[str]] = {}
        for prim_key, primitive in self.primitives.items():
            deps = set()
            for ref_type, ref_list in primitive.references.items():
                # Only check behavior primitive references (not schemas/roles)
                if ref_type in [
                    "expertise",
                    "procedure",
                    "snippet",
                    "playbook",
                    "adapter",
                ]:
                    for ref_id in ref_list:
                        dep_key = f"{ref_type}:{ref_id}"

                        # Skip playbook <-> adapter cycles (these are expected)
                        if primitive.type == "playbook" and ref_type == "adapter":
                            continue
                        if primitive.type == "adapter" and ref_type == "playbook":
                            continue

                        deps.add(dep_key)
            dep_graph[prim_key] = deps

        # Detect cycles using DFS
        def has_cycle(
            node: str, visited: set[str], rec_stack: set[str], path: list[str]
        ) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all dependencies
            if node in dep_graph:
                for neighbor in dep_graph[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack, path):
                            return True
                    elif neighbor in rec_stack:
                        # Found cycle - record it
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        errors.append(
                            f"Circular dependency detected: {' -> '.join(cycle)}"
                        )
                        return True

            path.pop()
            rec_stack.remove(node)
            return False

        visited: set[str] = set()
        for node in dep_graph:
            if node not in visited:
                has_cycle(node, visited, set(), [])

        return errors

    def check_orphans(self) -> list[str]:
        """Find primitives not referenced by any playbook/adapter.

        Returns:
            List of warning messages (not errors)
        """
        warnings = []

        # Build set of all primitives referenced by playbooks and adapters
        referenced: set[str] = set()
        for primitive in self.primitives.values():
            # Only count references from playbooks and adapters
            if primitive.type in ["playbook", "adapter"]:
                for ref_type, ref_list in primitive.references.items():
                    if ref_type in ["expertise", "procedure", "snippet"]:
                        for ref_id in ref_list:
                            referenced.add(f"{ref_type}:{ref_id}")

        # Find orphans (non-root primitives not referenced by roots)
        for prim_key, primitive in self.primitives.items():
            # Skip playbooks and adapters (they are roots)
            if primitive.type in ["playbook", "adapter"]:
                continue

            if prim_key not in referenced:
                warnings.append(
                    f"Warning: {prim_key} is not referenced by any playbook or adapter"
                )

        return warnings


def validate_manifest_structure(manifest: dict[str, Any]) -> list[str]:
    """Validate manifest structure against expected schema.

    Args:
        manifest: Manifest dictionary to validate

    Returns:
        List of error messages
    """
    errors = []

    # Required fields
    required_fields = [
        "manifest_version",
        "playbook_id",
        "display_name",
        "compiled_at",
    ]

    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")

    # Version check
    if "manifest_version" in manifest:
        version = manifest["manifest_version"]
        if not version.startswith("2."):
            errors.append(f"Invalid manifest_version: {version} (expected 2.x.x)")

    # Steps validation
    if "steps" in manifest:
        if not isinstance(manifest["steps"], list):
            errors.append("'steps' must be a list")
        else:
            for i, step in enumerate(manifest["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")
                    continue

                step_required = ["step_id", "description", "assigned_roles"]
                for field in step_required:
                    if field not in step:
                        errors.append(f"Step {i}: Missing required field '{field}'")

    return errors
