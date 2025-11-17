"""Manifest builder for generating JSON runtime manifests."""

import logging
from datetime import UTC, datetime
from typing import Any

from questfoundry_compiler.assemblers import ReferenceResolver
from questfoundry_compiler.types import BehaviorPrimitive, CompilationError

logger = logging.getLogger(__name__)


class ManifestBuilder:
    """Build JSON manifests for runtime execution."""

    def __init__(
        self,
        primitives: dict[str, BehaviorPrimitive],
        resolver: ReferenceResolver,
    ):
        """Initialize manifest builder.

        Args:
            primitives: Dictionary of loaded primitives
            resolver: Reference resolver
        """
        self.primitives = primitives
        self.resolver = resolver
        logger.debug(f"Initialized ManifestBuilder with {len(primitives)} primitives")

    def _collect_source_files(self, primitive: BehaviorPrimitive) -> list[str]:
        """Collect source files from primitive and its references.

        Args:
            primitive: Primitive to collect sources from

        Returns:
            List of source file paths
        """
        logger.debug(f"Collecting source files for {primitive.type}:{primitive.id}")
        source_files = [str(primitive.source_path)]

        # Collect source files from referenced primitives
        for ref_type, ref_list in primitive.references.items():
            if ref_type in ["expertise", "procedure", "snippet"]:
                for ref_id in ref_list:
                    prim = self.primitives.get(f"{ref_type}:{ref_id}")
                    if prim and prim.source_path:
                        source_files.append(str(prim.source_path))

        logger.debug(f"  Collected {len(source_files)} source files")
        return source_files

    def build_playbook_manifest(self, playbook_id: str) -> dict[str, Any]:
        """Build JSON manifest for a playbook.

        Args:
            playbook_id: ID of playbook

        Returns:
            Manifest dictionary

        Raises:
            CompilationError: If manifest cannot be built
        """
        logger.debug(f"Building playbook manifest for: {playbook_id}")
        playbook = self.primitives.get(f"playbook:{playbook_id}")
        if not playbook:
            logger.error(f"Playbook not found: {playbook_id}")
            raise CompilationError(f"Playbook not found: {playbook_id}")

        data = playbook.metadata
        source_files = self._collect_source_files(playbook)

        # Build manifest
        manifest: dict[str, Any] = {
            "$schema": "https://questfoundry.liesdonk.nl/manifests/playbook_manifest.schema.json",
            "manifest_version": "2.0.0",
            "playbook_id": playbook_id,
            "display_name": data.get("playbook_name", playbook_id),
            "compiled_at": datetime.now(UTC).isoformat(),
            "source_files": source_files,
        }

        # Add category if present
        if "category" in data:
            manifest["category"] = data["category"]

        # Add purpose and outcome
        if "purpose" in data:
            manifest["purpose"] = data["purpose"]
        if "outcome" in data:
            manifest["outcome"] = data["outcome"]

        # Build steps
        steps = []
        if "procedure_steps" in data:
            for step_data in data["procedure_steps"]:
                step = self._build_step(step_data)
                if step:
                    steps.append(step)

        manifest["steps"] = steps

        # Build RACI matrix
        if "raci" in data:
            manifest["raci"] = self._build_raci(data["raci"])

        # Add quality bars
        if "quality_bars" in data:
            quality_bars = data["quality_bars"]
            if isinstance(quality_bars, dict) and "primary" in quality_bars:
                manifest["quality_bars"] = quality_bars["primary"]
            else:
                manifest["quality_bars"] = quality_bars

        # Add artifacts
        if "artifacts" in data:
            manifest["artifacts"] = data["artifacts"]

        # Add inputs and deliverables
        if "inputs" in data:
            manifest["inputs"] = data["inputs"]
        if "deliverables" in data:
            manifest["deliverables"] = data["deliverables"]

        return manifest

    def _build_step(self, step_data: dict[str, Any]) -> dict[str, Any] | None:
        """Build a single step for the manifest.

        Args:
            step_data: Step data from playbook YAML

        Returns:
            Step dictionary or None if invalid
        """
        if "name" not in step_data:
            return None

        # Validate step ID exists
        if "step" not in step_data:
            step_name = step_data.get("name", "unknown")
            raise CompilationError(f"Step missing required 'step' field: {step_name}")

        step: dict[str, Any] = {
            "step_id": f"step_{step_data['step']}",
            "name": step_data["name"],
            "description": step_data.get("action", step_data["name"]),
        }

        # Extract owner/assigned roles
        if "owner" in step_data:
            owner = step_data["owner"]
            # Extract role ID from reference like "@adapter:lore_weaver"
            if owner.startswith("@adapter:"):
                role_id = owner.replace("@adapter:", "")
                step["assigned_roles"] = [role_id]
            else:
                step["assigned_roles"] = [owner]
        else:
            step["assigned_roles"] = []

        # Add protocol intent if present
        if "protocol_intent" in step_data:
            step["protocol_intent"] = step_data["protocol_intent"]

        # Add condition if present
        if "condition" in step_data:
            step["condition"] = step_data["condition"]

        # Add example if present
        if "example" in step_data:
            step["example"] = step_data["example"]

        return step

    def _build_raci(self, raci_data: dict[str, Any]) -> dict[str, Any]:
        """Build RACI matrix for manifest.

        Args:
            raci_data: RACI data from playbook YAML

        Returns:
            RACI dictionary with cleaned role IDs
        """
        raci: dict[str, Any] = {}

        for role_type in ["responsible", "accountable", "consulted", "informed"]:
            if role_type in raci_data:
                roles = []
                for item in raci_data[role_type]:
                    if isinstance(item, dict) and "role" in item:
                        # Extract role ID from reference
                        role_ref = item["role"]
                        if role_ref.startswith("@adapter:"):
                            role_id = role_ref.replace("@adapter:", "")
                        else:
                            role_id = role_ref
                        roles.append(role_id)
                    elif isinstance(item, str):
                        roles.append(item)

                if roles:
                    raci[role_type] = roles

        return raci

    def build_adapter_manifest(self, adapter_id: str) -> dict[str, Any]:
        """Build JSON manifest for an adapter (role).

        Args:
            adapter_id: ID of adapter

        Returns:
            Manifest dictionary

        Raises:
            CompilationError: If manifest cannot be built
        """
        adapter = self.primitives.get(f"adapter:{adapter_id}")
        if not adapter:
            raise CompilationError(f"Adapter not found: {adapter_id}")

        data = adapter.metadata
        source_files = self._collect_source_files(adapter)

        # Build manifest
        manifest: dict[str, Any] = {
            "$schema": "https://questfoundry.liesdonk.nl/manifests/adapter_manifest.schema.json",
            "manifest_version": "2.0.0",
            "adapter_id": adapter_id,
            "role_name": data.get("role_name", adapter_id),
            "compiled_at": datetime.now(UTC).isoformat(),
            "source_files": source_files,
        }

        # Add mission
        if "mission" in data:
            manifest["mission"] = data["mission"]

        # Add expertise reference
        if "expertise" in data:
            manifest["expertise"] = data["expertise"]

        # Add procedures
        if "procedures" in data:
            manifest["procedures"] = data["procedures"]

        # Add protocol intents
        if "protocol_intents" in data:
            manifest["protocol_intents"] = data["protocol_intents"]

        # Add loops
        if "loops" in data:
            manifest["loops"] = data["loops"]

        # Add quality bars
        if "quality_bars" in data:
            manifest["quality_bars"] = data["quality_bars"]

        # Add safety protocols
        if "safety_protocols" in data:
            manifest["safety_protocols"] = data["safety_protocols"]

        # Add handoffs
        if "handoffs" in data:
            manifest["handoffs"] = data["handoffs"]

        # Add artifacts
        if "artifacts" in data:
            manifest["artifacts"] = data["artifacts"]

        return manifest
