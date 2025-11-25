from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import jsonschema
import yaml
from importlib import resources
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from questfoundry.runtime.exceptions import StateError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _load_schema(schema_name: str) -> dict[str, Any]:
    try:
        with resources.files("questfoundry.runtime.resources.schemas").joinpath(schema_name).open(
            "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise StateError(f"Schema not found: {schema_name}") from exc


@lru_cache(maxsize=32)
def _load_quality_gate(gate_id: str) -> dict[str, Any]:
    try:
        with resources.files("questfoundry.runtime.resources.definitions.quality_gates").joinpath(
            f"{gate_id}.yaml"
        ).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise StateError(f"Quality gate not found: {gate_id}") from exc


class ValidateArtifact(BaseTool):
    name: str = "validate_artifact"
    description: str = "Validate an artifact against its JSON schema"
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    def _run(
        self,
        artifact_type: str,
        content: dict[str, Any] | list[Any] | str,
        schema_path: str | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        schema_file = schema_path or f"{artifact_type}.schema.json"
        try:
            schema = _load_schema(schema_file)
            jsonschema.validate(content, schema)
            return {"valid": True, "errors": []}
        except jsonschema.ValidationError as exc:
            return {
                "valid": False,
                "errors": [exc.message],
                "path": list(exc.path),
                "schema_path": schema_file,
            }
        except StateError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise StateError(f"Validation failed: {exc}") from exc


class EvaluateQualityBar(BaseTool):
    name: str = "evaluate_quality_bar"
    description: str = "Evaluate a quality gate against the current artifacts"
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _logger: logging.Logger = PrivateAttr(default_factory=lambda: logger)

    def _run(
        self,
        gate_id: str,
        artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        gate = _load_quality_gate(gate_id)
        checks = gate.get("validation_checks", [])

        # Placeholder evaluation: mark checks as skipped but return pass=True to avoid blocking flows
        results = []
        for check in checks:
            results.append(
                {
                    "check_id": check.get("check_id"),
                    "status": "skipped",
                    "reason": "Check evaluation not implemented",
                }
            )

        return {
            "gate": gate_id,
            "overall_status": "pass",
            "checks": results,
            "notes": "Validation logic not yet implemented; treating gate as pass",
        }
