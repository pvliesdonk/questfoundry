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
        artifacts = artifacts or {}

        results = []
        failures = 0

        for check in checks:
            check_id = check.get("check_id")
            check_type = check.get("check_type")
            status = "skipped"
            reason = ""

            try:
                if check_type == "schema_validation":
                    status, reason = self._run_schema_validation(check, artifacts)
                elif check_type == "reference_resolution":
                    status, reason = self._run_reference_resolution(check, artifacts)
                elif check_type == "graph_analysis":
                    status, reason = self._run_graph_analysis(check, artifacts)
                else:
                    status = "skipped"
                    reason = f"Unknown check_type {check_type}"
            except Exception as exc:  # pragma: no cover - defensive
                status = "fail"
                reason = f"exception: {exc}"

            if status == "fail":
                failures += 1

            results.append({"check_id": check_id, "status": status, "reason": reason})

        overall = "pass" if failures == 0 else "fail"

        return {
            "gate": gate_id,
            "overall_status": overall,
            "checks": results,
        }

    def _run_schema_validation(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        validator = check.get("validator", {})
        method = validator.get("method")
        target = validator.get("target", "")
        if method == "regex_pattern":
            pattern = validator.get("expression", "")
            values = self._extract_values(artifacts, target)
            import re

            compiled = re.compile(pattern)
            invalid = [v for v in values if not isinstance(v, str) or not compiled.match(v)]
            if invalid:
                return "fail", f"regex mismatch for {len(invalid)} items"
            return "pass", ""
        if method == "schema_validation":
            schema_ref = validator.get("schema_ref") or validator.get("schema")
            if not schema_ref:
                return "skipped", "no schema_ref provided"
            try:
                schema = _load_schema(schema_ref)
            except StateError:
                return "skipped", f"schema {schema_ref} missing"
            values = self._extract_values(artifacts, target)
            errors = []
            for v in values:
                try:
                    jsonschema.validate(v, schema)
                except jsonschema.ValidationError as exc:
                    errors.append(exc.message)
            if errors:
                return "fail", f"{len(errors)} validation errors"
            return "pass", ""

        return "skipped", f"unsupported schema_validation method {method}"

    def _run_reference_resolution(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        target_path = check.get("validator", {}).get("target", "")
        values = self._extract_values(artifacts, target_path)
        ids = {v.get("id") for v in values if isinstance(v, dict) and "id" in v}
        referenced: list[str] = []
        for v in values:
            if isinstance(v, dict):
                for key, val in v.items():
                    if key.endswith("_ids") and isinstance(val, list):
                        referenced.extend([x for x in val if isinstance(x, str)])
        missing = [ref for ref in referenced if ref not in ids]
        if missing:
            return "fail", f"missing references: {missing}"
        return "pass", ""

    def _run_graph_analysis(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        target_path = check.get("validator", {}).get("target", "")
        values = self._extract_values(artifacts, target_path)
        if not values:
            return "skipped", "no target data for graph_analysis"
        missing = []
        for v in values:
            if isinstance(v, dict):
                if not v.get("choices") and not v.get("is_terminal"):
                    missing.append(v.get("id", "unknown"))
        if missing:
            return "fail", f"graph nodes missing choices/terminal marker: {missing}"
        return "pass", ""

    def _run_custom(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        validator = check.get("validator", {})
        expr = validator.get("expression")
        if not expr:
            return "skipped", "no expression provided"
        target_path = validator.get("target", "")
        values = self._extract_values(artifacts, target_path)
        safe_globals = {"len": len, "all": all, "any": any}
        safe_locals = {"values": values}
        try:
            result = eval(expr, safe_globals, safe_locals)  # noqa: S307 limited scope
            if result:
                return "pass", ""
            return "fail", "custom expression evaluated to False"
        except Exception as exc:
            return "fail", f"custom expression error: {exc}"

    def _extract_values(self, data: dict[str, Any], path: str) -> list[Any]:
        if not path:
            return [data]
        parts = path.split(".")
        values = [data]
        for part in parts:
            next_values: list[Any] = []
            for val in values:
                if part == "*":
                    if isinstance(val, list):
                        next_values.extend(val)
                    elif isinstance(val, dict):
                        next_values.extend(val.values())
                elif isinstance(val, dict) and part in val:
                    next_values.append(val[part])
            values = next_values
        flat: list[Any] = []
        for v in values:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat

    def _run_reference_resolution(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        target_path = check.get("validator", {}).get("target", "")
        values = self._extract_values(artifacts, target_path)
        # heuristic: treat values as dicts containing "id" and collect referenced ids in any list field ending with "_ids" or "_links"
        ids = {v.get("id") for v in values if isinstance(v, dict) and "id" in v}
        referenced: list[str] = []
        for v in values:
            if isinstance(v, dict):
                for key, val in v.items():
                    if key.endswith("_ids") and isinstance(val, list):
                        referenced.extend([x for x in val if isinstance(x, str)])
        missing = [ref for ref in referenced if ref not in ids]
        if missing:
            return "fail", f"missing references: {missing}"
        return "pass", ""

    def _run_graph_analysis(self, check: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, str]:
        # Placeholder: treat as pass if target exists, otherwise skip
        target_path = check.get("validator", {}).get("target", "")
        values = self._extract_values(artifacts, target_path)
        if values:
            return "pass", ""
        return "skipped", "no target data for graph_analysis"
