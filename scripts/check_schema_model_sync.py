#!/usr/bin/env python3
"""Check that JSON schemas and Pydantic models are in sync.

This pre-commit hook validates that schema required fields match
the corresponding Pydantic model requirements.

Usage:
    python scripts/check_schema_model_sync.py

Exit codes:
    0: All schemas in sync with models
    1: Mismatch detected between schema and model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_dream_schema_sync() -> list[str]:
    """Check dream.schema.json matches DreamArtifact model.

    Returns:
        List of error messages (empty if in sync).
    """
    errors: list[str] = []

    # Load schema
    schema_path = Path(__file__).parent.parent / "schemas" / "dream.schema.json"
    if not schema_path.exists():
        errors.append(f"Schema not found: {schema_path}")
        return errors

    with schema_path.open() as f:
        schema = json.load(f)

    # Import Pydantic models (deferred to avoid import errors in minimal envs)
    try:
        from questfoundry.artifacts import DreamArtifact, Scope
    except ImportError as e:
        errors.append(f"Cannot import models: {e}")
        return errors

    # Check top-level required fields
    schema_required = set(schema.get("required", []))
    model_required = {
        name for name, field in DreamArtifact.model_fields.items() if field.is_required()
    }
    missing_from_schema = model_required - schema_required
    if missing_from_schema:
        errors.append(
            f"{DreamArtifact.__name__} requires {missing_from_schema} but schema doesn't. "
            f"Update {schema_path.name} required array."
        )

    # Check scope required fields
    scope_schema = schema.get("properties", {}).get("scope", {})
    scope_schema_required = set(scope_schema.get("required", []))
    scope_model_required = {
        name for name, field in Scope.model_fields.items() if field.is_required()
    }
    missing_scope_fields = scope_model_required - scope_schema_required
    if missing_scope_fields:
        errors.append(
            f"{Scope.__name__} model requires {missing_scope_fields} but schema.scope doesn't. "
            f"Update {schema_path.name} scope.required array."
        )

    return errors


def main() -> int:
    """Run all schema sync checks.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    all_errors: list[str] = []

    # Check dream schema
    all_errors.extend(check_dream_schema_sync())

    # Add checks for other schemas here as they're added
    # all_errors.extend(check_brainstorm_schema_sync())
    # all_errors.extend(check_seed_schema_sync())

    if all_errors:
        print("Schema/Model sync errors found:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("All schemas in sync with Pydantic models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
