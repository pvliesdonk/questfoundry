#!/usr/bin/env python3
"""
Validate JSON files against their $schema references.

This script:
1. Finds all JSON files in domain-v4/ and meta/schemas/ with $schema properties
2. Resolves the $schema path relative to each file
3. Validates the file against the resolved schema
4. Reports validation errors with actionable messages

Usage:
    uv run python scripts/validate_schemas.py                    # Warn mode (exit 0)
    uv run python scripts/validate_schemas.py --strict           # Strict mode (exit 1 on failures)
    uv run python scripts/validate_schemas.py domain-v4/tools/delegate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.exceptions import Unresolvable
    import referencing.jsonschema
except ImportError:
    print("ERROR: jsonschema/referencing not installed. Run: uv sync --extra dev")
    sys.exit(1)


def load_json(path: Path) -> dict | None:
    """Load JSON file, return None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {path}: {e}")
        return None
    except OSError as e:
        print(f"  ERROR: Cannot read {path}: {e}")
        return None


def resolve_schema_path(json_file: Path, schema_ref: str) -> Path | None:
    """Resolve $schema reference relative to the JSON file."""
    if schema_ref.startswith(("http://", "https://")):
        # Skip remote schemas
        return None

    # Resolve relative path from the JSON file's directory
    schema_path = (json_file.parent / schema_ref).resolve()

    if not schema_path.exists():
        print(f"  ERROR: Schema not found: {schema_ref}")
        print(f"         Resolved to: {schema_path}")
        return None

    return schema_path


def build_registry(schema_path: Path, repo_root: Path) -> Registry:
    """Build a registry with the schema and all referenced schemas."""
    loaded_schemas: dict[str, Resource] = {}

    def load_schema_resource(path: Path) -> Resource:
        """Load a schema file and create a Resource."""
        uri = path.as_uri()
        if uri in loaded_schemas:
            return loaded_schemas[uri]

        schema = load_json(path)
        if schema is None:
            raise ValueError(f"Failed to load schema: {path}")

        resource = Resource.from_contents(
            schema, default_specification=referencing.jsonschema.DRAFT202012
        )
        loaded_schemas[uri] = resource
        return resource

    def retrieve(uri: str) -> Resource:
        """Retrieve a schema by URI for the registry."""
        # Handle file:// URIs
        if uri.startswith("file://"):
            path = Path(uri[7:])
            if not path.exists():
                raise Unresolvable(uri)
            return load_schema_resource(path)

        # Check if it's already loaded (by short name or full URI)
        for loaded_uri, resource in loaded_schemas.items():
            # Match by filename
            if loaded_uri.endswith("/" + uri) or loaded_uri.endswith("/" + uri.lstrip("../")):
                return resource

        raise Unresolvable(uri)

    # Pre-load all schemas in meta/schemas/ tree for cross-references
    meta_schemas_dir = repo_root / "meta" / "schemas"
    if meta_schemas_dir.exists():
        for schema_file in meta_schemas_dir.rglob("*.json"):
            try:
                load_schema_resource(schema_file)
            except Exception:
                pass  # Skip files that fail to load

    # Build registry with all loaded schemas
    registry = Registry(retrieve=retrieve)
    for uri, resource in loaded_schemas.items():
        registry = registry.with_resource(uri, resource)

    return registry


def validate_file(
    json_file: Path, data: dict, repo_root: Path
) -> tuple[bool, list[str]]:
    """
    Validate a JSON file's data against its $schema.

    Args:
        json_file: Path to the JSON file (for resolving relative schema refs)
        data: Pre-loaded JSON data (to avoid double-loading)
        repo_root: Repository root path

    Returns (success, errors) tuple.
    """
    errors = []

    # Check for $schema property
    schema_ref = data.get("$schema")
    if not schema_ref:
        # No schema to validate against - that's OK
        return True, []

    # Strip $schema from data before validation (it's metadata, not content)
    data_to_validate = {k: v for k, v in data.items() if k != "$schema"}

    # Resolve schema path
    schema_path = resolve_schema_path(json_file, schema_ref)
    if schema_path is None:
        if schema_ref.startswith(("http://", "https://")):
            return True, []  # Skip remote schemas
        return False, [f"Schema not found: {schema_ref}"]

    # Load schema
    schema = load_json(schema_path)
    if schema is None:
        return False, [f"Failed to load schema: {schema_path}"]

    # Build registry for $ref handling
    try:
        registry = build_registry(schema_path, repo_root)
        validator = Draft202012Validator(schema, registry=registry)

        # Validate (without $schema property)
        validation_errors = list(validator.iter_errors(data_to_validate))
        if validation_errors:
            for error in validation_errors:
                path = ".".join(str(p) for p in error.absolute_path) or "(root)"
                errors.append(f"  {path}: {error.message}")
            return False, errors

    except Unresolvable as e:
        return False, [f"Schema $ref resolution error: {e}"]
    except Exception as e:
        return False, [f"Validation error: {e}"]

    return True, []


def find_json_files(directories: list[Path]) -> list[Path]:
    """Find all JSON files in the given directories."""
    files = []
    for directory in directories:
        if directory.is_dir():
            files.extend(directory.rglob("*.json"))
        elif directory.is_file() and directory.suffix == ".json":
            files.append(directory)
    return sorted(files)


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent.resolve()

    # Parse arguments
    args = sys.argv[1:]
    strict_mode = "--strict" in args
    args = [a for a in args if a != "--strict"]

    # Determine what to validate
    if args:
        # Validate specific files
        targets = [Path(arg).resolve() for arg in args]
    else:
        # Validate all domain and meta files
        targets = [
            repo_root / "domain-v4",
            repo_root / "meta" / "schemas",
        ]

    json_files = find_json_files(targets)

    if not json_files:
        print("No JSON files found to validate")
        return 0

    print(f"Validating {len(json_files)} JSON files...")

    failed = 0
    passed = 0
    skipped = 0

    for json_file in json_files:
        rel_path = json_file.relative_to(repo_root)

        # Load to check for $schema
        data = load_json(json_file)
        if data is None:
            print(f"FAIL: {rel_path}")
            print("  ERROR: Invalid JSON")
            failed += 1
            continue

        if "$schema" not in data:
            skipped += 1
            continue

        success, errors = validate_file(json_file, data, repo_root)

        if success:
            passed += 1
        else:
            print(f"FAIL: {rel_path}")
            for error in errors:
                print(error)
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (no $schema)")

    if failed > 0:
        if strict_mode:
            print("\n❌ Validation failed (strict mode)")
            return 1
        else:
            print("\n⚠️  Validation issues found (run with --strict to fail)")
            return 0

    print("\n✅ All validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
