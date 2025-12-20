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
    import jsonschema
    from jsonschema import Draft202012Validator, RefResolver
except ImportError:
    print("ERROR: jsonschema not installed. Run: uv sync --extra dev")
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


def build_resolver(schema_path: Path, schema: dict) -> RefResolver:
    """Build a resolver that can handle local $ref references."""
    schema_dir = schema_path.parent
    base_uri = schema_path.as_uri()

    # Custom handler for file:// URIs
    def file_handler(uri: str) -> dict:
        # Convert file:// URI back to path
        if uri.startswith("file://"):
            path = Path(uri[7:])  # Remove file://
        else:
            # Relative reference
            path = schema_dir / uri

        with open(path) as f:
            return json.load(f)

    handlers = {"file": lambda uri: file_handler(uri)}

    return RefResolver(base_uri, schema, handlers=handlers)


def validate_file(json_file: Path, repo_root: Path) -> tuple[bool, list[str]]:
    """
    Validate a JSON file against its $schema.

    Returns (success, errors) tuple.
    """
    errors = []

    # Load the JSON file
    data = load_json(json_file)
    if data is None:
        return False, ["Failed to load JSON"]

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

    # Build resolver for $ref handling
    try:
        resolver = build_resolver(schema_path, schema)
        validator = Draft202012Validator(schema, resolver=resolver)

        # Validate (without $schema property)
        validation_errors = list(validator.iter_errors(data_to_validate))
        if validation_errors:
            for error in validation_errors:
                path = ".".join(str(p) for p in error.absolute_path) or "(root)"
                errors.append(f"  {path}: {error.message}")
            return False, errors

    except jsonschema.exceptions.RefResolutionError as e:
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

        success, errors = validate_file(json_file, repo_root)

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
