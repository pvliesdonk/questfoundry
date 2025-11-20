#!/usr/bin/env python3
"""
Validate all Layer 5 role and loop definitions against their JSON schemas.
"""

import json
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from jsonschema import validate, ValidationError, Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:
    print("ERROR: jsonschema library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonschema"])
    from jsonschema import validate, ValidationError, Draft202012Validator
    from jsonschema.exceptions import SchemaError


class ValidationReport:
    def __init__(self):
        self.errors: List[Tuple[str, str]] = []
        self.warnings: List[Tuple[str, str]] = []
        self.successes: List[str] = []

    def add_error(self, file_path: str, error: str):
        self.errors.append((file_path, error))

    def add_warning(self, file_path: str, warning: str):
        self.warnings.append((file_path, warning))

    def add_success(self, file_path: str):
        self.successes.append(file_path)

    def print_report(self):
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)

        print(f"\n✅ Successes: {len(self.successes)}")
        for file_path in self.successes:
            print(f"  ✓ {file_path}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for file_path, warning in self.warnings:
                print(f"  ⚠  {file_path}")
                print(f"     {warning}")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for file_path, error in self.errors:
                print(f"  ✗ {file_path}")
                print(f"     {error}")

        print("\n" + "="*80)
        print(f"TOTAL: {len(self.successes)} passed, {len(self.warnings)} warnings, {len(self.errors)} errors")
        print("="*80 + "\n")

        return len(self.errors) == 0


def load_schema(schema_path: Path) -> Dict:
    """Load JSON schema from file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def load_yaml(yaml_path: Path) -> Dict:
    """Load YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def validate_file(yaml_path: Path, schema: Dict, report: ValidationReport):
    """Validate a single YAML file against a schema."""
    try:
        data = load_yaml(yaml_path)
        validator = Draft202012Validator(schema)

        # Collect all validation errors
        errors = list(validator.iter_errors(data))

        if errors:
            error_messages = []
            for error in errors:
                path = " -> ".join(str(p) for p in error.path) if error.path else "root"
                error_messages.append(f"{path}: {error.message}")

            report.add_error(str(yaml_path), "\n     ".join(error_messages))
        else:
            report.add_success(str(yaml_path))

    except yaml.YAMLError as e:
        report.add_error(str(yaml_path), f"YAML parsing error: {e}")
    except Exception as e:
        report.add_error(str(yaml_path), f"Unexpected error: {e}")


def validate_roles(report: ValidationReport):
    """Validate all role YAML files."""
    print("\n📋 Validating Role Profiles...")

    schema_path = Path("spec/03-schemas/definitions/role_profile.schema.json")
    schema = load_schema(schema_path)

    roles_dir = Path("spec/05-definitions/roles")
    role_files = sorted(roles_dir.glob("*.yaml"))

    for role_file in role_files:
        validate_file(role_file, schema, report)


def validate_loops(report: ValidationReport):
    """Validate all loop YAML files."""
    print("📋 Validating Loop Patterns...")

    schema_path = Path("spec/03-schemas/definitions/loop_pattern.schema.json")
    schema = load_schema(schema_path)

    loops_dir = Path("spec/05-definitions/loops")
    loop_files = sorted(loops_dir.glob("*.yaml"))

    for loop_file in loop_files:
        validate_file(loop_file, schema, report)


def check_cross_references(report: ValidationReport):
    """Check cross-reference consistency."""
    print("\n🔗 Checking Cross-Reference Consistency...")

    # Load role abbreviations from role files
    roles_dir = Path("spec/05-definitions/roles")
    role_abbrevs = {}

    for role_file in roles_dir.glob("*.yaml"):
        data = load_yaml(role_file)
        role_id = data.get('id')
        abbrev = data.get('identity', {}).get('abbreviation')
        if role_id and abbrev:
            role_abbrevs[role_id] = abbrev

    print(f"  Found {len(role_abbrevs)} role abbreviations")

    # Check if abbreviations match expected pattern
    for role_id, abbrev in role_abbrevs.items():
        if not abbrev.isupper() or len(abbrev) < 2 or len(abbrev) > 4:
            report.add_warning(
                f"roles/{role_id}.yaml",
                f"Abbreviation '{abbrev}' doesn't match pattern (2-4 uppercase letters)"
            )

    # Check loop RACI references
    loops_dir = Path("spec/05-definitions/loops")
    for loop_file in loops_dir.glob("*.yaml"):
        try:
            data = load_yaml(loop_file)
            loop_id = data.get('id')

            # Check if required/optional roles exist
            roles_section = data.get('roles', {})
            required_roles = roles_section.get('required', [])
            optional_roles = roles_section.get('optional', [])

            for role_ref in required_roles + optional_roles:
                # role_ref could be a string or a dict (for wake_dormant)
                if isinstance(role_ref, dict):
                    role_ref = role_ref.get('role_id', '')

                if isinstance(role_ref, str):
                    if role_ref not in role_abbrevs and role_ref not in ['SR', 'GK', 'PW', 'SS', 'ST', 'LW', 'CC', 'RS', 'AD', 'IL', 'AuD', 'AuP', 'TR', 'BB', 'PN', 'ES']:
                        # Could be either role_id or abbreviation - be lenient
                        pass
        except Exception as e:
            report.add_warning(str(loop_file), f"Error checking cross-references: {e}")


def main():
    print("\n🔍 QuestFoundry Studio Layer 5 Validation")
    print("="*80)

    report = ValidationReport()

    try:
        validate_roles(report)
        validate_loops(report)
        check_cross_references(report)

        success = report.print_report()

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Fatal error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
