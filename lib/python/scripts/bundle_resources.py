#!/usr/bin/env python3
"""
Bundle specification resources into the Python library package.

This script copies schemas and prompts from the spec/ directory
into the library's bundled resources directory.
"""

import shutil
from pathlib import Path


def main() -> None:
    """Bundle schemas and prompts from spec/ into library resources."""
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    lib_root = script_dir.parent
    repo_root = lib_root.parent.parent  # Up from lib/python/scripts

    spec_schemas = repo_root / "spec" / "03-schemas"
    spec_prompts = repo_root / "spec" / "05-prompts"

    target_schemas = lib_root / "src" / "questfoundry" / "resources" / "schemas"
    target_prompts = lib_root / "src" / "questfoundry" / "resources" / "prompts"

    # Validate source directories exist
    if not spec_schemas.exists():
        raise FileNotFoundError(f"Schemas directory not found: {spec_schemas}")
    if not spec_prompts.exists():
        raise FileNotFoundError(f"Prompts directory not found: {spec_prompts}")

    # Clean existing bundled resources (except __init__.py)
    print(f"Cleaning target directories...")
    for target_dir in [target_schemas, target_prompts]:
        if target_dir.exists():
            for item in target_dir.iterdir():
                if item.name != "__init__.py":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

    # Bundle schemas
    print(f"Bundling schemas from {spec_schemas}...")
    schema_count = 0
    for schema_file in spec_schemas.glob("*.schema.json"):
        target_file = target_schemas / schema_file.name
        shutil.copy2(schema_file, target_file)
        schema_count += 1
    print(f"  ✓ Bundled {schema_count} schemas")

    # Bundle prompts (copy entire role directories)
    print(f"Bundling prompts from {spec_prompts}...")
    prompt_count = 0
    for role_dir in spec_prompts.iterdir():
        if role_dir.is_dir() and not role_dir.name.startswith("_"):
            target_role_dir = target_prompts / role_dir.name

            # Copy the entire role directory structure
            if target_role_dir.exists():
                shutil.rmtree(target_role_dir)
            shutil.copytree(role_dir, target_role_dir)
            prompt_count += 1
    print(f"  ✓ Bundled {prompt_count} role prompts")

    # Copy shared prompt resources if they exist
    shared_dir = spec_prompts / "_shared"
    if shared_dir.exists():
        target_shared = target_prompts / "_shared"
        if target_shared.exists():
            shutil.rmtree(target_shared)
        shutil.copytree(shared_dir, target_shared)
        print(f"  ✓ Bundled shared prompt resources")

    print("\n✅ Resource bundling completed successfully!")
    print(f"   Schemas: {target_schemas}")
    print(f"   Prompts: {target_prompts}")


if __name__ == "__main__":
    main()
