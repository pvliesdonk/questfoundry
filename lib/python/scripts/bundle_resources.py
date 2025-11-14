#!/usr/bin/env python3
"""
Bundle specification resources into the Python library package.

This script copies schemas and prompts from the spec/ directory
into the library's bundled resources directory.

In v2 architecture:
- Compiles behavior primitives into manifests
- Bundles compiled manifests and standalone prompts
- Still supports v1 prompts for backward compatibility during migration
"""

import shutil
import subprocess
import sys
from pathlib import Path


def compile_spec(repo_root: Path, output_dir: Path) -> bool:
    """Run spec compiler to generate manifests.

    Args:
        repo_root: Repository root directory
        output_dir: Output directory for compiled artifacts

    Returns:
        True if compilation succeeded, False otherwise
    """
    try:
        print("Compiling behavior primitives...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "questfoundry.compiler.cli",
                "--spec-dir",
                str(repo_root / "spec"),
                "--output",
                str(output_dir),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"  ⚠️  Compilation warnings/errors:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            # Don't fail the build - just warn
            return False

        print("  ✓ Compilation completed")
        return True

    except Exception as e:
        print(f"  ⚠️  Compilation failed: {e}")
        return False


def main() -> None:
    """Bundle schemas and prompts from spec/ into library resources."""
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    lib_root = script_dir.parent
    repo_root = lib_root.parent.parent  # Up from lib/python/scripts

    spec_schemas = repo_root / "spec" / "03-schemas"
    spec_prompts = repo_root / "spec" / "05-prompts"
    spec_behavior = repo_root / "spec" / "05-behavior"

    target_schemas = lib_root / "src" / "questfoundry" / "resources" / "schemas"
    target_prompts = lib_root / "src" / "questfoundry" / "resources" / "prompts"
    target_manifests = lib_root / "src" / "questfoundry" / "resources" / "manifests"

    # Validate source directories exist
    if not spec_schemas.exists():
        raise FileNotFoundError(f"Schemas directory not found: {spec_schemas}")

    # Ensure target directories exist
    target_schemas.mkdir(parents=True, exist_ok=True)
    target_prompts.mkdir(parents=True, exist_ok=True)
    target_manifests.mkdir(parents=True, exist_ok=True)

    # Clean existing bundled resources (except __init__.py)
    print("Cleaning target directories...")
    for target_dir in [target_schemas, target_prompts, target_manifests]:
        for item in target_dir.iterdir():
            if item.name != "__init__.py":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # Bundle schemas (always needed)
    print(f"Bundling schemas from {spec_schemas}...")
    schema_count = 0
    for schema_file in spec_schemas.glob("*.schema.json"):
        target_file = target_schemas / schema_file.name
        shutil.copy2(schema_file, target_file)
        schema_count += 1
    print(f"  ✓ Bundled {schema_count} schemas")

    # V2 Architecture: Compile and bundle manifests
    if spec_behavior.exists():
        print(f"\n🔨 V2 Architecture detected: {spec_behavior}")
        compiled_dir = repo_root / "dist" / "compiled"

        # Compile behavior primitives
        compilation_ok = compile_spec(repo_root, compiled_dir)

        # Bundle compiled artifacts if they exist
        manifest_src = compiled_dir / "manifests"
        standalone_src = compiled_dir / "standalone_prompts"

        if manifest_src.exists():
            print(f"Bundling compiled manifests from {manifest_src}...")
            manifest_count = 0
            for manifest_file in manifest_src.glob("*.manifest.json"):
                target_file = target_manifests / manifest_file.name
                shutil.copy2(manifest_file, target_file)
                manifest_count += 1
            print(f"  ✓ Bundled {manifest_count} playbook manifests")

        if standalone_src.exists():
            print(f"Bundling standalone prompts from {standalone_src}...")
            standalone_count = 0
            for prompt_file in standalone_src.glob("*.md"):
                target_file = target_prompts / prompt_file.name
                shutil.copy2(prompt_file, target_file)
                standalone_count += 1
            print(f"  ✓ Bundled {standalone_count} standalone prompts")

    # V1 Architecture: Bundle legacy prompts (backward compatibility)
    if spec_prompts.exists():
        print(f"\nBundling legacy prompts from {spec_prompts}...")
        prompt_count = 0
        for role_dir in spec_prompts.iterdir():
            if role_dir.is_dir() and not role_dir.name.startswith("_"):
                target_role_dir = target_prompts / role_dir.name

                # Copy the entire role directory structure
                if target_role_dir.exists():
                    shutil.rmtree(target_role_dir)
                shutil.copytree(role_dir, target_role_dir)
                prompt_count += 1
        print(f"  ✓ Bundled {prompt_count} legacy role prompts")

        # Copy shared prompt resources if they exist
        shared_dir = spec_prompts / "_shared"
        if shared_dir.exists():
            target_shared = target_prompts / "_shared"
            if target_shared.exists():
                shutil.rmtree(target_shared)
            shutil.copytree(shared_dir, target_shared)
            print("  ✓ Bundled shared prompt resources")

    print("\n✅ Resource bundling completed successfully!")
    print(f"   Schemas: {target_schemas}")
    print(f"   Prompts: {target_prompts}")
    print(f"   Manifests: {target_manifests}")


if __name__ == "__main__":
    main()
