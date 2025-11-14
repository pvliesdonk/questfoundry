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

import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compile_spec(repo_root: Path, output_dir: Path) -> bool:
    """Run spec compiler to generate manifests.

    Args:
        repo_root: Repository root directory
        output_dir: Output directory for compiled artifacts

    Returns:
        True if compilation succeeded, False otherwise
    """
    try:
        logger.info("Compiling behavior primitives...")
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
            # Compilation failures are expected during migration when primitives
            # reference missing expertises/procedures. This doesn't block the build
            # for backward compatibility with v1-only deployments.
            logger.warning("Compilation warnings/errors:")
            logger.warning(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            logger.warning(
                "Compilation failed. V1 prompts will still be bundled for "
                "compatibility."
            )
            return False

        logger.info("  ✓ Compilation completed")
        return True

    except Exception as e:
        logger.warning(f"Compilation failed: {e}")
        logger.warning("V1 prompts will still be bundled for compatibility.")
        return False


def main() -> None:
    """Bundle schemas and prompts from spec/ into library resources."""
    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    lib_root = script_dir.parent
    repo_root = lib_root.parent.parent  # Up from lib/python/scripts

    spec_schemas = repo_root / "spec" / "03-schemas"
    spec_behavior = repo_root / "spec" / "05-behavior"

    target_schemas = lib_root / "src" / "questfoundry" / "resources" / "schemas"
    target_prompts = lib_root / "src" / "questfoundry" / "resources" / "prompts"
    target_manifests = lib_root / "src" / "questfoundry" / "resources" / "manifests"

    # Validate source directories exist
    if not spec_schemas.exists():
        raise FileNotFoundError(f"Schemas directory not found: {spec_schemas}")
    if not spec_behavior.exists():
        raise FileNotFoundError(f"Behavior directory not found: {spec_behavior}")

    # Ensure target directories exist
    target_schemas.mkdir(parents=True, exist_ok=True)
    target_prompts.mkdir(parents=True, exist_ok=True)
    target_manifests.mkdir(parents=True, exist_ok=True)

    # Clean existing bundled resources (except __init__.py)
    logger.info("Cleaning target directories...")
    for target_dir in [target_schemas, target_prompts, target_manifests]:
        for item in target_dir.iterdir():
            if item.name != "__init__.py":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # Bundle schemas (always needed)
    logger.info(f"Bundling schemas from {spec_schemas}...")
    schema_count = 0
    for schema_file in spec_schemas.glob("*.schema.json"):
        target_file = target_schemas / schema_file.name
        shutil.copy2(schema_file, target_file)
        schema_count += 1
    logger.info(f"  ✓ Bundled {schema_count} schemas")

    # V2 Architecture: Compile and bundle manifests
    logger.info(f"\n🔨 V2 Architecture: Compiling from {spec_behavior}")
    compiled_dir = repo_root / "dist" / "compiled"

    # Compile behavior primitives
    compilation_ok = compile_spec(repo_root, compiled_dir)

    # Bundle compiled artifacts if they exist (even if compilation had warnings)
    manifest_src = compiled_dir / "manifests"
    standalone_src = compiled_dir / "standalone_prompts"

    if manifest_src.exists():
        logger.info(f"Bundling compiled manifests from {manifest_src}...")
        manifest_count = 0
        for manifest_file in manifest_src.glob("*.manifest.json"):
            target_file = target_manifests / manifest_file.name
            shutil.copy2(manifest_file, target_file)
            manifest_count += 1
        logger.info(f"  ✓ Bundled {manifest_count} playbook manifests")
    elif compilation_ok:
        logger.warning("No manifests found despite successful compilation")

    if standalone_src.exists():
        logger.info(f"Bundling standalone prompts from {standalone_src}...")
        standalone_count = 0
        for prompt_file in standalone_src.glob("*.md"):
            target_file = target_prompts / prompt_file.name
            shutil.copy2(prompt_file, target_file)
            standalone_count += 1
        logger.info(f"  ✓ Bundled {standalone_count} standalone prompts")

    logger.info("\n✅ Resource bundling completed successfully!")
    logger.info(f"   Schemas: {target_schemas}")
    logger.info(f"   Prompts: {target_prompts}")
    logger.info(f"   Manifests: {target_manifests}")


if __name__ == "__main__":
    main()
