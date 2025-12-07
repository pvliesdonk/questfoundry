#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bundle spec resources into the runtime package for distribution.

This script copies spec files from the monorepo into the package structure
so they can be distributed with the wheel and accessed via importlib.resources.

Usage:
    python scripts/bundle_resources.py [--clean]

Options:
    --clean    Remove bundled resources instead of creating them
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

# Ensure UTF-8 encoding for stdout/stderr on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def get_paths() -> dict[str, Path]:
    """Get all relevant paths for bundling."""
    script_dir = Path(__file__).parent.resolve()
    runtime_root = script_dir.parent
    monorepo_root = runtime_root.parent.parent

    return {
        "monorepo_root": monorepo_root,
        "spec_root": monorepo_root / "spec",
        "runtime_root": runtime_root,
        "bundle_root": runtime_root / "src" / "questfoundry" / "runtime" / "resources",
    }


def validate_monorepo(spec_root: Path) -> None:
    """Validate that we're in a monorepo with the expected spec structure."""
    if not spec_root.is_dir():
        print(f"Error: spec/ directory not found at {spec_root}", file=sys.stderr)
        print("   This script must be run from within the questfoundry monorepo.", file=sys.stderr)
        sys.exit(1)

    required_layers = [
        "00-north-star",
        "01-roles",
        "02-dictionary",
        "03-schemas",
        "04-protocol",
        "05-definitions",
    ]

    for layer in required_layers:
        layer_path = spec_root / layer
        if not layer_path.is_dir():
            print(f"Error: Required layer {layer} not found at {layer_path}", file=sys.stderr)
            sys.exit(1)


def clean_bundle(bundle_root: Path) -> None:
    """Remove all bundled resources."""
    if bundle_root.exists():
        print(f"Cleaning bundled resources at {bundle_root}")
        shutil.rmtree(bundle_root)
        print("Bundled resources removed")
    else:
        print(f"No bundled resources found at {bundle_root}")


def create_bundle(spec_root: Path, bundle_root: Path) -> None:
    """Copy spec resources into the package structure."""
    # Remove existing bundle
    if bundle_root.exists():
        print(f"Removing existing bundle at {bundle_root}")
        shutil.rmtree(bundle_root)

    # Create bundle root
    bundle_root.mkdir(parents=True, exist_ok=True)

    # Layers to bundle (exclude documentation layers 0-2)
    layers_to_bundle = [
        ("03-schemas", "schemas"),
        ("04-protocol", "protocol"),
        ("05-definitions", "definitions"),
    ]

    for spec_layer, bundle_name in layers_to_bundle:
        src = spec_root / spec_layer
        dst = bundle_root / bundle_name

        if not src.exists():
            print(f"Warning: {spec_layer} not found at {src}, skipping", file=sys.stderr)
            continue

        print(f"Bundling {spec_layer} -> resources/{bundle_name}/")
        shutil.copytree(src, dst, dirs_exist_ok=True)

        # Create __init__.py to make it a package
        (dst / "__init__.py").write_text(
            f'"""Bundled spec resources from {spec_layer}."""\n',
            encoding="utf-8"
        )

        # Create __init__.py in all subdirectories to make them importable packages
        for subdir in dst.rglob("*"):
            if subdir.is_dir() and not (subdir / "__init__.py").exists():
                init_file = subdir / "__init__.py"
                rel_path = subdir.relative_to(dst)
                init_file.write_text(
                    f'"""Bundled resources from {spec_layer}/{rel_path}."""\n',
                    encoding="utf-8"
                )
                print(f"  Created __init__.py in {bundle_name}/{rel_path}/")

    # Create root __init__.py
    (bundle_root / "__init__.py").write_text(
        '"""Bundled QuestFoundry spec resources for distribution."""\n',
        encoding="utf-8"
    )

    print("Spec resources bundled successfully")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bundle spec resources into the runtime package"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove bundled resources instead of creating them"
    )
    args = parser.parse_args()

    paths = get_paths()

    if args.clean:
        clean_bundle(paths["bundle_root"])
    else:
        validate_monorepo(paths["spec_root"])
        create_bundle(paths["spec_root"], paths["bundle_root"])


if __name__ == "__main__":
    main()
