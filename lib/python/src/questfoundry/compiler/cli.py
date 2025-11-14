"""Command-line interface for the QuestFoundry spec compiler."""

import argparse
import sys
from pathlib import Path

from questfoundry.compiler.spec_compiler import CompilationError, SpecCompiler


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "QuestFoundry Spec Compiler - "
            "Transform behavior primitives into runtime artifacts"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile all
  qf-compile --spec-dir spec/ --output dist/compiled/

  # Compile specific playbook
  qf-compile --playbook lore_deepening --output dist/compiled/

  # Validate only (no output)
  qf-compile --validate-only

  # Watch mode (recompile on change)
  qf-compile --watch
        """,
    )

    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path("spec"),
        help="Root directory of spec/ (default: spec/)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/compiled"),
        help="Output directory for compiled artifacts (default: dist/compiled/)",
    )

    parser.add_argument(
        "--playbook",
        type=str,
        help="Compile only specific playbook (by ID)",
    )

    parser.add_argument(
        "--adapter",
        type=str,
        help="Compile only specific adapter (by ID)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate references without generating output",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: recompile on file changes (not yet implemented)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate spec directory exists
    if not args.spec_dir.exists():
        print(f"Error: Spec directory not found: {args.spec_dir}", file=sys.stderr)
        return 1

    behavior_dir = args.spec_dir / "05-behavior"
    if not behavior_dir.exists():
        print(
            f"Error: Behavior directory not found: {behavior_dir}",
            file=sys.stderr,
        )
        return 1

    # Initialize compiler
    compiler = SpecCompiler(args.spec_dir)

    try:
        # Load all primitives
        if args.verbose:
            print(f"Loading primitives from {behavior_dir}...")
        compiler.load_all_primitives()

        if args.verbose:
            print(f"Loaded {len(compiler.primitives)} primitives")

        # Validate
        if args.verbose:
            print("Validating references...")

        from questfoundry.compiler.validators import ReferenceValidator

        validator = ReferenceValidator(compiler.primitives, compiler.spec_root)
        errors = validator.validate_all()

        if errors:
            # Separate errors and warnings
            actual_errors = [e for e in errors if not e.startswith("Warning:")]
            warnings = [e for e in errors if e.startswith("Warning:")]

            if actual_errors:
                print("Validation errors:", file=sys.stderr)
                for error in actual_errors:
                    print(f"  ❌ {error}", file=sys.stderr)
                return 1

            if warnings:
                print("Validation warnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")

        print("✅ Validation passed")

        # If validate-only mode, exit here
        if args.validate_only:
            return 0

        # Check for watch mode
        if args.watch:
            print("Error: Watch mode not yet implemented", file=sys.stderr)
            return 1

        # Compile
        if args.playbook or args.adapter:
            # Compile specific artifact
            if args.playbook:
                if args.verbose:
                    print(f"Compiling playbook: {args.playbook}")

                result = compiler.compile_playbook(args.playbook, args.output)
                print(f"✅ Generated: {result['manifest_path']}")

            if args.adapter:
                if args.verbose:
                    print(f"Compiling adapter: {args.adapter}")

                result = compiler.compile_adapter(args.adapter, args.output)
                print(f"✅ Generated: {result['manifest_path']}")
                print(f"✅ Generated: {result['prompt_path']}")

        else:
            # Compile all
            if args.verbose:
                print(f"Compiling all to {args.output}...")

            stats = compiler.compile_all(args.output)

            print("\n📦 Compilation complete!")
            print(f"  Primitives loaded: {stats['primitives_loaded']}")
            print(f"  Playbook manifests: {stats['playbook_manifests_generated']}")
            print(f"  Adapter manifests: {stats['adapter_manifests_generated']}")
            print(f"  Standalone prompts: {stats['standalone_prompts_generated']}")
            print(f"  Output directory: {args.output}")

        return 0

    except CompilationError as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
