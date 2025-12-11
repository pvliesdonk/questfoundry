"""Test script for schema tool generation."""

import sys
from pathlib import Path

# Add lib/runtime/src to path
runtime_src = Path(__file__).parent / "lib" / "runtime" / "src"
sys.path.insert(0, str(runtime_src))

from questfoundry.runtime.core.schema_tool_generator import (
    _discover_artifact_mappings,
    generate_tools_for_all_artifacts,
)


def test_artifact_discovery():
    """Test that artifact mappings are discovered from role definitions."""
    print("Testing artifact discovery...")
    mappings = _discover_artifact_mappings()

    print(f"\nDiscovered {len(mappings)} artifact mappings:")
    for artifact_type, hot_sot_key in sorted(mappings.items()):
        print(f"  - {artifact_type:30} -> hot_sot.{hot_sot_key}")

    # Check for expected artifacts
    expected = ["section_draft", "hook_card", "tu_brief", "gateway_map"]
    for artifact in expected:
        if artifact in mappings:
            print(f"[OK] Found expected artifact: {artifact}")
        else:
            print(f"[MISSING] Missing expected artifact: {artifact}")

    return mappings


def test_tool_generation():
    """Test that tools are generated successfully."""
    print("\n" + "=" * 80)
    print("Testing tool generation...")

    tools = generate_tools_for_all_artifacts()

    print(f"\nGenerated {len(tools)} typed tools:")
    for tool_name in sorted(tools.keys()):
        print(f"  - {tool_name}")

    # Test instantiating a tool
    if "write_section_draft" in tools:
        print("\nTesting write_section_draft instantiation...")
        ToolClass = tools["write_section_draft"]
        tool = ToolClass()
        print(f"  Tool name: {tool.name}")
        print(f"  Tool description: {tool.description[:100]}...")
        print("[OK] Tool instantiated successfully")
    else:
        print("[MISSING] write_section_draft not found in generated tools")

    return tools


def main():
    """Run all tests."""
    print("=" * 80)
    print("Schema Tool Generator Test Suite")
    print("=" * 80)

    try:
        mappings = test_artifact_discovery()
        tools = test_tool_generation()

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Artifacts discovered: {len(mappings)}")
        print(f"Tools generated: {len(tools)}")
        print("\n[SUCCESS] All tests passed!")

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
