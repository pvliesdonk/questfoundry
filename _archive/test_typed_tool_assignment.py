"""Test that typed tools are assigned to roles based on interface.outputs."""

import sys
from pathlib import Path

# Add runtime to path
runtime_path = Path(__file__).parent / "lib" / "runtime" / "src"
sys.path.insert(0, str(runtime_path))

from questfoundry.runtime.core.runtime_context_assembler import RuntimeContextAssembler


def test_scene_smith_tools():
    """Test that Scene Smith gets write_section_draft and write_hook_card tools."""
    print("\n=== Testing Scene Smith Tool Assignment ===\n")

    # Create assembler
    assembler = RuntimeContextAssembler()

    # Create minimal state dict
    state = {
        "tu_id": "TU-2025-12-04-TEST01",
        "loop_id": "story_spark",
        "hot_sot": {},
        "cold_sot": {},
        "protocol_inbox": [],
        "protocol_outbox": [],
    }

    # Assemble context for Scene Smith
    context = assembler.assemble_context(
        role_id="scene_smith",
        loop_id="story_spark",
        node_id="test_node",
        state=state
    )

    # Extract tools
    tools = context.get("tools", [])
    print(f"Scene Smith has {len(tools)} tools\n")

    # Check for typed tools
    tool_names = []
    typed_tools = []

    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            tool_name = tool["function"]["name"]
            tool_names.append(tool_name)

            # Check if it's a typed write tool
            if tool_name.startswith("write_") and tool_name != "write_hot_sot" and tool_name != "write_cold_sot":
                typed_tools.append(tool_name)
                print(f"[TYPED TOOL] {tool_name}")
                print(f"  Description: {tool['function']['description'][:80]}...")

                # Check schema
                params = tool['function'].get('parameters', {})
                props = params.get('properties', {})
                required = params.get('required', [])
                print(f"  Parameters: {len(props)} properties, {len(required)} required")

                # Show first few required fields
                if required:
                    print(f"  Required fields: {', '.join(required[:5])}")
                    if len(required) > 5:
                        print(f"    ... and {len(required) - 5} more")
                print()

    # Expected typed tools for Scene Smith
    expected_typed_tools = ["write_section_draft", "write_hook_card"]

    print("\n=== Verification ===\n")

    for expected in expected_typed_tools:
        if expected in typed_tools:
            print(f"[OK] Found expected typed tool: {expected}")
        else:
            print(f"[MISSING] Expected typed tool not found: {expected}")

    # Check for core tools
    core_tools = ["write_hot_sot", "read_hot_sot", "consult_schema"]
    print()
    for core_tool in core_tools:
        if core_tool in tool_names:
            print(f"[OK] Found core tool: {core_tool}")
        else:
            print(f"[MISSING] Core tool not found: {core_tool}")

    print(f"\n=== Summary ===")
    print(f"Total tools: {len(tools)}")
    print(f"Typed tools: {len(typed_tools)}")
    print(f"Tool names: {', '.join(sorted(tool_names[:10]))}")
    if len(tool_names) > 10:
        print(f"  ... and {len(tool_names) - 10} more")

    # Success if we have the expected typed tools
    if all(t in typed_tools for t in expected_typed_tools):
        print("\n[SUCCESS] All expected typed tools found!")
        return True
    else:
        print("\n[FAILED] Some expected typed tools missing!")
        return False


if __name__ == "__main__":
    try:
        success = test_scene_smith_tools()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
