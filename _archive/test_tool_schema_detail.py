"""Debug tool schema conversion."""

import sys
from pathlib import Path

# Add runtime to path
runtime_path = Path(__file__).parent / "lib" / "runtime" / "src"
sys.path.insert(0, str(runtime_path))

from questfoundry.runtime.plugins.tools.registry import get_tool_registry


def test_tool_schema():
    """Examine the write_section_draft tool schema."""
    print("\n=== Examining write_section_draft Tool Schema ===\n")

    registry = get_tool_registry()

    # Get the tool
    tool_wrapper = registry.get_tool("write_section_draft")
    if not tool_wrapper:
        print("[ERROR] Tool not found in registry")
        return

    print(f"Tool wrapper type: {type(tool_wrapper).__name__}")

    # Extract BaseTool
    if hasattr(tool_wrapper, "to_langchain_tool"):
        base_tool = tool_wrapper.to_langchain_tool()
        print(f"Base tool type: {type(base_tool).__name__}")
        print(f"Tool name: {base_tool.name}")
        print(f"Tool description: {base_tool.description[:100]}...")

        # Check args_schema
        args_schema = getattr(base_tool, "args_schema", None)
        print(f"\nargs_schema: {args_schema}")

        if args_schema:
            print(f"args_schema type: {type(args_schema).__name__}")

            # Try to get JSON schema
            try:
                schema_dict = args_schema.model_json_schema()
                print(f"\nJSON Schema keys: {list(schema_dict.keys())}")
                print(f"Properties: {list(schema_dict.get('properties', {}).keys())[:10]}")
                print(f"Required: {schema_dict.get('required', [])[:5]}")

                # Show a sample property
                props = schema_dict.get('properties', {})
                if props:
                    first_prop = list(props.keys())[0]
                    print(f"\nSample property '{first_prop}':")
                    print(f"  {props[first_prop]}")
            except Exception as e:
                print(f"[ERROR] Failed to convert to JSON schema: {e}")
        else:
            print("[WARNING] Tool has no args_schema")

            # Check if tool has input schema
            if hasattr(base_tool, "get_input_schema"):
                print("\nTrying get_input_schema()...")
                try:
                    input_schema = base_tool.get_input_schema()
                    print(f"Input schema type: {type(input_schema).__name__}")
                    schema_dict = input_schema.model_json_schema()
                    print(f"JSON Schema keys: {list(schema_dict.keys())}")
                    print(f"Properties: {list(schema_dict.get('properties', {}).keys())[:10]}")
                    print(f"Required: {schema_dict.get('required', [])[:5]}")
                except Exception as e:
                    print(f"[ERROR] Failed: {e}")


if __name__ == "__main__":
    try:
        test_tool_schema()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
