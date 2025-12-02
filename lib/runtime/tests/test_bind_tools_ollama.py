"""
Minimal test for bind_tools with Ollama Qwen3:8b.

This test explores native tool binding vs text-based Action/Action Input parsing.
Run with: uv run python tests/test_bind_tools_ollama.py
"""

import os
import json
from typing import Literal, Any

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# Define simple tools for testing
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    # Fake implementation
    return f"The weather in {location} is sunny, 72°F"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 2' or '10 * 5'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_knowledge(query: str, category: str = "general") -> str:
    """Search the knowledge base for information.

    Args:
        query: The search query
        category: Category to search in (general, science, history)
    """
    return f"Found 3 results for '{query}' in {category}: [result1, result2, result3]"


def test_bind_tools_basic():
    """Test basic bind_tools functionality with Ollama."""
    print("\n" + "="*60)
    print("TEST 1: Basic bind_tools with Ollama Qwen3:8b")
    print("="*60)

    # Get Ollama host from environment
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"Using OLLAMA_HOST: {ollama_host}")

    # Create LLM with bind_tools
    llm = ChatOllama(
        model="qwen3:8b",
        base_url=ollama_host,
        temperature=0.1,  # Low temp for deterministic tool calls
    )

    # Bind tools
    tools = [get_weather, calculate, search_knowledge]
    llm_with_tools = llm.bind_tools(tools)

    # Test prompt that should trigger tool use
    test_prompt = "What's the weather like in Seattle?"

    print(f"\nPrompt: {test_prompt}")
    print("-"*40)

    try:
        response = llm_with_tools.invoke([HumanMessage(content=test_prompt)])

        print(f"Response type: {type(response).__name__}")
        print(f"Content: {response.content[:200] if response.content else '(empty)'}")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\nTool calls detected: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"  - Tool: {tc.get('name', 'unknown')}")
                print(f"    Args: {tc.get('args', {})}")
                print(f"    ID: {tc.get('id', 'no-id')}")
            return response.tool_calls
        else:
            print("\nNo tool_calls attribute or empty")
            # Check for additional_kwargs
            if hasattr(response, 'additional_kwargs'):
                print(f"additional_kwargs: {response.additional_kwargs}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_bind_tools_multi_tool():
    """Test with a prompt that might use multiple tools."""
    print("\n" + "="*60)
    print("TEST 2: Multi-tool prompt")
    print("="*60)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model="qwen3:8b",
        base_url=ollama_host,
        temperature=0.1,
    )

    tools = [get_weather, calculate, search_knowledge]
    llm_with_tools = llm.bind_tools(tools)

    test_prompt = "Calculate 15 * 7 and also tell me what the weather is in Boston"

    print(f"\nPrompt: {test_prompt}")
    print("-"*40)

    try:
        response = llm_with_tools.invoke([HumanMessage(content=test_prompt)])

        print(f"Content: {response.content[:300] if response.content else '(empty)'}")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\nTool calls: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"  - {tc.get('name')}: {tc.get('args')}")
            return response.tool_calls
        else:
            print("No tool calls detected")
            return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tool_execution_loop():
    """Test full tool execution loop with tool responses."""
    print("\n" + "="*60)
    print("TEST 3: Full tool execution loop")
    print("="*60)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model="qwen3:8b",
        base_url=ollama_host,
        temperature=0.1,
    )

    tools = [get_weather, calculate, search_knowledge]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content="What's 25 + 17?")]

    print(f"\nInitial prompt: {messages[0].content}")
    print("-"*40)

    try:
        # Step 1: Get tool call
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        print(f"Step 1 - AI response: {response.content[:100] if response.content else '(thinking...)'}")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Tool calls: {[tc.get('name') for tc in response.tool_calls]}")

            # Step 2: Execute tools and add results
            for tc in response.tool_calls:
                tool_name = tc.get('name')
                tool_args = tc.get('args', {})
                tool_id = tc.get('id', 'call-1')

                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(tool_args)
                    print(f"  Executed {tool_name}({tool_args}) -> {result}")

                    # Add tool result
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                    ))

            # Step 3: Get final response
            final_response = llm_with_tools.invoke(messages)
            print(f"\nStep 2 - Final response: {final_response.content}")
            return True
        else:
            print("No tool calls - model answered directly")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_choice_modes():
    """Test different tool_choice settings."""
    print("\n" + "="*60)
    print("TEST 4: Tool choice modes")
    print("="*60)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model="qwen3:8b",
        base_url=ollama_host,
        temperature=0.1,
    )

    tools = [get_weather, calculate]

    test_prompt = "Hello, how are you?"  # Should NOT need tools

    modes = [
        ("auto", {"tool_choice": "auto"}),
        ("required", {"tool_choice": "required"}),
        ("none", {"tool_choice": "none"}),
    ]

    for mode_name, kwargs in modes:
        print(f"\n--- Mode: {mode_name} ---")
        try:
            llm_with_tools = llm.bind_tools(tools, **kwargs)
            response = llm_with_tools.invoke([HumanMessage(content=test_prompt)])

            has_tools = hasattr(response, 'tool_calls') and bool(response.tool_calls)
            print(f"  Has tool calls: {has_tools}")
            print(f"  Content: {response.content[:80] if response.content else '(empty)'}...")

        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def test_streaming_with_tools():
    """Test streaming response with tool calls."""
    print("\n" + "="*60)
    print("TEST 5: Streaming with tools")
    print("="*60)

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model="qwen3:8b",
        base_url=ollama_host,
        temperature=0.1,
    )

    tools = [get_weather]
    llm_with_tools = llm.bind_tools(tools)

    test_prompt = "What's the weather in Miami?"

    print(f"\nPrompt: {test_prompt}")
    print("Streaming response:")
    print("-"*40)

    try:
        chunks = []
        for chunk in llm_with_tools.stream([HumanMessage(content=test_prompt)]):
            chunks.append(chunk)
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print("\n")

        # Check if any chunks have tool calls
        tool_calls = []
        for chunk in chunks:
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        if tool_calls:
            print(f"Tool calls found in stream: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"  - {tc.get('name')}: {tc.get('args')}")
        else:
            # Check tool_call_chunks
            tool_call_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                    tool_call_chunks.extend(chunk.tool_call_chunks)

            if tool_call_chunks:
                print(f"Tool call chunks found: {len(tool_call_chunks)}")
                for tcc in tool_call_chunks:
                    print(f"  - {tcc}")
            else:
                print("No tool calls in stream")

        return bool(tool_calls)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# bind_tools Test Suite for Ollama Qwen3:8b")
    print("#"*60)

    results = {}

    # Test 1: Basic bind_tools
    results["basic"] = test_bind_tools_basic()

    # Test 2: Multi-tool prompt
    results["multi_tool"] = test_bind_tools_multi_tool()

    # Test 3: Full execution loop
    results["execution_loop"] = test_tool_execution_loop()

    # Test 4: Tool choice modes
    test_tool_choice_modes()  # This one prints its own results

    # Test 5: Streaming
    results["streaming"] = test_streaming_with_tools()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL/NO TOOLS"
        print(f"  {test_name}: {status}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
