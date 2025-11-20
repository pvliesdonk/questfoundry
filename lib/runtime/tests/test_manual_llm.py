"""
Manual LLM Integration Tests

Run these tests with real API keys to verify end-to-end functionality.

Usage:
    # Test with OpenAI
    export OPENAI_API_KEY="sk-..."
    export QF_LLM_PROVIDER="openai"
    poetry run python tests/test_manual_llm.py

    # Test with Anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    export QF_LLM_PROVIDER="anthropic"
    poetry run python tests/test_manual_llm.py
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_openai_adapter():
    """Test OpenAI adapter with real API call."""
    print("\n=== Testing OpenAI Adapter ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
        return False

    try:
        from questfoundry.runtime.plugins.llm.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        llm = adapter.get_llm(model='gpt-3.5-turbo', temperature=0.1)

        print("📤 Sending: 'Say hello in exactly 3 words'")
        response = llm.invoke('Say hello in exactly 3 words')
        result = response.content if hasattr(response, 'content') else str(response)

        print(f"📥 Received: {result}")
        print("✅ OpenAI adapter working!")
        return True

    except Exception as e:
        print(f"❌ OpenAI adapter failed: {e}")
        return False


def test_anthropic_adapter():
    """Test Anthropic adapter with real API call."""
    print("\n=== Testing Anthropic Adapter ===")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic test")
        return False

    try:
        from questfoundry.runtime.plugins.llm.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()
        llm = adapter.get_llm(model='claude-3-5-haiku-20241022', temperature=0.1)

        print("📤 Sending: 'Say hello in exactly 3 words'")
        response = llm.invoke('Say hello in exactly 3 words')
        result = response.content if hasattr(response, 'content') else str(response)

        print(f"📥 Received: {result}")
        print("✅ Anthropic adapter working!")
        return True

    except Exception as e:
        print(f"❌ Anthropic adapter failed: {e}")
        return False


def test_node_factory_llm():
    """Test NodeFactory with real LLM invocation."""
    print("\n=== Testing NodeFactory LLM Integration ===")

    provider = os.getenv("QF_LLM_PROVIDER", "anthropic")
    key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"

    if not os.getenv(key_var):
        print(f"⚠️  {key_var} not set, skipping NodeFactory test")
        return False

    try:
        from questfoundry.runtime.core.node_factory import NodeFactory
        from questfoundry.runtime.core.state_manager import StateManager

        # Initialize components
        factory = NodeFactory()
        state_manager = StateManager()

        # Create initial state
        state = state_manager.initialize_state("story_spark", {
            "scene_text": "A quick test scene"
        })

        print(f"📋 Testing with provider: {provider}")
        print(f"📋 TU ID: {state['tu_id']}")

        # Create and execute a role node
        print("🔧 Creating plotwright node...")
        plotwright_node = factory.create_role_node("plotwright")

        print("🚀 Executing plotwright with real LLM...")
        result_state = plotwright_node(state)

        # Check for artifacts
        artifacts = result_state.get("artifacts", {})
        if artifacts:
            print(f"✅ NodeFactory created {len(artifacts)} artifacts")
            for key, artifact in list(artifacts.items())[:2]:  # Show first 2
                content = artifact.get("content", "")[:100]
                print(f"   • {key}: {content}...")
        else:
            print("⚠️  No artifacts created")

        print("✅ NodeFactory LLM integration working!")
        return True

    except Exception as e:
        print(f"❌ NodeFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_story_spark_loop():
    """Test complete story_spark loop execution."""
    print("\n=== Testing Story Spark Loop (End-to-End) ===")

    provider = os.getenv("QF_LLM_PROVIDER", "anthropic")
    key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"

    if not os.getenv(key_var):
        print(f"⚠️  {key_var} not set, skipping loop test")
        return False

    try:
        from questfoundry.runtime.cli.showrunner import Showrunner, ParsedIntent

        # Create showrunner
        showrunner = Showrunner()

        # Create intent
        intent = ParsedIntent(
            action="write",
            args=["A tense confrontation in the cargo bay"],
            flags={"mode": "workshop"},
            loop_id="story_spark"
        )

        print(f"📋 Provider: {provider}")
        print(f"📝 Scene: {intent.args[0]}")
        print("🚀 Executing story_spark loop...")

        # Execute
        result = showrunner.execute_request(
            "write A tense confrontation in the cargo bay",
            intent
        )

        if result.success:
            print(f"✅ Loop completed successfully!")
            print(f"📋 TU ID: {result.tu_id}")
            print(f"📊 Artifacts: {len(result.artifacts)}")
            print(f"\n{result.summary}")
            return True
        else:
            print(f"❌ Loop failed: {result.error}")
            return False

    except Exception as e:
        print(f"❌ Loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_switching():
    """Test switching between providers."""
    print("\n=== Testing Provider Switching ===")

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_openai and has_anthropic):
        print("⚠️  Both API keys needed for provider switching test")
        print(f"   OpenAI: {'✓' if has_openai else '✗'}")
        print(f"   Anthropic: {'✓' if has_anthropic else '✗'}")
        return False

    try:
        from questfoundry.runtime.core.node_factory import NodeFactory

        # Test 1: OpenAI
        os.environ["QF_LLM_PROVIDER"] = "openai"
        factory1 = NodeFactory()
        role1 = factory1.load_role("plotwright")
        config1 = factory1.select_llm(role1)

        print(f"📋 Test 1 - Provider: {config1['type']}, Model: {config1['model']}")
        assert config1["type"] == "openai", "Provider should be openai"

        # Test 2: Anthropic
        os.environ["QF_LLM_PROVIDER"] = "anthropic"
        factory2 = NodeFactory()
        role2 = factory2.load_role("plotwright")
        config2 = factory2.select_llm(role2)

        print(f"📋 Test 2 - Provider: {config2['type']}, Model: {config2['model']}")
        assert config2["type"] == "anthropic", "Provider should be anthropic"

        print("✅ Provider switching working!")
        return True

    except Exception as e:
        print(f"❌ Provider switching failed: {e}")
        return False


def main():
    """Run all manual tests."""
    print("=" * 60)
    print("QuestFoundry Runtime - Manual LLM Integration Tests")
    print("=" * 60)

    # Check environment
    provider = os.getenv("QF_LLM_PROVIDER", "anthropic")
    print(f"\n📋 Current provider: {provider}")
    print(f"📋 OPENAI_API_KEY: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    print(f"📋 ANTHROPIC_API_KEY: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")

    # Run tests
    results = []

    # Adapter tests
    results.append(("OpenAI Adapter", test_openai_adapter()))
    results.append(("Anthropic Adapter", test_anthropic_adapter()))

    # Integration tests
    results.append(("NodeFactory LLM", test_node_factory_llm()))
    results.append(("Story Spark Loop", test_story_spark_loop()))

    # Advanced tests
    results.append(("Provider Switching", test_provider_switching()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
