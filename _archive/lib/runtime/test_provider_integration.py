"""
Test script for ProviderManager integration.

Tests:
1. Provider detection
2. Tier resolution
3. Model selection workflow
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup UTF-8 console for Unicode support on Windows
from questfoundry.runtime.utils.console import setup_utf8_console, warn_git_bash
setup_utf8_console()

from questfoundry.runtime.core.provider_manager import ProviderManager
from questfoundry.runtime.core.node_factory import NodeFactory

def test_provider_detection():
    """Test provider detection via environment variables."""
    print("=" * 60)
    print("TEST 1: Provider Detection")
    print("=" * 60)

    pm = ProviderManager()
    print(f"✓ Available providers: {', '.join(pm.available_providers)}")
    print()

def test_tier_resolution():
    """Test tier resolution for different work types."""
    print("=" * 60)
    print("TEST 2: Tier Resolution")
    print("=" * 60)

    pm = ProviderManager()

    # Test different work-type tiers
    tiers = [
        "creative-writing",
        "structured-thinking",
        "validation",
        "customer-facing",
        "quick-feedback",
        "long-context"
    ]

    # Test with first available provider
    if pm.available_providers:
        provider = pm.available_providers[0]
        print(f"Testing with provider: {provider}")
        print()

        for tier in tiers:
            model = pm.resolve_model(provider, tier)
            print(f"  {tier:25} → {model}")

    print()

def test_role_tier_recommendations():
    """Test role-to-tier mapping."""
    print("=" * 60)
    print("TEST 3: Role Tier Recommendations")
    print("=" * 60)

    pm = ProviderManager()

    roles = [
        "showrunner",
        "plotwright",
        "scene_smith",
        "gatekeeper",
        "style_lead",
        "lore_weaver"
    ]

    for role in roles:
        tier = pm.get_recommended_tier(role)
        print(f"  {role:15} → {tier}")

    print()

def test_node_factory_integration():
    """Test NodeFactory integration with ProviderManager."""
    print("=" * 60)
    print("TEST 4: NodeFactory Integration")
    print("=" * 60)

    factory = NodeFactory()

    # Load a role and check LLM configuration
    role = factory.load_role("showrunner")
    llm_config = factory.select_llm(role)

    if llm_config:
        print(f"✓ Role: {role.name}")
        print(f"  Provider: {llm_config['provider']}")
        print(f"  Model: {llm_config['model']}")
        print(f"  Temperature: {llm_config['temperature']}")
        print(f"  Max Tokens: {llm_config['max_tokens']}")
    else:
        print("✗ LLM config is None (service type?)")

    print()

def test_client_creation():
    """Test LangChain client creation."""
    print("=" * 60)
    print("TEST 5: LangChain Client Creation")
    print("=" * 60)

    pm = ProviderManager()

    if not pm.available_providers:
        print("✗ No providers available")
        return

    provider = pm.available_providers[0]
    tier = "validation"  # Use fast tier for quick test
    model = pm.resolve_model(provider, tier)

    try:
        client = pm.create_llm_client(
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=100
        )
        print(f"✓ Created {provider} client")
        print(f"  Model: {model}")
        print(f"  Client type: {type(client).__name__}")
        print()

        # Test a simple invocation
        print("  Testing simple invocation...")
        response = client.invoke("Say 'Hello from QuestFoundry' and nothing else.")
        result = response.content if hasattr(response, 'content') else str(response)
        print(f"  Response: {result[:100]}")

    except Exception as e:
        print(f"✗ Client creation failed: {e}")

    print()

if __name__ == "__main__":
    print()
    print("QUESTFOUNDRY PROVIDER INTEGRATION TEST")
    print("=" * 60)
    print()

    # Warn if running in Git Bash
    warn_git_bash()

    try:
        test_provider_detection()
        test_tier_resolution()
        test_role_tier_recommendations()
        test_node_factory_integration()
        test_client_creation()

        print("=" * 60)
        print("✓ All tests completed")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
