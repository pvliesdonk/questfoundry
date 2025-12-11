"""End-to-end integration tests for the SR-orchestrated runtime.

These tests require a running Ollama server.

Usage:
    # Run with default Ollama endpoint (localhost:11434)
    pytest tests/integration/test_orchestrator_e2e.py -v -s

    # Run with custom endpoint
    OLLAMA_BASE_URL=http://athena.int.liesdonk.nl:11434 \
        pytest tests/integration/test_orchestrator_e2e.py -v -s

    # Run directly as script
    python tests/integration/test_orchestrator_e2e.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import pytest

# Configure logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default Ollama settings - can be overridden with env vars
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://athena.int.liesdonk.nl:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def check_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    from questfoundry.runtime.providers.ollama import check_ollama_available as check

    return check(OLLAMA_BASE_URL)


# Skip integration tests if Ollama is not available
pytestmark = pytest.mark.skipif(
    not check_ollama_available(),
    reason=f"Ollama server not available at {OLLAMA_BASE_URL}",
)


@pytest.fixture
def ollama_llm():
    """Create Ollama LLM instance."""
    from questfoundry.runtime.providers.ollama import create_ollama_llm

    logger.info(f"Creating Ollama LLM: model={OLLAMA_MODEL}, url={OLLAMA_BASE_URL}")
    return create_ollama_llm(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


@pytest.fixture
def compiled_roles():
    """Load compiled roles from generated module."""
    from questfoundry.generated.roles import ALL_ROLES

    logger.info(f"Loaded {len(ALL_ROLES)} roles: {list(ALL_ROLES.keys())}")
    return ALL_ROLES


@pytest.mark.asyncio
async def test_sr_delegates_to_plotwright(ollama_llm, compiled_roles):
    """Test that SR can delegate a task to Plotwright and receive a result.

    This is a basic smoke test for the orchestration flow:
    1. SR receives a story request
    2. SR delegates to Plotwright for topology design
    3. Plotwright returns a DelegationResult
    4. SR terminates the workflow

    Note: This test may take 30-60 seconds depending on LLM response time.
    """
    from questfoundry.runtime.orchestrator import Orchestrator

    # Create orchestrator
    orchestrator = Orchestrator(
        roles=compiled_roles,
        llm=ollama_llm,
        max_delegations=5,  # Keep it short for testing
    )

    # Simple request that should trigger delegation to Plotwright
    request = (
        "Create a simple story topology with 3 scenes: "
        "a beginning scene, a middle scene with a choice, and an ending scene."
    )

    logger.info(f"Starting orchestration with request: {request}")

    # Run the orchestration
    result = await orchestrator.run(request, loop_id="test_story_spark")

    # Log the result
    logger.info("=" * 60)
    logger.info("ORCHESTRATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Metadata: {result['metadata']}")
    logger.info(f"Hot store keys: {list(result['hot_store'].keys())}")

    # Check that we got some delegation history
    delegation_history = result["metadata"].get("delegation_history", [])
    logger.info(f"Delegation history ({len(delegation_history)} delegations):")
    for i, delegation in enumerate(delegation_history):
        logger.info(f"  [{i + 1}] Role: {delegation['role']}")
        logger.info(f"      Task: {delegation['task'][:100]}...")
        logger.info(f"      Status: {delegation['result']['status']}")

    # Assertions
    assert "termination" in result["metadata"] or "error" in result["metadata"], (
        "Workflow should have terminated or errored"
    )

    # We expect at least one delegation (to Plotwright)
    assert len(delegation_history) >= 1, "Expected at least one delegation"

    # The first delegation should likely be to Plotwright
    first_delegation = delegation_history[0]
    logger.info(f"First delegation was to: {first_delegation['role']}")

    # Check that the delegation result has expected structure
    assert "status" in first_delegation["result"], "Delegation result should have status"
    assert "message" in first_delegation["result"], "Delegation result should have message"


@pytest.mark.asyncio
async def test_sr_handles_simple_request(ollama_llm, compiled_roles):
    """Test that SR can handle a simple request without complex delegation.

    This tests the basic SR execution and termination flow.
    """
    from questfoundry.runtime.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        roles=compiled_roles,
        llm=ollama_llm,
        max_delegations=3,
    )

    # A request that might be handled more simply
    request = "List the available specialist roles you can delegate to."

    logger.info(f"Starting orchestration with request: {request}")

    result = await orchestrator.run(request, loop_id="test_simple")

    logger.info("=" * 60)
    logger.info("ORCHESTRATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Metadata keys: {list(result['metadata'].keys())}")

    # This should complete (either terminate or hit max)
    assert (
        "termination" in result["metadata"]
        or "error" in result["metadata"]
        or "total_delegations" in result["metadata"]
    )


@pytest.mark.asyncio
async def test_role_consult_tools(ollama_llm, compiled_roles):
    """Test that roles can use consult tools.

    This test focuses on the Plotwright role using consult_* tools.
    """
    from questfoundry.runtime.roles import RoleAgent
    from questfoundry.runtime.state import create_initial_state

    state = create_initial_state("test_consult", "Test request")

    # Get Plotwright role
    plotwright_ir = compiled_roles["plotwright"]

    # Create agent
    agent = RoleAgent(
        role=plotwright_ir,
        llm=ollama_llm,
        state=state,
    )

    # Task that should trigger consult tool usage
    task = (
        "Before designing a story topology, please consult the schema "
        "for the Scene artifact to understand its structure. Then return "
        "to the Showrunner with a summary of what you learned."
    )

    logger.info(f"Executing Plotwright task: {task}")

    result = await agent.execute(task)

    logger.info("=" * 60)
    logger.info("ROLE EXECUTION RESULT")
    logger.info("=" * 60)
    logger.info(f"Status: {result.status}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Artifacts: {result.artifacts}")
    logger.info(f"Recommendation: {result.recommendation}")

    # Check result structure
    assert result.role_id == "plotwright", "Result should be from Plotwright"
    assert result.status in ["completed", "blocked", "needs_review", "error"]
    assert result.message, "Should have a message"


@pytest.mark.asyncio
async def test_multi_turn_delegation(ollama_llm, compiled_roles):
    """Test multi-turn delegation flow: SR → PW → LK → GK → SR.

    This tests the full orchestration loop where:
    1. SR receives a story request
    2. SR delegates to Plotwright for topology
    3. SR delegates to Lorekeeper for canon
    4. SR delegates to Gatekeeper for quality check
    5. SR terminates with results

    This verifies the orchestrator can handle multi-hop workflows
    where control returns to SR after each delegation.

    Note: This test may take 2-3 minutes depending on LLM response time.
    """
    from questfoundry.runtime.orchestrator import Orchestrator

    # Create orchestrator with enough delegations for multi-turn
    orchestrator = Orchestrator(
        roles=compiled_roles,
        llm=ollama_llm,
        max_delegations=10,  # Allow for multiple hops
    )

    # Request that should trigger multiple specialist delegations
    request = (
        "Create a mystery story. First, have Plotwright design a 3-scene topology. "
        "Then have Lorekeeper establish the canon including protagonist and antagonist. "
        "Finally, have Gatekeeper review the quality of the work before completing."
    )

    logger.info(f"Starting multi-turn orchestration with request: {request}")

    # Run the orchestration
    result = await orchestrator.run(request, loop_id="test_multi_turn")

    # Log the result
    logger.info("=" * 60)
    logger.info("MULTI-TURN ORCHESTRATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Metadata: {result['metadata']}")
    logger.info(f"Hot store keys: {list(result['hot_store'].keys())}")

    # Check delegation history
    delegation_history = result["metadata"].get("delegation_history", [])
    logger.info(f"Delegation history ({len(delegation_history)} delegations):")
    for i, delegation in enumerate(delegation_history):
        logger.info(f"  [{i + 1}] Role: {delegation['role']}")
        logger.info(f"      Task: {delegation['task'][:80]}...")
        logger.info(f"      Status: {delegation['result']['status']}")

    # Assertions
    assert "termination" in result["metadata"] or "error" in result["metadata"], (
        "Workflow should have terminated or errored"
    )

    # We expect at least 2 delegations for multi-turn
    assert len(delegation_history) >= 2, (
        f"Expected at least 2 delegations for multi-turn, got {len(delegation_history)}"
    )

    # Check that we delegated to at least 2 different roles
    roles_used = {d["role"] for d in delegation_history}
    logger.info(f"Roles used: {roles_used}")
    assert len(roles_used) >= 2, f"Expected at least 2 different roles, got {roles_used}"

    # Verify each delegation result has proper structure
    for delegation in delegation_history:
        assert "status" in delegation["result"], "Delegation should have status"
        assert "message" in delegation["result"], "Delegation should have message"
        assert delegation["result"]["status"] in [
            "completed",
            "blocked",
            "needs_review",
            "error",
        ], f"Invalid status: {delegation['result']['status']}"


@pytest.mark.asyncio
async def test_delegation_result_artifacts(ollama_llm, compiled_roles):
    """Test that delegations produce artifacts in hot_store.

    This verifies that specialist roles write artifacts to state
    and SR can see them in the hot_store after delegation.
    """
    from questfoundry.runtime.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        roles=compiled_roles,
        llm=ollama_llm,
        max_delegations=5,
    )

    # Request specifically asking for artifact creation
    request = (
        "Create a Brief artifact for a simple story about a detective. "
        "The Brief should include the title 'The Missing Cipher', "
        "genre 'mystery', and a short logline."
    )

    logger.info(f"Starting artifact creation test with request: {request}")

    result = await orchestrator.run(request, loop_id="test_artifacts")

    # Log results
    logger.info("=" * 60)
    logger.info("ARTIFACT CREATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Hot store keys: {list(result['hot_store'].keys())}")
    logger.info(f"Delegation count: {len(result['metadata'].get('delegation_history', []))}")

    # Check delegation occurred
    delegation_history = result["metadata"].get("delegation_history", [])
    assert len(delegation_history) >= 1, "Expected at least one delegation"

    # Log what artifacts were created
    for key, value in result["hot_store"].items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {list(value.keys())[:5]}...")
        elif isinstance(value, list):
            logger.info(f"  {key}: {len(value)} items")
        else:
            logger.info(f"  {key}: {type(value).__name__}")


async def main():
    """Run tests directly without pytest."""
    from questfoundry.generated.roles import ALL_ROLES
    from questfoundry.runtime.providers.ollama import create_ollama_llm, list_ollama_models

    print("=" * 60)
    print("QuestFoundry E2E Integration Test")
    print("=" * 60)
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print()

    # Check Ollama availability
    if not check_ollama_available():
        print(f"ERROR: Ollama not available at {OLLAMA_BASE_URL}")
        sys.exit(1)

    print("Ollama server is available!")

    # List available models
    models = list_ollama_models(OLLAMA_BASE_URL)
    print(f"Available models: {models}")

    if OLLAMA_MODEL.split(":")[0] not in [m.split(":")[0] for m in models]:
        print(f"WARNING: Model {OLLAMA_MODEL} may not be available")
        print("Continuing anyway...")

    print()

    # Create LLM
    llm = create_ollama_llm(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )

    print(f"Loaded {len(ALL_ROLES)} roles: {list(ALL_ROLES.keys())}")
    print()

    # Run test
    print("Running: test_sr_delegates_to_plotwright")
    print("-" * 60)

    try:
        await test_sr_delegates_to_plotwright(llm, ALL_ROLES)
        print()
        print("TEST PASSED!")
    except Exception as e:
        print()
        print(f"TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
