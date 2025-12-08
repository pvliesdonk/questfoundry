"""RoleAgent - executes specialist roles with their own conversation history.

Each role runs as an independent agent that:
- Has its own conversation history (maintained across delegations)
- Uses tools specific to its function + common consult tools
- Returns control to SR via return_to_sr tool
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from questfoundry.compiler.models import RoleIR
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.state import DelegationResult, StudioState
from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
)
from questfoundry.runtime.tools.gatekeeper import (
    CreateGatecheckReport,
    EvaluateAccessibility,
    EvaluateDeterminism,
    EvaluateGateways,
    EvaluateIntegrity,
    EvaluateNonlinearity,
    EvaluatePresentation,
    EvaluateReachability,
    EvaluateStyle,
)
from questfoundry.runtime.tools.role import (
    PromoteToCanon,
    ReadColdSot,
    ReadHotSot,
    ReturnToSR,
    WriteHotSot,
)
from questfoundry.runtime.tracing import trace_role_execution

if TYPE_CHECKING:
    from questfoundry.runtime.stores import ColdStore

logger = logging.getLogger(__name__)


def _render_gatekeeper_prompt(role: RoleIR) -> str:
    """Render Gatekeeper-specific prompt with quality bar tools."""
    constraints_section = ""
    if role.constraints:
        constraints_lines = "\n".join(f"- {c}" for c in role.constraints)
        constraints_section = f"""
## Constraints

{constraints_lines}
"""

    return f"""You are the **{role.archetype}** ({role.abbr}), the quality auditor in QuestFoundry.

## Your Mandate

**{role.mandate}**

{constraints_section}
## CRITICAL: You Must Use Tools

You are a TOOL-USING agent. You MUST use the tools provided to validate artifacts.
DO NOT just describe what you would check - actually CHECK IT by calling evaluation tools.

## Your Tools

### Quality Bar Evaluation Tools (8 bars)
- **evaluate_integrity(artifact_id)**: Check for contradictions in canon
- **evaluate_reachability(artifact_id)**: Check all content is accessible via valid paths
- **evaluate_nonlinearity(artifact_id)**: Check multiple valid paths exist
- **evaluate_gateways(artifact_id)**: Check all gates have valid unlock conditions
- **evaluate_style(artifact_id)**: Check voice and tone consistency
- **evaluate_determinism(artifact_id)**: Check same inputs produce same outputs
- **evaluate_presentation(artifact_id)**: Check formatting and structure
- **evaluate_accessibility(artifact_id)**: Check content is usable by all players

### Report Tool
- **create_gatecheck_report(target_artifact, bars_checked, status, bar_results, issues, recommendations)**: Create formal validation report

### Promotion Tool (use ONLY after all bars pass)
- **promote_to_canon(artifact_ids, snapshot_description)**: Move validated artifacts to cold store

### State Tools
- **read_hot_sot(key)**: Read artifacts to validate from hot_store
- **read_cold_sot(key)**: Read from cold_store for comparison

### Knowledge Tools
- **consult_schema(artifact_type)**: Look up artifact field requirements
- **consult_playbook(query)**: Get workflow guidance

### Completion Tool (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner

## Validation Workflow

1. **Read the artifact** to validate using read_hot_sot
2. **Identify which bars apply** (from the task or Brief)
3. **Evaluate each bar** using the appropriate evaluate_* tool
4. **Collect findings** from each evaluation
5. **Create a GatecheckReport** documenting all results
6. **If ALL bars pass**: Call promote_to_canon, then return_to_sr with status="completed"
7. **If ANY bar fails**: Return_to_sr with status="needs_review" and list issues

## Example Workflow

```
# Read the artifact to validate
read_hot_sot("topology_001")

# Evaluate required bars
evaluate_integrity("topology_001")
evaluate_reachability("topology_001")
evaluate_nonlinearity("topology_001")

# Create report
create_gatecheck_report(
    target_artifact="topology_001",
    bars_checked=["integrity", "reachability", "nonlinearity"],
    status="passed",  # or "failed"
    bar_results={{"integrity": "PASS - no contradictions", "reachability": "PASS - all scenes accessible", "nonlinearity": "PASS - 3 distinct paths"}},
    issues=[],  # or list of issues if failed
    recommendations=[]
)

# If passed, promote and return
promote_to_canon(artifact_ids=["topology_001"], snapshot_description="Validated topology")
return_to_sr(status="completed", message="Topology validated and promoted to canon", artifacts=["gatecheck_topology_001_1"])
```

## Important

- ALWAYS evaluate ALL requested bars before making a decision
- ALWAYS create a GatecheckReport documenting your findings
- ONLY call promote_to_canon if ALL bars pass
- If bars fail, recommend which role should fix the issues
- Call return_to_sr with your final verdict
"""


def _get_artifact_menu() -> str:
    """Get menu of available artifact types for prompts."""
    # These are the compiled artifact types from domain/ontology/artifacts.md
    # Format: type_id -> brief description
    artifacts = {
        "brief": "Work order from SR to specialist role",
        "scene": "Narrative unit with content, gates, choices",
        "hook_card": "Story hook that captures change/event",
        "canon_entry": "Validated fact in cold store",
        "gatecheck_report": "Quality validation results (Gatekeeper only)",
    }
    lines = ["## Available Artifact Types", ""]
    for artifact_id, desc in artifacts.items():
        lines.append(f"- **{artifact_id}**: {desc}")
    lines.append("")
    lines.append("Use `consult_schema(artifact_type)` to see required/optional fields.")
    return "\n".join(lines)


# Role -> primary artifact types mapping
# This tells each role what artifact types they typically create
ROLE_PRIMARY_ARTIFACTS: dict[str, list[str]] = {
    "showrunner": ["brief"],
    "plotwright": ["scene"],  # Creates scene topology/structure
    "scene_smith": ["scene"],  # Fills scene content/prose
    "lorekeeper": ["canon_entry"],
    "creative_director": ["scene"],  # Style guidance on scenes
    "narrator": ["scene"],  # Runtime scene delivery
    "publisher": [],  # Assembles, doesn't create artifacts
    "gatekeeper": ["gatecheck_report"],
}


def _get_role_artifact_hint(role_id: str) -> str:
    """Get hint about which artifact types this role should create."""
    artifacts = ROLE_PRIMARY_ARTIFACTS.get(role_id.lower(), [])
    if not artifacts:
        return ""

    artifact_list = ", ".join(f"`{a}`" for a in artifacts)
    consult_calls = ", ".join(f'`consult_schema("{a}")`' for a in artifacts)

    return f"""## Your Primary Artifact Types

You typically create: {artifact_list}

**FIRST STEP**: Call {consult_calls} to see required fields before writing.
"""


def _render_prompt(role: RoleIR) -> str:
    """Render role's system prompt using menu+consult pattern.

    The prompt is minimal - just enough for the agent to know:
    1. Who it is (archetype, mandate)
    2. What tools are available
    3. What artifact types exist (menu)
    4. How to look up details (consult_* tools)

    Agents use consult_schema, consult_playbook, consult_role_charter
    to get detailed information when needed.
    """
    # Gatekeeper gets a specialized prompt
    if role.id.lower() == "gatekeeper":
        return _render_gatekeeper_prompt(role)

    # Build constraints section if role has constraints
    constraints_section = ""
    if role.constraints:
        constraints_lines = "\n".join(f"- {c}" for c in role.constraints)
        constraints_section = f"""
## Constraints

{constraints_lines}
"""

    artifact_menu = _get_artifact_menu()
    role_artifact_hint = _get_role_artifact_hint(role.id)

    return f"""You are the **{role.archetype}** ({role.abbr}), a specialist role in QuestFoundry.

## Your Mandate

**{role.mandate}**

{constraints_section}{role_artifact_hint}
## Tools

You MUST use tools to accomplish tasks. Do not describe what you would do - call tools.

### State Tools
- **write_hot_sot(key, value)**: Write artifact to hot_store (key = artifact type like "scene", "brief")
- **read_hot_sot(key)**: Read from hot_store
- **read_cold_sot(key)**: Read from cold_store (canon)

### Knowledge Tools (use these to look up details)
- **consult_schema(artifact_type)**: Get required/optional fields for an artifact type
- **consult_playbook(loop_id)**: Get workflow guidance
- **consult_role_charter(role_id)**: Learn about a role's capabilities

### Completion (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner
  - status: "completed" | "blocked" | "needs_review" | "error"
  - artifacts: list of keys you created/modified

{artifact_menu}

## Workflow

1. Read task from Showrunner delegation
2. Call `consult_schema(artifact_type)` to learn required fields BEFORE creating artifacts
3. Use `write_hot_sot` to save work
4. Call `return_to_sr` when done - THIS IS MANDATORY
"""


def _build_role_tools(
    role: RoleIR,
    state: StudioState,
    cold_store: ColdStore | None = None,
) -> list[BaseTool]:
    """Build tool list for a role.

    All roles get:
    - Consult tools (playbook, schema, role_charter)
    - State tools (read_hot_sot, write_hot_sot, read_cold_sot)
    - return_to_sr (the "done" signal)

    Gatekeeper ONLY gets:
    - promote_to_canon (write to cold_store)

    Role-specific tools from the spec are noted in the prompt but
    may map to the same underlying tools with different permissions.
    """
    tools: list[BaseTool] = []

    # Consult tools - available to ALL roles
    tools.append(ConsultPlaybook())
    tools.append(ConsultRoleCharter())
    tools.append(ConsultSchema())

    # State tools with injected state and role_id
    # Cast StudioState to dict[str, Any] for Pydantic field assignment
    state_dict = cast(dict[str, Any], state)

    # Hot store tools (read/write)
    read_hot_tool = ReadHotSot()
    read_hot_tool.state = state_dict
    tools.append(read_hot_tool)

    write_hot_tool = WriteHotSot()
    write_hot_tool.state = state_dict
    write_hot_tool.role_id = role.id
    tools.append(write_hot_tool)

    # Cold store tools (read for ALL roles)
    read_cold_tool = ReadColdSot()
    read_cold_tool.cold_store = cold_store
    tools.append(read_cold_tool)

    # Gatekeeper ONLY: quality bar evaluation tools + promote_to_canon
    if role.id.lower() == "gatekeeper":
        # Quality bar evaluation tools (8 bars)
        for EvalTool in [
            EvaluateIntegrity,
            EvaluateReachability,
            EvaluateNonlinearity,
            EvaluateGateways,
            EvaluateStyle,
            EvaluateDeterminism,
            EvaluatePresentation,
            EvaluateAccessibility,
        ]:
            eval_tool = EvalTool()
            eval_tool.state = state_dict
            eval_tool.cold_store = cold_store
            tools.append(eval_tool)

        # Create gatecheck report tool
        report_tool = CreateGatecheckReport()
        report_tool.state = state_dict
        report_tool.role_id = role.id
        tools.append(report_tool)

        # Promote to canon tool (final approval)
        promote_tool = PromoteToCanon()
        promote_tool.state = state_dict
        promote_tool.cold_store = cold_store
        promote_tool.role_id = role.id
        tools.append(promote_tool)

    # Return to SR tool with role_id
    return_tool = ReturnToSR()
    return_tool.role_id = role.id
    tools.append(return_tool)

    return tools


class RoleAgent:
    """Agent for a specialist role.

    Each RoleAgent:
    - Maintains its own conversation history across delegations
    - Has access to consult tools + state tools + return_to_sr
    - Has access to cold_store for canon lookup (all roles)
    - Has access to promote_to_canon (Gatekeeper only)
    - Executes until it calls return_to_sr

    Parameters
    ----------
    role : RoleIR
        The role definition from compiled domain.
    llm : BaseChatModel
        LangChain-compatible LLM.
    state : StudioState
        Shared state (hot_store, etc.).
    cold_store : ColdStore | None, optional
        SQLite-based Cold Store for persistent canon. Defaults to None.

    Examples
    --------
    Execute a role for a task::

        agent = RoleAgent(plotwright_ir, llm, state, cold_store)
        result = await agent.execute("Design a topology for a mystery story")
        if result.status == "completed":
            print(f"Created artifacts: {result.artifacts}")
    """

    def __init__(
        self,
        role: RoleIR,
        llm: BaseChatModel,
        state: StudioState,
        cold_store: ColdStore | None = None,
    ):
        self.role = role
        self.llm = llm
        self.state = state
        self.cold_store = cold_store

        # Build tools and prompt (cold_store passed for all roles)
        self.tools = _build_role_tools(role, state, cold_store)
        self.system_prompt = _render_prompt(role)

        # Create executor (maintains conversation history)
        self.executor = ToolExecutor(
            llm=llm,
            tools=self.tools,
            done_tool_name="return_to_sr",
            system_prompt=self.system_prompt,
        )

    @trace_role_execution
    async def execute(self, task: str) -> DelegationResult:
        """Execute a task and return result to SR.

        Parameters
        ----------
        task : str
            The task description from SR's delegation.

        Returns
        -------
        DelegationResult
            The role's work summary including status, artifacts, and message.
        """
        logger.info(f"[{self.role.id}] Starting task: {task[:100]}...")

        # Run executor until return_to_sr is called
        result = await self.executor.run(task)

        if not result.success:
            # Execution failed (max iterations, max failures, etc.)
            logger.error(f"[{self.role.id}] Execution failed: {result.error}")
            return DelegationResult(
                role_id=self.role.id,
                status="error",
                message=f"Execution failed: {result.error}",
                artifacts=[],
                recommendation="Check task clarity or try different approach.",
            )

        # Parse DelegationResult from return_to_sr output
        done_result = result.done_tool_result or {}

        # Handle both direct result and nested delegation_result
        dr = done_result.get("delegation_result", done_result)

        return DelegationResult(
            role_id=dr.get("role_id", self.role.id),
            status=dr.get("status", "completed"),
            message=dr.get("message", "Task completed."),
            artifacts=dr.get("artifacts", []),
            recommendation=dr.get("recommendation"),
        )

    def reset(self) -> None:
        """Reset conversation history for fresh execution."""
        self.executor.reset()


class RoleAgentPool:
    """Pool of RoleAgents for efficient reuse.

    Maintains agent instances per role so conversation history
    persists across delegations within a session.

    Parameters
    ----------
    roles : dict[str, RoleIR]
        Role definitions indexed by role_id.
    llm : BaseChatModel
        LangChain-compatible LLM (shared across agents).
    state : StudioState
        Shared state.
    cold_store : ColdStore | None, optional
        SQLite-based Cold Store for persistent canon. Defaults to None.
    """

    def __init__(
        self,
        roles: dict[str, RoleIR],
        llm: BaseChatModel,
        state: StudioState,
        cold_store: ColdStore | None = None,
    ):
        self.roles = roles
        self.llm = llm
        self.state = state
        self.cold_store = cold_store
        self._agents: dict[str, RoleAgent] = {}

    def get_agent(self, role_id: str) -> RoleAgent | None:
        """Get or create a RoleAgent for the given role.

        Parameters
        ----------
        role_id : str
            The role identifier.

        Returns
        -------
        RoleAgent | None
            The agent, or None if role not found.
        """
        if role_id not in self.roles:
            logger.warning(f"Role '{role_id}' not found in pool")
            return None

        if role_id not in self._agents:
            self._agents[role_id] = RoleAgent(
                role=self.roles[role_id],
                llm=self.llm,
                state=self.state,
                cold_store=self.cold_store,
            )

        return self._agents[role_id]

    def reset_all(self) -> None:
        """Reset all agents' conversation histories."""
        for agent in self._agents.values():
            agent.reset()

    def available_roles(self) -> list[str]:
        """List available role IDs."""
        return list(self.roles.keys())
