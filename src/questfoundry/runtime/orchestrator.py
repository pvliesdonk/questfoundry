"""Orchestrator - SR-centric hub-and-spoke execution engine.

The Orchestrator runs the Showrunner as the hub agent that:
1. Receives requests and decides how to handle them
2. Delegates work to specialist roles via delegate_to tool
3. Evaluates results and decides next action
4. Terminates when work is complete

Architecture
------------
```
                    ┌─────────────────┐
                    │   Showrunner    │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │ delegate_to(role, task)
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ Plotwright │    │ Lorekeeper │    │  Narrator  │
    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │ returns DelegationResult
                            ▼
                    ┌─────────────────┐
                    │   Showrunner    │
                    │ (decides next)  │
                    └─────────────────┘
```

The key insight: SR doesn't call roles directly. It calls delegate_to(),
and the orchestrator intercepts this to execute the role and return the result.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from questfoundry.compiler.models import RoleIR

if TYPE_CHECKING:
    from questfoundry.runtime.cold_store import ColdStore
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.roles import RoleAgentPool
from questfoundry.runtime.state import DelegationResult, StudioState, create_initial_state
from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
)
from questfoundry.runtime.tools.sr import DelegateTo, ReadArtifact, Terminate, WriteArtifact
from questfoundry.runtime.tracing import TracedSRTurn, configure_tracing, trace_orchestrator_run

logger = logging.getLogger(__name__)


def _build_sr_system_prompt(roles: dict[str, RoleIR]) -> str:
    """Build SR's system prompt with role menu and soft guards."""
    # Build role menu table
    role_menu = []
    for role_id, role in roles.items():
        if role_id == "showrunner":
            continue  # Don't list self
        role_menu.append(f"| {role.abbr} | **{role_id}** | {role.archetype} | {role.mandate} |")

    role_menu_str = "\n".join(role_menu)

    return f"""You are the **Showrunner (SR)**, the strategic orchestrator of QuestFoundry.

## Your Role

You coordinate creative work by delegating to specialist roles. You don't do detailed work yourself - you:
1. Understand requests and break them into delegatable tasks
2. Choose the right specialist for each task
3. Delegate work via delegate_to(role, task)
4. Evaluate results and decide next steps
5. Terminate when all work is complete

## Available Specialist Roles

| Code | Role ID | Archetype | Mandate |
|------|---------|-----------|---------|
{role_menu_str}

## Your Tools

### Orchestration Tools
- **delegate_to(role, task)**: Assign a task to a specialist role. Returns DelegationResult.
- **terminate(reason)**: End the workflow when all work is complete.

### State Tools
- **read_artifact(key)**: Read artifacts from hot_store or cold_store.
- **write_artifact(key, value)**: Create/update artifacts in hot_store.

### Knowledge Tools (CONSULT BEFORE ACTING)
- **consult_playbook(query)**: Get workflow guidance from loop definitions. Use this FIRST to understand recommended workflow steps.
- **consult_role_charter(role_id)**: Look up a role's capabilities and constraints.
- **consult_schema(artifact_type)**: Look up artifact schema requirements.

## CRITICAL: Consult the Playbook First

**Before your first delegation, you MUST call consult_playbook() to:**
1. Understand the recommended workflow for this type of request
2. Learn which roles typically handle which tasks
3. Identify quality bars that apply at each stage
4. Know what artifacts are expected as deliverables

Example: `consult_playbook("story spark workflow")` or `consult_playbook("what roles for topology design")`

The playbook contains valuable guidance about:
- Standard workflow patterns (Story Spark, Hook Harvest, Scene Craft, etc.)
- Role sequences for common tasks
- Quality gates and validation checkpoints
- When to invoke the Gatekeeper for validation

## Workflow Best Practices

**Starting a new request:**
1. **CALL consult_playbook()** to get workflow guidance for this request type
2. Identify which loop pattern applies (story_spark, scene_craft, etc.)
3. Note which roles and quality bars the loop recommends
4. Plan your delegation sequence based on loop guidance

**For each delegation:**
1. Be clear and specific about goals
2. Include relevant context from prior delegations
3. Example: `delegate_to("plotwright", "Design a 3-act story topology for a mystery adventure...")`

**After receiving DelegationResult:**
1. Check status: completed, blocked, needs_review, or error
2. Review artifacts created/modified
3. Consider the role's recommendation for next steps
4. Consult playbook if unsure what to do next
5. Either delegate to next role OR terminate if complete

**Before promoting content to canon:**
1. Delegate to gatekeeper to validate quality bars
2. Review the GatecheckReport for any issues
3. Only terminate after gatekeeper approval

## Important Guidelines

- ALWAYS consult_playbook before starting a new type of work
- Trust your specialists - don't micromanage or re-do their work
- Be specific about goals when delegating, but let roles decide how to achieve them
- Read DelegationResults carefully - they contain recommendations for next steps
- Delegate to gatekeeper before considering work "done" if quality matters
- Call terminate() with a summary when the workflow is complete
- Roles write artifacts using write_hot_sot - they'll be in hot_store when you need them
"""


def _build_sr_tools(state: StudioState) -> list[BaseTool]:
    """Build SR's tool list."""
    tools: list[BaseTool] = []

    # Consult tools (available to all, including SR)
    tools.append(ConsultPlaybook())
    tools.append(ConsultRoleCharter())
    tools.append(ConsultSchema())

    # SR-specific orchestration tools
    tools.append(DelegateTo())
    tools.append(Terminate())

    # State tools with injected state
    # Cast StudioState to dict[str, Any] for Pydantic field assignment
    state_dict = cast(dict[str, Any], state)
    read_tool = ReadArtifact()
    read_tool.state = state_dict
    tools.append(read_tool)

    write_tool = WriteArtifact()
    write_tool.state = state_dict
    tools.append(write_tool)

    return tools


class Orchestrator:
    """SR-centric orchestration engine.

    Runs the Showrunner as hub, delegating to specialist roles on demand.

    Parameters
    ----------
    roles : dict[str, RoleIR]
        Role definitions indexed by role_id.
    llm : BaseChatModel
        LangChain-compatible LLM.
    max_delegations : int
        Maximum number of delegations before forced termination.
    cold_store : ColdStore | None, optional
        SQLite-based Cold Store for persistent canon. If None, roles won't
        have access to cold_store tools. Defaults to None.

    Examples
    --------
    Run a complete workflow::

        from questfoundry.runtime import get_cold_store

        cold = get_cold_store("project.qfproj")
        orchestrator = Orchestrator(roles, llm, cold_store=cold)
        result = await orchestrator.run("Create a mystery story")
        print(f"Completed: {result.metadata}")
    """

    def __init__(
        self,
        roles: dict[str, RoleIR],
        llm: BaseChatModel,
        max_delegations: int = 50,
        cold_store: ColdStore | None = None,
    ):
        self.roles = roles
        self.llm = llm
        self.max_delegations = max_delegations
        self.cold_store = cold_store

    @trace_orchestrator_run
    async def run(
        self,
        request: str,
        loop_id: str = "default",
    ) -> StudioState:
        """Execute a complete workflow for the given request.

        Parameters
        ----------
        request : str
            The user's request to process.
        loop_id : str
            Loop identifier for state tracking.

        Returns
        -------
        StudioState
            Final state after workflow completion.
        """
        # Configure tracing (sets project name if not set)
        configure_tracing()

        # Create initial state
        state = create_initial_state(loop_id, request)

        # Create role agent pool with cold_store access
        role_pool = RoleAgentPool(self.roles, self.llm, state, self.cold_store)

        # Build SR tools and prompt
        sr_tools = _build_sr_tools(state)
        sr_prompt = _build_sr_system_prompt(self.roles)

        # Create SR executor
        # Note: SR's "done" tool is "terminate", but we also stop on delegate_to
        # so the orchestrator can intercept and execute delegations
        sr_executor = ToolExecutor(
            llm=self.llm,
            tools=sr_tools,
            done_tool_name="terminate",
            system_prompt=sr_prompt,
            stop_tools=["delegate_to"],  # Stop when SR delegates so we can execute it
        )

        # Track delegations and turns
        delegation_count = 0
        sr_turn = 0
        delegation_history: list[dict[str, Any]] = []

        # Initial prompt to SR
        sr_prompt_msg = f"New request: {request}"

        while delegation_count < self.max_delegations:
            sr_turn += 1
            logger.info(f"SR turn {sr_turn}, delegations so far: {delegation_count}")

            # Run SR until it calls delegate_to or terminate (traced)
            async with TracedSRTurn(turn=sr_turn, delegation_count=delegation_count, prompt=sr_prompt_msg):
                sr_result = await sr_executor.run(sr_prompt_msg)

            if not sr_result.success:
                # SR execution failed
                logger.error(f"SR execution failed: {sr_result.error}")
                state["metadata"]["error"] = sr_result.error
                state["metadata"]["delegation_history"] = delegation_history
                return state

            # Check done_tool_result for what stopped execution
            done_result = sr_result.done_tool_result or {}
            stop_tool = done_result.get("_stop_tool", "")

            # Check if SR called terminate
            if stop_tool == "terminate" or "termination" in done_result:
                # Workflow complete
                term = done_result.get("termination", {"reason": "terminated"})
                logger.info(f"Workflow terminated: {term.get('reason')}")
                state["metadata"]["termination"] = term
                state["metadata"]["delegation_history"] = delegation_history
                state["metadata"]["total_delegations"] = delegation_count
                return state

            # Check for delegation request (either from stop_tool or tool_results)
            delegation_request = None
            if stop_tool == "delegate_to" and done_result.get("success"):
                delegation_request = done_result.get("delegation_request")
            if delegation_request is None:
                # Fallback: search tool_results (shouldn't be needed now)
                delegation_request = self._find_delegation_request(sr_result.tool_results)

            if delegation_request is None:
                # SR didn't delegate or terminate - give it another chance
                # This shouldn't happen often due to executor nudging
                sr_prompt_msg = (
                    "You must either delegate_to a role or terminate the workflow. "
                    "What would you like to do next?"
                )
                continue

            # Execute delegation
            role_id = delegation_request["role"]
            task = delegation_request["task"]

            logger.info(f"Delegating to {role_id}: {task[:100]}...")
            delegation_count += 1

            # Get role agent
            agent = role_pool.get_agent(role_id)
            if agent is None:
                # Role not found - inform SR
                sr_prompt_msg = (
                    f"Delegation failed: Role '{role_id}' not found. "
                    f"Available roles: {', '.join(role_pool.available_roles())}"
                )
                continue

            # Execute role
            delegation_result = await agent.execute(task)

            # Record in history
            delegation_history.append(
                {
                    "role": role_id,
                    "task": task,
                    "result": delegation_result.model_dump(),
                }
            )

            # Format result for SR
            sr_prompt_msg = self._format_delegation_result(delegation_result)

        # Max delegations reached
        logger.warning(f"Max delegations ({self.max_delegations}) reached")
        state["metadata"]["error"] = f"Max delegations ({self.max_delegations}) reached"
        state["metadata"]["delegation_history"] = delegation_history
        return state

    def _find_delegation_request(self, tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find a delegation request in tool results."""
        for result in tool_results:
            if result.get("tool") == "delegate_to" and result.get("success"):
                # Parse the result string
                try:
                    parsed = json.loads(result.get("result", "{}"))
                    if parsed.get("success") and "delegation_request" in parsed:
                        return cast(dict[str, Any], parsed["delegation_request"])
                except json.JSONDecodeError:
                    pass
        return None

    def _format_delegation_result(self, result: DelegationResult) -> str:
        """Format DelegationResult for SR to process."""
        lines = [
            f"## Delegation Result from {result.role_id}",
            "",
            f"**Status**: {result.status}",
            "",
            "**Summary**:",
            result.message,
            "",
        ]

        if result.artifacts:
            lines.append(f"**Artifacts Created/Modified**: {', '.join(result.artifacts)}")
            lines.append("")

        if result.recommendation:
            lines.append(f"**Recommendation**: {result.recommendation}")
            lines.append("")

        lines.append("What would you like to do next? Delegate to another role or terminate?")

        return "\n".join(lines)
