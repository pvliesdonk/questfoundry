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
    from questfoundry.runtime.stores import ColdStore
from questfoundry.runtime.checkpoint import Checkpoint, CheckpointStore
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.prompts import build_sr_prompt
from questfoundry.runtime.roles import RoleAgentPool
from questfoundry.runtime.state import DelegationResult, StudioState, create_initial_state
from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
    ConsultTool,
)
from questfoundry.runtime.tools.role import ListColdStoreKeys, ListHotStoreKeys
from questfoundry.runtime.tools.sr import DelegateTo, ReadArtifact, Terminate, WriteArtifact
from questfoundry.runtime.tracing import TracedSRTurn, configure_tracing, trace_orchestrator_run

logger = logging.getLogger(__name__)


def _build_sr_tools(state: StudioState, cold_store: Any = None) -> list[BaseTool]:
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

    # Discovery tools
    list_hot_tool = ListHotStoreKeys()
    list_hot_tool.state = state_dict
    tools.append(list_hot_tool)

    list_cold_tool = ListColdStoreKeys()
    list_cold_tool.cold_store = cold_store
    tools.append(list_cold_tool)

    # ConsultTool needs the tool registry - create it last and inject the registry
    consult_tool_inst = ConsultTool()
    consult_tool_inst.tool_registry = {t.name: t for t in tools}
    tools.append(consult_tool_inst)

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
    checkpoint_store : CheckpointStore | None, optional
        Checkpoint store for workflow state persistence. If None, checkpointing
        is disabled. Defaults to None.

    Examples
    --------
    Run a complete workflow::

        from questfoundry.runtime import get_cold_store
        from questfoundry.runtime.checkpoint import CheckpointStore

        cold = get_cold_store("project.qfproj")
        checkpoints = CheckpointStore(Path("project_1"))
        orchestrator = Orchestrator(roles, llm, cold_store=cold, checkpoint_store=checkpoints)
        result = await orchestrator.run("Create a mystery story")
        print(f"Completed: {result.metadata}")

    Resume from checkpoint::

        # Resume from latest checkpoint
        result = await orchestrator.run("Create a mystery story", resume_run_id="run-2025-12-09-001")

        # Or resume from specific checkpoint
        result = await orchestrator.run("Create a mystery story", resume_checkpoint_id=5)
    """

    def __init__(
        self,
        roles: dict[str, RoleIR],
        llm: BaseChatModel,
        max_delegations: int = 50,
        cold_store: ColdStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        stream: bool = False,
        callbacks: Any = None,
    ):
        self.roles = roles
        self.llm = llm
        self.max_delegations = max_delegations
        self.cold_store = cold_store
        self.checkpoint_store = checkpoint_store
        self.stream = stream
        self.callbacks = callbacks

    @trace_orchestrator_run
    async def run(
        self,
        request: str,
        loop_id: str = "default",
        resume_run_id: str | None = None,
        resume_checkpoint_id: int | None = None,
    ) -> StudioState:
        """Execute a complete workflow for the given request.

        Parameters
        ----------
        request : str
            The user's request to process.
        loop_id : str
            Loop identifier for state tracking.
        resume_run_id : str | None
            If provided, resume from the latest checkpoint of this run.
        resume_checkpoint_id : int | None
            If provided, resume from this specific checkpoint ID.
            Takes precedence over resume_run_id.

        Returns
        -------
        StudioState
            Final state after workflow completion.
        """
        # Configure tracing (sets project name if not set)
        configure_tracing()

        # Determine if we're resuming from checkpoint
        checkpoint: Checkpoint | None = None
        run_id: str | None = None

        if resume_checkpoint_id is not None and self.checkpoint_store:
            checkpoint = self.checkpoint_store.get_checkpoint(resume_checkpoint_id)
            if checkpoint is None:
                raise ValueError(f"Checkpoint {resume_checkpoint_id} not found")
            run_id = checkpoint.run_id
            logger.info(f"Resuming from checkpoint {resume_checkpoint_id} (run {run_id})")
        elif resume_run_id is not None and self.checkpoint_store:
            checkpoint = self.checkpoint_store.get_latest_checkpoint(resume_run_id)
            if checkpoint is None:
                raise ValueError(f"No checkpoints found for run {resume_run_id}")
            run_id = resume_run_id
            logger.info(f"Resuming from latest checkpoint of run {run_id}")

        # Initialize state - either fresh or from checkpoint
        if checkpoint is not None:
            # Restore state from checkpoint
            state = create_initial_state(loop_id, request)
            state["hot_store"] = checkpoint.hot_store
            delegation_history = checkpoint.delegation_history
            sr_turn = checkpoint.sr_turn
            delegation_count = len(delegation_history)

            # Build resume prompt for SR
            sr_prompt_msg = self._build_resume_prompt(request, checkpoint, delegation_history)
            logger.info(
                f"Restored state: turn={sr_turn}, delegations={delegation_count}, "
                f"hot_store keys={list(state['hot_store'].keys())}"
            )
        else:
            # Fresh start
            state = create_initial_state(loop_id, request)
            delegation_history = []
            sr_turn = 0
            delegation_count = 0
            sr_prompt_msg = f"New request: {request}"

            # Start new run if checkpointing is enabled
            if self.checkpoint_store:
                run_id = self.checkpoint_store.start_run(request, loop_id)
                logger.info(f"Started new run: {run_id}")

        # Store run_id in metadata for reference
        if run_id:
            state["metadata"]["run_id"] = run_id

        # Create role agent pool with cold_store access
        # NOTE: Roles don't get streaming callbacks - only SR uses the Live panel
        # This prevents overlapping/conflicting panel updates
        role_pool = RoleAgentPool(
            self.roles,
            self.llm,
            state,
            self.cold_store,
            stream=self.stream,
            callbacks=None,  # Roles don't stream to avoid panel conflicts
        )

        # Build SR tools and prompt (domain + runtime nudges)
        sr_tools = _build_sr_tools(state, self.cold_store)
        sr_prompt = build_sr_prompt(self.roles)

        # Create SR executor
        # Note: SR's "done" tool is "terminate", but we also stop on delegate_to
        # so the orchestrator can intercept and execute delegations
        sr_executor = ToolExecutor(
            llm=self.llm,
            tools=sr_tools,
            done_tool_name="terminate",
            system_prompt=sr_prompt,
            stop_tools=["delegate_to"],  # Stop when SR delegates so we can execute it
            stream=self.stream,
            callbacks=self.callbacks,
        )

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

                # Mark run as failed
                if self.checkpoint_store and run_id:
                    self.checkpoint_store.complete_run(run_id, status="failed")

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

                # Mark run as completed
                if self.checkpoint_store and run_id:
                    self.checkpoint_store.complete_run(run_id, status="completed")

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
            artifacts = delegation_request.get("artifacts", [])

            # If SR specified artifacts, include them in the task for the role
            if artifacts:
                artifact_list = ", ".join(artifacts)
                task = f"{task}\n\n**Artifacts to work with**: {artifact_list}"

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

            # Save checkpoint after each delegation
            if self.checkpoint_store and run_id:
                # Convert messages to dict format for JSON serialization
                sr_msgs = [m.model_dump() for m in sr_executor.messages]
                role_msgs = None
                if hasattr(agent, "messages") and agent.messages:
                    role_msgs = [m.model_dump() for m in agent.messages]

                checkpoint_id = self.checkpoint_store.save_checkpoint(
                    run_id=run_id,
                    sr_turn=sr_turn,
                    hot_store=dict(state["hot_store"]),
                    sr_messages=sr_msgs,
                    delegation_history=delegation_history,
                    role_id=role_id,
                    role_messages=role_msgs,
                )
                logger.info(f"Checkpoint {checkpoint_id}: {role_id} completed (turn {sr_turn})")

            # Format result for SR
            sr_prompt_msg = self._format_delegation_result(delegation_result)

        # Max delegations reached
        logger.warning(f"Max delegations ({self.max_delegations}) reached")
        state["metadata"]["error"] = f"Max delegations ({self.max_delegations}) reached"
        state["metadata"]["delegation_history"] = delegation_history

        # Mark run as failed
        if self.checkpoint_store and run_id:
            self.checkpoint_store.complete_run(run_id, status="failed")

        return state

    def _build_resume_prompt(
        self,
        request: str,
        checkpoint: Checkpoint,
        delegation_history: list[dict[str, Any]],
    ) -> str:
        """Build a prompt for SR when resuming from checkpoint.

        Summarizes previous work so SR can continue intelligently.
        """
        lines = [
            f"**RESUMING WORKFLOW** (from checkpoint {checkpoint.id})",
            "",
            f"**Original Request**: {request}",
            "",
            f"**Previous Progress**: {len(delegation_history)} delegation(s) completed",
            "",
        ]

        # Summarize recent delegations (last 3)
        if delegation_history:
            lines.append("**Recent Work**:")
            for d in delegation_history[-3:]:
                result = d.get("result", {})
                status = result.get("status", "?")
                role = d.get("role", "?")
                msg = result.get("message", "")[:100]
                lines.append(f"- {role}: [{status}] {msg}...")
            lines.append("")

        # Hot store summary
        if checkpoint.hot_store:
            lines.append(f"**Hot Store**: {list(checkpoint.hot_store.keys())}")
            lines.append("")

        lines.append(
            "Continue from where the previous run left off. "
            "Review the hot_store artifacts and decide what to do next."
        )

        return "\n".join(lines)

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
