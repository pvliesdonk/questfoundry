"""Orchestrator v4 - Direct JSON consumption orchestrator.

This orchestrator works directly with domain-v4 JSON without compilation.
It uses the new Studio model, capability-driven tools, and knowledge injection.

Key differences from v3 orchestrator:
- Accepts Studio instead of dict[str, RoleIR]
- Uses entry_mode to select authoring vs playtest entry agent
- Uses build_agent_tools for capability-driven tool building
- Uses build_agent_prompt for knowledge injection
- Integrates PlaybookTracker for nudging
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from questfoundry.runtime.domain.models import Agent, Studio
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.knowledge import (
    build_agent_prompt,
    inject_playbook_context,
)
from questfoundry.runtime.playbook_tracker import PlaybookTracker
from questfoundry.runtime.state import DelegationResult, StudioState, create_initial_state
from questfoundry.runtime.tools.consult import (
    ConsultRoleCharter,
    ConsultSchema,
    ConsultTool,
)
from questfoundry.runtime.tools.playbook import create_consult_playbook_tool
from questfoundry.runtime.tools.registry import build_agent_tools
from questfoundry.runtime.tools.role import ListColdStoreKeys, ListHotStoreKeys
from questfoundry.runtime.tools.sr import DelegateTo, ReadArtifact, Terminate, WriteArtifact

if TYPE_CHECKING:
    from questfoundry.runtime.stores import ColdStore

logger = logging.getLogger(__name__)


def _build_entry_agent_tools(
    agent: Agent,
    studio: Studio,
    state: StudioState,
    cold_store: Any = None,
    playbook_tracker: PlaybookTracker | None = None,
) -> list[BaseTool]:
    """Build tools for an entry agent.

    Entry agents (like SR or Player-Narrator) get additional tools
    for orchestration and discovery.
    """
    tools: list[BaseTool] = []
    state_dict = cast(dict[str, Any], state)

    # 1. Get capability-driven tools
    capability_tools = build_agent_tools(
        agent, studio, state, cold_store, playbook_tracker
    )
    tools.extend(capability_tools)

    # 2. Add consult tools if not already present
    tool_names = {t.name for t in tools}

    if "consult_playbook" not in tool_names:
        playbook_tool = create_consult_playbook_tool(studio, playbook_tracker)
        tools.append(playbook_tool)

    if "consult_role_charter" not in tool_names:
        tools.append(ConsultRoleCharter())

    if "consult_schema" not in tool_names:
        tools.append(ConsultSchema())

    # 3. Add state access tools if not already present
    if "read_artifact" not in tool_names:
        read_tool = ReadArtifact()
        read_tool.state = state_dict
        tools.append(read_tool)

    if "write_artifact" not in tool_names:
        write_tool = WriteArtifact()
        write_tool.state = state_dict
        tools.append(write_tool)

    # 4. Add discovery tools
    if "list_hot_store_keys" not in tool_names:
        list_hot_tool = ListHotStoreKeys()
        list_hot_tool.state = state_dict
        list_hot_tool.cold_store = cold_store
        tools.append(list_hot_tool)

    if "list_cold_store_keys" not in tool_names:
        list_cold_tool = ListColdStoreKeys()
        list_cold_tool.cold_store = cold_store
        tools.append(list_cold_tool)

    # 5. Add delegation tool for orchestrators
    if "orchestrator" in agent.archetypes and "delegate_to" not in tool_names:
        tools.append(DelegateTo())

    # 6. ConsultTool needs the tool registry - add last
    if "consult_tool" not in tool_names:
        consult_tool_inst = ConsultTool()
        consult_tool_inst.tool_registry = {t.name: t for t in tools}
        tools.append(consult_tool_inst)

    return tools


class OrchestratorV4:
    """Studio-based orchestration engine (v4).

    Uses domain-v4 JSON directly without compilation.

    Parameters
    ----------
    studio : Studio
        Loaded studio containing agents, playbooks, tools, etc.
    llm : BaseChatModel
        LangChain-compatible LLM.
    entry_mode : str
        Entry mode - "authoring" or "playtest" (determines entry agent).
    max_delegations : int
        Maximum number of delegations before forced termination.
    cold_store : ColdStore | None, optional
        SQLite-based Cold Store for persistent canon.
    stream : bool
        Whether to stream responses.
    callbacks : Any
        LangChain callbacks for streaming.

    Examples
    --------
    Run with domain-v4 studio::

        from questfoundry.runtime.domain import load_studio
        from questfoundry.runtime.orchestrator_v4 import OrchestratorV4

        studio = load_studio(Path("domain-v4/studio.json"))
        orchestrator = OrchestratorV4(studio, llm, entry_mode="authoring")
        result = await orchestrator.run("Create a mystery story")
    """

    def __init__(
        self,
        studio: Studio,
        llm: BaseChatModel,
        entry_mode: str = "authoring",
        max_delegations: int = 50,
        cold_store: "ColdStore | None" = None,
        stream: bool = False,
        callbacks: Any = None,
    ):
        self.studio = studio
        self.llm = llm
        self.entry_mode = entry_mode
        self.max_delegations = max_delegations
        self.cold_store = cold_store
        self.stream = stream
        self.callbacks = callbacks

        # Validate entry mode
        entry_agent_id = studio.entry_agents.get(entry_mode)
        if not entry_agent_id:
            available = list(studio.entry_agents.keys())
            raise ValueError(
                f"No entry agent for mode '{entry_mode}'. "
                f"Available modes: {available}"
            )

        self.entry_agent_id = entry_agent_id
        self.playbook_tracker = PlaybookTracker()

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
        # Initialize state
        state = create_initial_state(loop_id, request)
        delegation_history: list[dict[str, Any]] = []
        entry_turn = 0
        delegation_count = 0

        # Get entry agent
        entry_agent = self.studio.agents[self.entry_agent_id]
        logger.info(
            f"Starting {self.entry_mode} workflow with entry agent: {entry_agent.name}"
        )

        # Build entry agent tools and prompt
        entry_tools = _build_entry_agent_tools(
            entry_agent,
            self.studio,
            state,
            self.cold_store,
            self.playbook_tracker,
        )
        entry_prompt = build_agent_prompt(entry_agent, self.studio)

        # Create entry agent executor
        # For orchestrators, stop on delegate_to so we can intercept
        stop_tools = ["delegate_to"] if "orchestrator" in entry_agent.archetypes else []

        entry_executor = ToolExecutor(
            llm=self.llm,
            tools=entry_tools,
            done_tool_name="terminate",
            system_prompt=entry_prompt,
            stop_tools=stop_tools,
            stream=self.stream,
            callbacks=self.callbacks,
        )

        # Initial prompt
        current_prompt = f"New request: {request}"

        while delegation_count < self.max_delegations:
            entry_turn += 1
            logger.info(
                f"Entry agent turn {entry_turn}, delegations so far: {delegation_count}"
            )

            # Inject playbook context if available
            full_prompt = inject_playbook_context(current_prompt, self.playbook_tracker)

            # Run entry agent until it delegates or terminates
            result = await entry_executor.run(full_prompt)

            if not result.success:
                logger.error(f"Entry agent execution failed: {result.error}")
                state["metadata"]["error"] = result.error
                state["metadata"]["delegation_history"] = delegation_history
                return state

            # Check what stopped execution
            done_result = result.done_tool_result or {}
            stop_tool = done_result.get("_stop_tool", "")

            # Check for termination
            if stop_tool == "terminate" or "termination" in done_result:
                term = done_result.get("termination", {"reason": "terminated"})
                logger.info(f"Workflow terminated: {term.get('reason')}")
                state["metadata"]["termination"] = term
                state["metadata"]["delegation_history"] = delegation_history
                state["metadata"]["total_delegations"] = delegation_count
                state["metadata"]["playbook_progress"] = (
                    self.playbook_tracker.get_progress_summary()
                )
                return state

            # Check for delegation (orchestrators only)
            delegation_request = None
            if stop_tool == "delegate_to" and done_result.get("success"):
                delegation_request = done_result.get("delegation_request")

            if delegation_request is None and stop_tool == "delegate_to":
                # Try parsing from tool results
                delegation_request = self._find_delegation_request(result.tool_results)

            if delegation_request is None:
                # Entry agent didn't delegate or terminate
                current_prompt = (
                    "You must either delegate_to a role or terminate the workflow. "
                    "What would you like to do next?"
                )
                continue

            # Execute delegation
            role_id = delegation_request["role"]
            task = delegation_request["task"]
            artifacts = delegation_request.get("artifacts", [])

            if artifacts:
                artifact_list = ", ".join(artifacts)
                task = f"{task}\n\n**Artifacts to work with**: {artifact_list}"

            logger.info(f"Delegating to {role_id}: {task[:100]}...")
            delegation_count += 1

            # Execute the delegated work
            delegation_result = await self._execute_delegation(
                role_id, task, state, delegation_count
            )

            # Record in history
            delegation_history.append(
                {
                    "role": role_id,
                    "task": task,
                    "result": delegation_result.model_dump(),
                }
            )

            # Track artifact creation for playbook nudging
            if delegation_result.artifacts:
                for artifact_key in delegation_result.artifacts:
                    # Extract artifact type from key (format: type/id)
                    artifact_type = artifact_key.split("/")[0] if "/" in artifact_key else artifact_key
                    self.playbook_tracker.on_artifact_created(artifact_type)

            # Format result for entry agent
            current_prompt = self._format_delegation_result(delegation_result)

        # Max delegations reached
        logger.warning(f"Max delegations ({self.max_delegations}) reached")
        state["metadata"]["error"] = f"Max delegations ({self.max_delegations}) reached"
        state["metadata"]["delegation_history"] = delegation_history
        return state

    async def _execute_delegation(
        self,
        role_id: str,
        task: str,
        state: StudioState,
        delegation_count: int,
    ) -> DelegationResult:
        """Execute a delegation to a specialist agent.

        Args:
            role_id: ID of the agent to delegate to
            task: The task to perform
            state: Current studio state
            delegation_count: Number of delegations so far

        Returns:
            DelegationResult from the specialist
        """
        # Get the agent
        agent = self.studio.agents.get(role_id)
        if agent is None:
            return DelegationResult(
                role_id=role_id,
                status="failed",
                message=f"Agent '{role_id}' not found. Available agents: {list(self.studio.agents.keys())}",
                artifacts=[],
                recommendation="Delegate to an existing agent.",
            )

        # Build agent tools and prompt
        agent_tools = build_agent_tools(
            agent,
            self.studio,
            state,
            self.cold_store,
            self.playbook_tracker,
        )
        agent_prompt = build_agent_prompt(agent, self.studio)

        # Create agent executor
        agent_executor = ToolExecutor(
            llm=self.llm,
            tools=agent_tools,
            done_tool_name="return_to_sr",
            system_prompt=agent_prompt,
            stream=self.stream,
            callbacks=None,  # Specialists don't stream to avoid conflicts
        )

        # Execute
        result = await agent_executor.run(task)

        if not result.success:
            return DelegationResult(
                role_id=role_id,
                status="failed",
                message=f"Agent execution failed: {result.error}",
                artifacts=[],
                recommendation="Review the task and try again.",
            )

        # Extract result from done_tool_result
        done_result = result.done_tool_result or {}
        return DelegationResult(
            role_id=role_id,
            status=done_result.get("status", "completed"),
            message=done_result.get("message", "Task completed."),
            artifacts=done_result.get("artifacts", []),
            recommendation=done_result.get("recommendation"),
        )

    def _find_delegation_request(
        self, tool_results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Find a delegation request in tool results."""
        for result in tool_results:
            if result.get("tool") == "delegate_to" and result.get("success"):
                try:
                    parsed = json.loads(result.get("result", "{}"))
                    if parsed.get("success") and "delegation_request" in parsed:
                        return cast(dict[str, Any], parsed["delegation_request"])
                except json.JSONDecodeError:
                    pass
        return None

    def _format_delegation_result(self, result: DelegationResult) -> str:
        """Format DelegationResult for entry agent to process."""
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
