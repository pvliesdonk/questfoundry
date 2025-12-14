"""Orchestrator v4 - Direct JSON consumption orchestrator.

This orchestrator works directly with domain-v4 JSON without compilation.
It uses the new Studio model, capability-driven tools, and knowledge injection.

Key differences from v3 orchestrator:
- Accepts Studio instead of dict[str, RoleIR]
- Uses entry_mode to select authoring vs playtest entry agent
- Uses build_agent_tools for capability-driven tool building
- Uses build_agent_prompt for knowledge injection
- Integrates PlaybookTracker for nudging
- Full LangSmith tracing support
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from questfoundry.runtime.domain.models import Agent, Studio
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.flow_control import FlowController
from questfoundry.runtime.knowledge import (
    build_agent_prompt,
    inject_playbook_context,
)
from questfoundry.runtime.messaging import MessageBroker
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
from questfoundry.runtime.tools.sr import DelegateTo, ReadArtifact, WriteArtifact
from questfoundry.runtime.logging import (
    is_structured_logging_configured,
    log_delegation,
)
from questfoundry.runtime.tracing import (
    TracedAgentTurn,
    TracedDelegation,
    configure_tracing,
    trace_orchestrator_run,
)

if TYPE_CHECKING:
    from questfoundry.runtime.checkpoint import Checkpoint, CheckpointStore
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
        cold_store: ColdStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        stream: bool = False,
        callbacks: Any = None,
    ):
        self.studio = studio
        self.llm = llm
        self.entry_mode = entry_mode
        self.max_delegations = max_delegations
        self.cold_store = cold_store
        self.checkpoint_store = checkpoint_store
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

        # Initialize runtime services (Issue #140)
        self.message_broker = MessageBroker.from_studio(studio)
        self.flow_controller = FlowController.from_studio(studio)

    async def _inject_mailbox_context(self, agent_id: str, prompt: str) -> str:
        """Inject mailbox messages into agent prompt.

        Applies the Secretary pattern if needed (summarizes overflow messages),
        then formats messages for context injection.

        Parameters
        ----------
        agent_id : str
            The agent receiving context.
        prompt : str
            The base prompt to augment.

        Returns
        -------
        str
            Prompt with mailbox context prepended.
        """
        # Get agent's mailbox
        mailbox = self.message_broker.get_mailbox(agent_id)

        # Apply Secretary pattern if mailbox is overflowing
        await self.flow_controller.apply_secretary_pattern(mailbox, self.llm)

        # Get messages (expires stale ones based on TTL)
        messages, digests = self.message_broker.get_messages_for_agent(agent_id)

        if not messages and not digests:
            return prompt

        # Format for context
        mailbox_context = self.flow_controller.format_messages_for_context(
            messages, digests
        )

        if mailbox_context:
            return f"## Mailbox Messages\n\n{mailbox_context}\n\n---\n\n{prompt}"

        return prompt

    @trace_orchestrator_run
    async def run(
        self,
        request: str | None = None,
        loop_id: str = "default",
        resume_run_id: str | None = None,
        resume_checkpoint_id: int | None = None,
        force_resume: bool = False,
    ) -> StudioState:
        """Execute a complete workflow for the given request.

        Parameters
        ----------
        request : str | None
            The user's request to process. Required for new workflows,
            optional when resuming (will use original request from checkpoint).
        loop_id : str
            Loop identifier for state tracking.
        resume_run_id : str | None
            If provided, resume from the latest checkpoint of this run.
        resume_checkpoint_id : int | None
            If provided, resume from this specific checkpoint ID.
            Takes precedence over resume_run_id.
        force_resume : bool
            If True, bypass studio version mismatch check when resuming.

        Returns
        -------
        StudioState
            Final state after workflow completion.

        Notes
        -----
        This method is automatically traced with LangSmith when
        LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
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

        # Studio version compatibility check
        if checkpoint is not None:
            self._check_studio_version_compatibility(checkpoint, force_resume)

        # If resuming without a request, retrieve original request from the run
        if request is None and run_id is not None and self.checkpoint_store:
            run = self.checkpoint_store.get_run(run_id)
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            request = run.request
            logger.info(f"Using original request from run: {request[:100]}...")

        # Validate request is provided (either directly or from checkpoint)
        if request is None:
            raise ValueError("Request is required for new workflows")

        # Initialize state - either fresh or from checkpoint
        if checkpoint is not None:
            # Restore state from checkpoint
            state = create_initial_state(loop_id, request)
            state["hot_store"] = checkpoint.hot_store
            delegation_history = checkpoint.delegation_history
            entry_turn = checkpoint.sr_turn
            delegation_count = len(delegation_history)

            # Build resume prompt
            current_prompt = self._build_resume_prompt(request, checkpoint, delegation_history)
            logger.info(
                f"Restored state: turn={entry_turn}, delegations={delegation_count}, "
                f"hot_store keys={list(state['hot_store'].keys())}"
            )
        else:
            # Fresh start
            state = create_initial_state(loop_id, request)
            delegation_history = []
            entry_turn = 0
            delegation_count = 0
            current_prompt = f"New request: {request}"

            # Start new run if checkpointing is enabled
            if self.checkpoint_store:
                run_id = self.checkpoint_store.start_run(request, loop_id)
                logger.info(f"Started new run: {run_id}")

        # Store run_id in metadata for reference
        if run_id:
            state["metadata"]["run_id"] = run_id

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

        # current_prompt is set above (either fresh request or resume prompt)

        while delegation_count < self.max_delegations:
            entry_turn += 1
            logger.info(
                f"Entry agent turn {entry_turn}, delegations so far: {delegation_count}"
            )

            # Inject playbook context if available
            full_prompt = inject_playbook_context(current_prompt, self.playbook_tracker)

            # Inject mailbox context (messages from other agents)
            full_prompt = await self._inject_mailbox_context(
                self.entry_agent_id, full_prompt
            )

            # Generate session ID for structured logging
            session_id = f"{self.entry_agent_id}-{entry_turn}-{uuid.uuid4().hex[:8]}"
            session_start_time = time.perf_counter()

            # Log session start
            self._log_agent_session_start(
                agent_id=self.entry_agent_id,
                task=full_prompt,
                system_prompt=entry_prompt,
                hot_store=dict(state["hot_store"]),
                session_id=session_id,
            )

            # Run entry agent until it delegates or terminates (traced)
            async with TracedAgentTurn(
                agent_id=self.entry_agent_id,
                turn=entry_turn,
                delegation_count=delegation_count,
                prompt=full_prompt,
                extra_metadata={"entry_mode": self.entry_mode},
            ):
                result = await entry_executor.run(full_prompt)

            # Log session end
            session_duration_ms = (time.perf_counter() - session_start_time) * 1000
            self._log_agent_session_end(
                agent_id=self.entry_agent_id,
                status="completed" if result.success else "failed",
                hot_store=dict(state["hot_store"]),
                session_id=session_id,
                duration_ms=session_duration_ms,
            )

            if not result.success:
                logger.error(f"Entry agent execution failed: {result.error}")
                state["metadata"]["error"] = result.error
                state["metadata"]["delegation_history"] = delegation_history

                # Mark run as failed
                if self.checkpoint_store and run_id:
                    self.checkpoint_store.complete_run(run_id, status="failed")

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

                # Mark run as completed
                if self.checkpoint_store and run_id:
                    self.checkpoint_store.complete_run(run_id, status="completed")

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

            # Advance turn counter and expire stale messages (TTL pattern)
            expired = self.message_broker.advance_turn()
            if expired:
                for agent_id, expired_msgs in expired.items():
                    if expired_msgs:
                        logger.debug(
                            f"Expired {len(expired_msgs)} messages for agent {agent_id}"
                        )

            # Save checkpoint after each delegation
            if self.checkpoint_store and run_id:
                # Convert messages to dict format for JSON serialization
                entry_msgs = [m.model_dump() for m in entry_executor.messages]

                checkpoint_id = self.checkpoint_store.save_checkpoint(
                    run_id=run_id,
                    sr_turn=entry_turn,
                    hot_store=dict(state["hot_store"]),
                    sr_messages=entry_msgs,
                    delegation_history=delegation_history,
                    role_id=role_id,
                    role_messages=None,  # v4 doesn't persist role messages yet
                    domain_version=None,  # v4 uses studio version string instead
                )
                logger.info(f"Checkpoint {checkpoint_id}: {role_id} completed (turn {entry_turn})")

            # Format result for entry agent
            current_prompt = self._format_delegation_result(delegation_result)

        # Max delegations reached
        logger.warning(f"Max delegations ({self.max_delegations}) reached")
        state["metadata"]["error"] = f"Max delegations ({self.max_delegations}) reached"
        state["metadata"]["delegation_history"] = delegation_history

        # Mark run as failed
        if self.checkpoint_store and run_id:
            self.checkpoint_store.complete_run(run_id, status="failed")

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

        # Check backpressure (bouncer pattern)
        can_accept, rejection_reason = self.message_broker.check_delegation_capacity(role_id)
        if not can_accept:
            logger.warning(f"Delegation rejected (backpressure): {rejection_reason}")
            return DelegationResult(
                role_id=role_id,
                status="failed",
                message=rejection_reason,
                artifacts=[],
                recommendation="Try delegating to a different agent or wait for current tasks to complete.",
            )

        # Register the delegation (tracks active_delegations for bouncer)
        self.message_broker.register_delegation(role_id)

        try:
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

            # Inject mailbox context for delegated agent
            task_with_mailbox = await self._inject_mailbox_context(role_id, task)

            # Generate session ID for structured logging
            session_id = f"{role_id}-d{delegation_count}-{uuid.uuid4().hex[:8]}"
            session_start_time = time.perf_counter()

            # Log session start
            self._log_agent_session_start(
                agent_id=role_id,
                task=task_with_mailbox,
                system_prompt=agent_prompt,
                hot_store=dict(state["hot_store"]),
                session_id=session_id,
            )

            # Log delegation start to delegations.jsonl
            if is_structured_logging_configured():
                log_delegation(
                    from_role=self.entry_agent_id,
                    to_role=role_id,
                    task=task,
                    delegation_id=session_id,
                    status="started",
                )

            # Execute (traced)
            async with TracedDelegation(
                agent_id=role_id,
                task=task_with_mailbox,
                delegation_count=delegation_count,
                extra_metadata={"archetypes": agent.archetypes},
            ):
                result = await agent_executor.run(task_with_mailbox)

            # Log session end
            session_duration_ms = (time.perf_counter() - session_start_time) * 1000
            self._log_agent_session_end(
                agent_id=role_id,
                status="completed" if result.success else "failed",
                hot_store=dict(state["hot_store"]),
                session_id=session_id,
                duration_ms=session_duration_ms,
            )

            # Log delegation completion to delegations.jsonl
            if is_structured_logging_configured():
                log_delegation(
                    from_role=self.entry_agent_id,
                    to_role=role_id,
                    task=task,
                    delegation_id=session_id,
                    status="completed" if result.success else "failed",
                )
        finally:
            # Complete delegation tracking (bouncer pattern) - always decrement
            self.message_broker.complete_delegation(role_id)

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

    def _build_resume_prompt(
        self,
        request: str,
        checkpoint: Checkpoint,
        delegation_history: list[dict[str, Any]],
    ) -> str:
        """Build a prompt for entry agent when resuming from checkpoint.

        Summarizes previous work so the agent can continue intelligently.
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

    def _check_studio_version_compatibility(
        self, checkpoint: Checkpoint, force_resume: bool
    ) -> None:
        """Check studio version compatibility when resuming from checkpoint.

        Note: v4 checkpoints don't store domain_version (that's a v3 concept).
        This method is here for future use when we add studio version tracking.

        Raises
        ------
        ValueError
            If versions mismatch and force_resume is False.
        """
        # v4 checkpoints use studio.version string instead of domain_version int
        # For now, we just log a warning since v4 checkpoints don't have version info yet
        checkpoint_version = checkpoint.domain_version  # Will be None for v4 checkpoints
        current_version = self.studio.version

        if checkpoint_version is None:
            # v4 checkpoint or legacy - no version check available
            logger.debug(
                f"Checkpoint {checkpoint.id} has no version info. "
                f"Current studio version is {current_version}."
            )
            return

        # For future: compare versions when we add version tracking to checkpoints
        # force_resume would allow bypassing version mismatch
        _ = force_resume  # Placeholder for future version comparison
        logger.debug(f"Studio version: {current_version}")

    def _log_agent_session_start(
        self,
        agent_id: str,
        task: str,
        system_prompt: str,
        hot_store: dict[str, Any],
        session_id: str,
    ) -> None:
        """Log agent session start to structured JSONL logs."""
        try:
            from questfoundry.runtime.logging import (
                is_structured_logging_configured,
                log_role_session_start,
            )

            if not is_structured_logging_configured():
                return

            log_role_session_start(
                role_id=agent_id,
                task=task,
                system_prompt=system_prompt,
                hot_store=hot_store,
                session_id=session_id,
            )
        except Exception as e:
            logger.debug(f"Failed to log session start: {e}")

    def _log_agent_session_end(
        self,
        agent_id: str,
        status: str,
        hot_store: dict[str, Any],
        session_id: str,
        duration_ms: float,
    ) -> None:
        """Log agent session end to structured JSONL logs."""
        try:
            from questfoundry.runtime.logging import (
                is_structured_logging_configured,
                log_role_session_end,
            )

            if not is_structured_logging_configured():
                return

            log_role_session_end(
                role_id=agent_id,
                status=status,
                hot_store=hot_store,
                session_id=session_id,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.debug(f"Failed to log session end: {e}")
