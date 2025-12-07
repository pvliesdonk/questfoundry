"""RoleAgent - executes specialist roles with their own conversation history.

Each role runs as an independent agent that:
- Has its own conversation history (maintained across delegations)
- Uses tools specific to its function + common consult tools
- Returns control to SR via return_to_sr tool
"""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import Template
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
from questfoundry.runtime.tools.role import ReadHotSot, ReturnToSR, WriteHotSot

logger = logging.getLogger(__name__)


def _render_prompt(role: RoleIR) -> str:
    """Render role's prompt template with Jinja2."""
    template_str = role.prompt_template
    if not template_str:
        # Fallback to basic prompt
        return f"""You are the {role.archetype}.

Your mandate: {role.mandate}

When you complete your work, call return_to_sr with:
- status: completed, blocked, needs_review, or error
- message: summary of what you did
- artifacts: list of artifact IDs you created/modified
- recommendation: (optional) suggested next action for SR
"""

    # Render Jinja2 template
    template = Template(template_str)
    return template.render(role=role)


def _build_role_tools(
    role: RoleIR,
    state: StudioState,
) -> list[BaseTool]:
    """Build tool list for a role.

    All roles get:
    - Consult tools (playbook, schema, role_charter)
    - State tools (read_hot_sot, write_hot_sot)
    - return_to_sr (the "done" signal)

    Role-specific tools from the spec are noted in the prompt but
    may map to the same underlying tools with different permissions.
    """
    tools: list[BaseTool] = []

    # Consult tools - available to ALL roles
    tools.append(ConsultPlaybook())
    tools.append(ConsultRoleCharter())
    tools.append(ConsultSchema())

    # State tools with injected state and role_id
    read_tool = ReadHotSot()
    read_tool.state = state

    write_tool = WriteHotSot()
    write_tool.state = state
    write_tool.role_id = role.id

    tools.append(read_tool)
    tools.append(write_tool)

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
    - Executes until it calls return_to_sr

    Parameters
    ----------
    role : RoleIR
        The role definition from compiled domain.
    llm : BaseChatModel
        LangChain-compatible LLM.
    state : StudioState
        Shared state (hot_store, cold_store, etc.).

    Examples
    --------
    Execute a role for a task::

        agent = RoleAgent(plotwright_ir, llm, state)
        result = await agent.execute("Design a topology for a mystery story")
        if result.status == "completed":
            print(f"Created artifacts: {result.artifacts}")
    """

    def __init__(
        self,
        role: RoleIR,
        llm: BaseChatModel,
        state: StudioState,
    ):
        self.role = role
        self.llm = llm
        self.state = state

        # Build tools and prompt
        self.tools = _build_role_tools(role, state)
        self.system_prompt = _render_prompt(role)

        # Create executor (maintains conversation history)
        self.executor = ToolExecutor(
            llm=llm,
            tools=self.tools,
            done_tool_name="return_to_sr",
            system_prompt=self.system_prompt,
        )

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
    """

    def __init__(
        self,
        roles: dict[str, RoleIR],
        llm: BaseChatModel,
        state: StudioState,
    ):
        self.roles = roles
        self.llm = llm
        self.state = state
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
            )

        return self._agents[role_id]

    def reset_all(self) -> None:
        """Reset all agents' conversation histories."""
        for agent in self._agents.values():
            agent.reset()

    def available_roles(self) -> list[str]:
        """List available role IDs."""
        return list(self.roles.keys())
