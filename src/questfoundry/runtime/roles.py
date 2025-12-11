"""RoleAgent - executes specialist roles with their own conversation history.

Each role runs as an independent agent that:
- Has its own conversation history (maintained across delegations)
- Uses tools specific to its function + common consult tools
- Returns control to SR via return_to_sr tool
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from questfoundry.compiler.models import RoleIR
from questfoundry.runtime.executor import ToolExecutor
from questfoundry.runtime.logging import log_role_session_end, log_role_session_start
from questfoundry.runtime.prompts import build_role_prompt
from questfoundry.runtime.state import DelegationResult, StudioState
from questfoundry.runtime.tools.consult import (
    ConsultPlaybook,
    ConsultRoleCharter,
    ConsultSchema,
    ConsultTool,
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
    ListColdStoreKeys,
    ListHotStoreKeys,
    PromoteToCanon,
    ReadColdSot,
    ReadHotSot,
    ReturnToSR,
    WriteHotSot,
)
from questfoundry.runtime.tools.searxng import WebSearchTool
from questfoundry.runtime.tracing import trace_role_execution

if TYPE_CHECKING:
    from questfoundry.runtime.stores import ColdStore

logger = logging.getLogger(__name__)


# =============================================================================
# Runtime Tool Appendix (actual tool names for LLM execution)
# =============================================================================

COMMON_RUNTIME_TOOLS = """
## Runtime Tools

These are the actual tool names you must use:

### State Tools
- **write_hot_sot(key, value)**: Write artifact to hot_store
- **read_hot_sot(key)**: Read from hot_store
- **list_hot_store_keys()**: List available artifacts in hot_store
- **read_cold_sot(key)**: Read from cold_store (canon)
- **list_cold_store_keys()**: List sections/snapshots in cold_store

### Knowledge Tools
- **consult_schema(artifact_type)**: Get required/optional fields for artifact type
- **consult_playbook(query)**: Get workflow guidance
- **consult_role_charter(role_id)**: Learn about another role's capabilities

### Completion (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner
  - status: "completed" | "blocked" | "error"
  - message: what you did (include success/failure details)
  - artifacts: list of keys you created/modified in hot_store
  - recommendation: suggested next action (use for review requests, escalations)

**IMPORTANT**: You MUST call return_to_sr when done. Do not just describe what you would do.
"""

GATEKEEPER_RUNTIME_TOOLS = """
## Runtime Tools

These are the actual tool names you must use:

### Quality Bar Evaluation Tools (8 bars)
- **evaluate_integrity(artifact_id)**: Check for contradictions in canon
- **evaluate_reachability(artifact_id)**: Check all content is accessible
- **evaluate_nonlinearity(artifact_id)**: Check multiple valid paths exist
- **evaluate_gateways(artifact_id)**: Check gates have valid unlock conditions
- **evaluate_style(artifact_id)**: Check voice and tone consistency
- **evaluate_determinism(artifact_id)**: Check same inputs produce same outputs
- **evaluate_presentation(artifact_id)**: Check formatting and structure
- **evaluate_accessibility(artifact_id)**: Check content is usable by all

### Report Tool
- **create_gatecheck_report(target_artifact, bars_checked, status, bar_results, issues, recommendations)**: Create formal validation report

### State Tools
- **read_hot_sot(key)**: Read artifacts to validate
- **list_hot_store_keys()**: List available artifacts
- **read_cold_sot(key)**: Read from cold_store for comparison
- **list_cold_store_keys()**: List sections/snapshots in cold_store

### Knowledge Tools
- **consult_schema(artifact_type)**: Look up artifact requirements

### Completion (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner

**IMPORTANT**: You MUST call return_to_sr when done. Do not just describe what you would do.
"""

LOREKEEPER_RUNTIME_TOOLS = """
## Runtime Tools

These are the actual tool names you must use:

### Canon Promotion Tool (Lorekeeper exclusive)
- **promote_to_canon(artifact_key, section_id)**: Promote validated artifact from hot_store to cold_store

### Web Search Tool (optional)
- **web_search(query, categories)**: Search the web for research and fact-checking
  - query: search terms (e.g., "medieval castle architecture")
  - categories: optional, one of "general", "news", "science", "images"
  - Returns results with title, URL, and snippet
  - If unavailable, returns a message - continue without web search

### State Tools
- **write_hot_sot(key, value)**: Write artifact to hot_store
- **read_hot_sot(key)**: Read from hot_store
- **list_hot_store_keys()**: List available artifacts
- **read_cold_sot(key)**: Read from cold_store
- **list_cold_store_keys()**: List sections/snapshots in cold_store

### Knowledge Tools
- **consult_schema(artifact_type)**: Get artifact requirements
- **consult_playbook(query)**: Get workflow guidance

### Completion (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner

**IMPORTANT**: You MUST call return_to_sr when done. Do not just describe what you would do.
"""

PLOTWRIGHT_RUNTIME_TOOLS = """
## Runtime Tools

These are the actual tool names you must use:

### Web Search Tool (optional)
- **web_search(query, categories)**: Search the web for research and fact-checking
  - query: search terms (e.g., "medieval siege weapons", "undetectable poisons")
  - categories: optional, one of "general", "news", "science", "images"
  - Returns results with title, URL, and snippet
  - If unavailable, returns a message - continue without web search

### State Tools
- **write_hot_sot(key, value)**: Write artifact to hot_store
- **read_hot_sot(key)**: Read from hot_store
- **list_hot_store_keys()**: List available artifacts
- **read_cold_sot(key)**: Read from cold_store
- **list_cold_store_keys()**: List sections/snapshots in cold_store

### Knowledge Tools
- **consult_schema(artifact_type)**: Get artifact requirements
- **consult_playbook(query)**: Get workflow guidance

### Completion (REQUIRED)
- **return_to_sr(status, message, artifacts, recommendation)**: Return control to Showrunner

**IMPORTANT**: You MUST call return_to_sr when done. Do not just describe what you would do.
"""


def _render_prompt(role: RoleIR) -> str:
    """Render role's system prompt: domain template + runtime tools.

    Architecture:
    1. Domain template from build_role_prompt() - identity, mandate, constraints, process
    2. Runtime tool appendix - actual tool names for LLM execution

    The domain template comes from the {role-prompt} directive in MyST files.
    The runtime appendix provides the actual tool names the LLM needs to call.
    """
    # Get domain prompt (renders {role-prompt} Jinja2 template)
    domain_prompt = build_role_prompt(role)

    # Select runtime tools appendix based on role
    role_id = role.id.lower()
    if role_id == "gatekeeper":
        runtime_tools = GATEKEEPER_RUNTIME_TOOLS
    elif role_id == "lorekeeper":
        runtime_tools = LOREKEEPER_RUNTIME_TOOLS
    elif role_id == "plotwright":
        runtime_tools = PLOTWRIGHT_RUNTIME_TOOLS
    else:
        runtime_tools = COMMON_RUNTIME_TOOLS

    return f"""{domain_prompt}

{runtime_tools}
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
    - Quality bar evaluation tools (advisory role)

    Lorekeeper ONLY gets:
    - promote_to_canon (executes promotion after SR authorization)

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

    # Hot store tools (read/write/list)
    read_hot_tool = ReadHotSot()
    read_hot_tool.state = state_dict
    tools.append(read_hot_tool)

    write_hot_tool = WriteHotSot()
    write_hot_tool.state = state_dict
    write_hot_tool.role_id = role.id
    tools.append(write_hot_tool)

    list_hot_tool = ListHotStoreKeys()
    list_hot_tool.state = state_dict
    list_hot_tool.cold_store = cold_store  # For promotion status hints
    tools.append(list_hot_tool)

    # Cold store tools (read/list for ALL roles)
    read_cold_tool = ReadColdSot()
    read_cold_tool.cold_store = cold_store
    tools.append(read_cold_tool)

    list_cold_tool = ListColdStoreKeys()
    list_cold_tool.cold_store = cold_store
    tools.append(list_cold_tool)

    # Gatekeeper ONLY: quality bar evaluation tools (advisory role)
    if role.id.lower() == "gatekeeper":
        # Quality bar evaluation tools (8 bars)
        # Note: These tools all have state/cold_store Fields from Pydantic
        gatekeeper_tool_classes = [
            EvaluateIntegrity,
            EvaluateReachability,
            EvaluateNonlinearity,
            EvaluateGateways,
            EvaluateStyle,
            EvaluateDeterminism,
            EvaluatePresentation,
            EvaluateAccessibility,
        ]
        for ToolClass in gatekeeper_tool_classes:
            eval_tool = ToolClass()  # type: ignore[abstract]
            eval_tool.state = state_dict  # type: ignore[attr-defined]
            eval_tool.cold_store = cold_store  # type: ignore[attr-defined]
            tools.append(eval_tool)

        # Create gatecheck report tool
        report_tool = CreateGatecheckReport()
        report_tool.state = state_dict
        report_tool.role_id = role.id
        tools.append(report_tool)

    # Lorekeeper ONLY: promote_to_canon (executes promotion after SR authorization)
    if role.id.lower() == "lorekeeper":
        promote_tool = PromoteToCanon()
        promote_tool.state = state_dict
        promote_tool.cold_store = cold_store
        promote_tool.role_id = role.id
        tools.append(promote_tool)

    # Web search tool for research (Lorekeeper and Plotwright)
    # Optional - gracefully degrades if SearXNG not configured
    if role.id.lower() in ("lorekeeper", "plotwright"):
        from questfoundry.runtime.config import get_settings

        settings = get_settings()
        web_search_tool = WebSearchTool()
        web_search_tool.searxng_url = settings.searxng.url
        web_search_tool.timeout = settings.searxng.timeout
        web_search_tool.max_results = settings.searxng.max_results
        tools.append(web_search_tool)

    # Return to SR tool with role_id
    return_tool = ReturnToSR()
    return_tool.role_id = role.id
    tools.append(return_tool)

    # ConsultTool needs the tool registry - create it last and inject the registry
    consult_tool_inst = ConsultTool()
    consult_tool_inst.tool_registry = {t.name: t for t in tools}
    tools.append(consult_tool_inst)

    return tools


class RoleAgent:
    """Agent for a specialist role.

    Each RoleAgent:
    - Maintains its own conversation history across delegations
    - Has access to consult tools + state tools + return_to_sr
    - Has access to cold_store for canon lookup (all roles)
    - Has access to promote_to_canon (Lorekeeper only)
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
        stream: bool = False,
        callbacks: Any = None,
    ):
        self.role = role
        self.llm = llm
        self.state = state
        self.cold_store = cold_store
        self.stream = stream
        self.callbacks = callbacks

        # Build tools and prompt (cold_store passed for all roles)
        self.tools = _build_role_tools(role, state, cold_store)
        self.system_prompt = _render_prompt(role)

        # Create executor (maintains conversation history)
        self.executor = ToolExecutor(
            llm=llm,
            tools=self.tools,
            done_tool_name="return_to_sr",
            system_prompt=self.system_prompt,
            stream=stream,
            callbacks=callbacks,
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

        # Generate session ID for VCR correlation
        session_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Log session start for VCR recording
        log_role_session_start(
            role_id=self.role.id,
            task=task,
            system_prompt=self.system_prompt,
            hot_store=dict(self.state.get("hot_store", {})),
            session_id=session_id,
        )

        # Run executor until return_to_sr is called
        result = await self.executor.run(task)

        duration_ms = (time.perf_counter() - start_time) * 1000

        if not result.success:
            # Execution failed (max iterations, max failures, etc.)
            logger.error(f"[{self.role.id}] Execution failed: {result.error}")

            # Log session end for VCR
            log_role_session_end(
                role_id=self.role.id,
                status="error",
                hot_store=dict(self.state.get("hot_store", {})),
                session_id=session_id,
                duration_ms=duration_ms,
            )

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

        status = dr.get("status", "completed")

        # Log session end for VCR
        log_role_session_end(
            role_id=self.role.id,
            status=status,
            hot_store=dict(self.state.get("hot_store", {})),
            session_id=session_id,
            duration_ms=duration_ms,
        )

        return DelegationResult(
            role_id=dr.get("role_id", self.role.id),
            status=status,
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
    stream : bool
        Whether to enable streaming for role agents.
    callbacks : Any
        Streaming callbacks for progress updates.
    """

    def __init__(
        self,
        roles: dict[str, RoleIR],
        llm: BaseChatModel,
        state: StudioState,
        cold_store: ColdStore | None = None,
        stream: bool = False,
        callbacks: Any = None,
    ):
        self.roles = roles
        self.llm = llm
        self.state = state
        self.cold_store = cold_store
        self.stream = stream
        self.callbacks = callbacks
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
                stream=self.stream,
                callbacks=self.callbacks,
            )

        return self._agents[role_id]

    def reset_all(self) -> None:
        """Reset all agents' conversation histories."""
        for agent in self._agents.values():
            agent.reset()

    def available_roles(self) -> list[str]:
        """List available role IDs."""
        return list(self.roles.keys())
