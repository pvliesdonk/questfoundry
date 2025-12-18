"""
Delegation module.

Provides async-first delegation orchestration with domain-aligned
loop termination via playbook-scoped rework budgets.

Key Components:
- PlaybookTracker: Tracks playbook instances and rework budgets
- DelegationBouncer: Pre-flight checks (concurrent limits, budgets)
- AsyncDelegationExecutor: Executes delegations with full lifecycle handling
"""

from questfoundry.runtime.delegation.bouncer import (
    AgentFlowControl,
    BouncerResult,
    DelegationBouncer,
    create_bouncer_from_agent_defs,
)
from questfoundry.runtime.delegation.executor import (
    AgentActivator,
    AsyncDelegationExecutor,
    DelegationContext,
    DelegationResult,
)
from questfoundry.runtime.delegation.nudger import (
    NudgeContext,
    PlaybookNudger,
)
from questfoundry.runtime.delegation.tracker import (
    BudgetCheckResult,
    PlaybookInstance,
    PlaybookTracker,
)

__all__ = [
    # Tracker
    "PlaybookTracker",
    "PlaybookInstance",
    "BudgetCheckResult",
    # Bouncer
    "DelegationBouncer",
    "BouncerResult",
    "AgentFlowControl",
    "create_bouncer_from_agent_defs",
    # Executor
    "AsyncDelegationExecutor",
    "DelegationContext",
    "DelegationResult",
    "AgentActivator",
    # Nudger
    "PlaybookNudger",
    "NudgeContext",
]
