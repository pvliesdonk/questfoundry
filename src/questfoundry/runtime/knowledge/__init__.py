"""Knowledge injection for agent prompts.

This module provides:
- build_agent_prompt(): Construct system prompts with injected knowledge
- build_playbook_nudge(): Generate playbook progress nudges
- inject_playbook_context(): Add playbook context to existing prompts
- consult_knowledge(): Tool for agents to retrieve knowledge on demand
"""

from questfoundry.runtime.knowledge.injector import (
    build_agent_prompt,
    build_playbook_nudge,
    inject_playbook_context,
)
from questfoundry.runtime.knowledge.retrieval import (
    ConsultKnowledgeTool,
    QueryKnowledgeTool,
    create_consult_knowledge_tool,
    create_query_knowledge_tool,
)

__all__ = [
    "build_agent_prompt",
    "build_playbook_nudge",
    "inject_playbook_context",
    "ConsultKnowledgeTool",
    "QueryKnowledgeTool",
    "create_consult_knowledge_tool",
    "create_query_knowledge_tool",
]
