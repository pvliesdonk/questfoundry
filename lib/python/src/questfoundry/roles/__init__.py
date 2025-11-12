"""
Role system for QuestFoundry.

Roles are specialized agents that execute specific tasks using LLM providers.
Each role has domain expertise and loads prompts from the spec directory.
"""

from .base import Role, RoleContext, RoleResult
from .registry import RoleRegistry

__all__ = ["Role", "RoleContext", "RoleResult", "RoleRegistry"]
