"""Flow control for the runtime messaging system.

This module implements the three flow control patterns per meta/ specification:

1. **Secretary Pattern** - Auto-summarization when inbox exceeds threshold
2. **Bouncer Pattern** - Backpressure when agent at delegation capacity
3. **TTL Pattern** - Message expiration after N turns

The Secretary pattern requires LLM calls to summarize messages, which has
cost and latency implications. Mitigations:
- Use smaller/cheaper model for summarization
- Pre-compute summaries asynchronously (background job)
- Keep original messages in audit trail

Usage
-----
Create a flow controller from a studio::

    from questfoundry.runtime.flow_control import FlowController

    controller = FlowController.from_studio(studio, llm)

Summarize an overflowing mailbox::

    digest = await controller.summarize_mailbox(mailbox)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .messaging.mailbox import Mailbox
from .messaging.models import Message, MessageDigest, MessagePriority

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.runtime.domain.models import Studio

logger = logging.getLogger(__name__)


SUMMARIZATION_SYSTEM_PROMPT = """You are a secretary assistant that summarizes messages for an AI agent.

Your task is to create a concise digest of the provided messages that:
1. Preserves the key information and context
2. Extracts actionable items
3. Notes the highest urgency level
4. Identifies the senders involved

Format your response as JSON with this structure:
{
  "summary": "A 2-3 sentence summary of the key points from all messages",
  "action_items": ["List of specific action items or requests"],
  "highest_urgency": "critical|high|normal|low"
}

Be concise but don't lose critical information. Action items should be specific and actionable."""


class FlowControlConfig(BaseModel):
    """Configuration for flow control behavior.

    Attributes
    ----------
    max_inbox_size : int
        Maximum messages in inbox before summarization.
    secretary_summarization_trigger : int
        Number of messages that triggers summarization.
    max_active_delegations : int
        Maximum concurrent delegations per agent.
    backpressure_threshold : float
        Fraction of capacity that triggers backpressure warning.
    default_message_ttl_turns : int
        Default TTL for messages without explicit TTL.
    """

    max_inbox_size: int = 20
    secretary_summarization_trigger: int = 10
    max_active_delegations: int = 5
    backpressure_threshold: float = 0.8
    default_message_ttl_turns: int = 24


class FlowController(BaseModel):
    """Manages flow control for the messaging system.

    Coordinates Secretary (summarization), Bouncer (backpressure),
    and TTL (expiration) patterns.

    Attributes
    ----------
    config : FlowControlConfig
        Flow control configuration.
    agent_overrides : dict[str, FlowControlConfig]
        Per-agent configuration overrides.
    """

    config: FlowControlConfig = Field(default_factory=FlowControlConfig)
    agent_overrides: dict[str, FlowControlConfig] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_studio(cls, studio: Studio) -> "FlowController":
        """Create a flow controller from studio configuration.

        Parameters
        ----------
        studio : Studio
            The loaded studio definition.

        Returns
        -------
        FlowController
            Configured flow controller.
        """
        # Get defaults from studio
        defaults = getattr(studio, "defaults", None)
        flow_control = {}
        if defaults and hasattr(defaults, "flow_control"):
            flow_control = defaults.flow_control or {}
        elif isinstance(defaults, dict):
            flow_control = defaults.get("flow_control", {})

        config = FlowControlConfig(
            max_inbox_size=flow_control.get("max_inbox_size", 20),
            secretary_summarization_trigger=flow_control.get(
                "secretary_summarization_trigger", 10
            ),
            max_active_delegations=flow_control.get("max_active_delegations", 5),
            backpressure_threshold=flow_control.get("backpressure_threshold", 0.8),
            default_message_ttl_turns=flow_control.get("default_message_ttl_turns", 24),
        )

        # Collect agent overrides
        agent_overrides = {}
        for agent_id, agent in studio.agents.items():
            if hasattr(agent, "flow_control_override") and agent.flow_control_override:
                override = agent.flow_control_override
                override_config = FlowControlConfig(
                    max_inbox_size=getattr(
                        override, "max_inbox_size", config.max_inbox_size
                    ),
                    secretary_summarization_trigger=getattr(
                        override,
                        "secretary_summarization_trigger",
                        config.secretary_summarization_trigger,
                    ),
                    max_active_delegations=getattr(
                        override,
                        "max_active_delegations",
                        config.max_active_delegations,
                    ),
                    backpressure_threshold=getattr(
                        override,
                        "backpressure_threshold",
                        config.backpressure_threshold,
                    ),
                    default_message_ttl_turns=getattr(
                        override,
                        "default_message_ttl_turns",
                        config.default_message_ttl_turns,
                    ),
                )
                agent_overrides[agent_id] = override_config

        return cls(config=config, agent_overrides=agent_overrides)

    def get_config_for_agent(self, agent_id: str) -> FlowControlConfig:
        """Get the effective flow control config for an agent.

        Parameters
        ----------
        agent_id : str
            The agent ID.

        Returns
        -------
        FlowControlConfig
            The config (agent override or default).
        """
        return self.agent_overrides.get(agent_id, self.config)

    def should_summarize(self, mailbox: Mailbox) -> bool:
        """Check if a mailbox needs Secretary summarization.

        Parameters
        ----------
        mailbox : Mailbox
            The mailbox to check.

        Returns
        -------
        bool
            True if summarization should be triggered.
        """
        config = self.get_config_for_agent(mailbox.agent_id)
        return mailbox.needs_summarization(config.secretary_summarization_trigger)

    def check_backpressure(self, mailbox: Mailbox) -> tuple[bool, float]:
        """Check if an agent is under backpressure.

        Parameters
        ----------
        mailbox : Mailbox
            The mailbox to check.

        Returns
        -------
        tuple[bool, float]
            (is_under_pressure, current_load_fraction)
        """
        config = self.get_config_for_agent(mailbox.agent_id)
        if config.max_active_delegations == 0:
            return False, 0.0

        load = mailbox.active_delegations / config.max_active_delegations
        under_pressure = load >= config.backpressure_threshold
        return under_pressure, load

    async def summarize_messages(
        self,
        messages: list[Message],
        llm: BaseChatModel,
    ) -> MessageDigest:
        """Summarize a list of messages using LLM.

        Parameters
        ----------
        messages : list[Message]
            Messages to summarize.
        llm : BaseChatModel
            The LLM to use for summarization.

        Returns
        -------
        MessageDigest
            The summarized digest.
        """
        import json
        import uuid

        # Format messages for the LLM
        message_texts = []
        for msg in messages:
            text = f"[{msg.type.value}] From: {msg.sender}"
            if msg.recipient:
                text += f" To: {msg.recipient}"
            text += f"\nPriority: {msg.priority.value}"
            text += f"\nContent: {json.dumps(msg.payload, indent=2)}"
            message_texts.append(text)

        combined = "\n\n---\n\n".join(message_texts)

        # Call LLM for summarization
        llm_messages = [
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Summarize these {len(messages)} messages:\n\n{combined}"),
        ]

        try:
            response = await llm.ainvoke(llm_messages)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Map urgency to priority
            urgency_map = {
                "critical": MessagePriority.CRITICAL,
                "high": MessagePriority.HIGH,
                "normal": MessagePriority.NORMAL,
                "low": MessagePriority.LOW,
            }
            highest_priority = urgency_map.get(
                result.get("highest_urgency", "normal"), MessagePriority.NORMAL
            )

        except Exception as e:
            logger.warning("Failed to parse LLM summarization response: %s", e)
            # Fallback to basic summary
            result = {
                "summary": f"Digest of {len(messages)} messages from {', '.join(set(m.sender for m in messages))}",
                "action_items": [],
            }
            highest_priority = max(
                (m.priority for m in messages),
                key=lambda p: list(MessagePriority).index(p),
                default=MessagePriority.NORMAL,
            )

        return MessageDigest(
            id=f"digest-{uuid.uuid4().hex[:12]}",
            original_message_ids=[m.id for m in messages],
            summary=result.get("summary", ""),
            action_items=result.get("action_items", []),
            senders=list(set(m.sender for m in messages)),
            highest_priority=highest_priority,
        )

    async def apply_secretary_pattern(
        self,
        mailbox: Mailbox,
        llm: BaseChatModel,
    ) -> MessageDigest | None:
        """Apply Secretary pattern to a mailbox if needed.

        Checks if summarization is needed and performs it.

        Parameters
        ----------
        mailbox : Mailbox
            The mailbox to potentially summarize.
        llm : BaseChatModel
            The LLM to use for summarization.

        Returns
        -------
        MessageDigest | None
            The created digest, or None if not needed.
        """
        if not self.should_summarize(mailbox):
            return None

        config = self.get_config_for_agent(mailbox.agent_id)
        to_summarize = mailbox.get_messages_needing_summarization(
            config.secretary_summarization_trigger
        )

        if not to_summarize:
            return None

        logger.info(
            "Applying Secretary pattern to %s: summarizing %d messages",
            mailbox.agent_id,
            len(to_summarize),
        )

        digest = await self.summarize_messages(to_summarize, llm)
        mailbox.add_digest(digest)

        return digest

    def format_messages_for_context(
        self,
        messages: list[Message],
        digests: list[MessageDigest],
    ) -> str:
        """Format messages and digests for agent context injection.

        Parameters
        ----------
        messages : list[Message]
            Active messages.
        digests : list[MessageDigest]
            Summarized digests.

        Returns
        -------
        str
            Formatted context string for the agent prompt.
        """
        import json

        parts = []

        # Add digests first (they represent older, summarized messages)
        for digest in digests:
            parts.append(f"## Message Digest (summarized {len(digest.original_message_ids)} messages)")
            parts.append(f"**From:** {', '.join(digest.senders)}")
            parts.append(f"**Priority:** {digest.highest_priority.value}")
            parts.append(f"**Summary:** {digest.summary}")
            if digest.action_items:
                parts.append("**Action Items:**")
                for item in digest.action_items:
                    parts.append(f"  - {item}")
            parts.append("")

        # Add active messages
        for msg in messages:
            parts.append(f"## {msg.type.value.replace('_', ' ').title()}")
            parts.append(f"**From:** {msg.sender}")
            if msg.recipient:
                parts.append(f"**To:** {msg.recipient}")
            parts.append(f"**Priority:** {msg.priority.value}")
            if msg.payload:
                parts.append(f"**Content:** {json.dumps(msg.payload, indent=2)}")
            parts.append("")

        return "\n".join(parts)
