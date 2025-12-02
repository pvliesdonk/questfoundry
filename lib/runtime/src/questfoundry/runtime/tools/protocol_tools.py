from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools.base import _is_injected_arg_type
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model
from pydantic.fields import PydanticUndefined

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.exceptions import StateError
from questfoundry.runtime.models.state import StudioState
from questfoundry.runtime.protocol import Protocol
from questfoundry.runtime.protocol.types import Envelope

logger = logging.getLogger(__name__)


class _StrictToolSchemaMixin:
    """Preserve args_schema config and drop injected fields from tool schema."""

    @property
    def tool_call_schema(self):  # type: ignore[override]
        args_schema = getattr(self, "args_schema", None)
        if args_schema is None or not isinstance(args_schema, type):
            return super().tool_call_schema  # pragma: no cover - fallback

        if hasattr(args_schema, "model_fields"):
            pruned_fields = {}
            for name, field in args_schema.model_fields.items():
                annotated_type = field.annotation
                if field.metadata:
                    annotated_type = Annotated[field.annotation, *field.metadata]
                if _is_injected_arg_type(annotated_type):
                    continue
                pruned_fields[name] = field
            if len(pruned_fields) == len(args_schema.model_fields):
                return args_schema

            field_defs: dict[str, tuple[Any, Field]] = {}
            for name, field in pruned_fields.items():
                default = field.default if field.default is not PydanticUndefined else ...
                field_defs[name] = (
                    field.annotation,
                    Field(default=default, description=field.description),
                )

            return create_model(  # type: ignore[misc]
                f"{args_schema.__name__}Public",
                __config__=args_schema.model_config,
                __module__=args_schema.__module__,
                **field_defs,
            )

        return super().tool_call_schema


class _BaseProtocolTool(_StrictToolSchemaMixin, BaseTool):
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}
    _protocol: Protocol = PrivateAttr()
    _state_manager: StateManager = PrivateAttr()

    def __init__(
        self,
        *,
        protocol: Protocol | None = None,
        state_manager: StateManager | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state_manager = state_manager or StateManager()
        self._protocol = protocol or Protocol(self._state_manager)


class SendProtocolMessage(_BaseProtocolTool):
    name: str = "send_protocol_message"
    description: str = "Send a TU Protocol message to a receiver"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        receiver: str = Field(..., description="Protocol receiver (role ID or '*' for broadcast)")
        intent: str = Field(..., description="Intent code")
        payload: Annotated[dict[str, Any], InjectedToolArg] = Field(
            default_factory=dict, description="Payload"
        )
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Sender role id"
        )
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )

    args_schema = Args

    def _run(
        self,
        receiver: str,
        intent: str,
        payload: dict[str, Any] | None = None,
        role_id: str | None = None,
        state: StudioState | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        if not state:
            raise StateError("State payload is required")
        if payload is None:
            payload = {}
        message = self._protocol.send_message(
            state,
            sender=role_id,
            receiver=receiver,
            intent=intent,
            payload=payload,
        )
        return {"messages": [message]}


class SendProtocolEnvelope(_BaseProtocolTool):
    name: str = "send_protocol_envelope"
    description: str = "Send a full TU Protocol envelope"

    class Args(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        envelope: Envelope = Field(..., description="Protocol envelope")
        role_id: Annotated[str | None, InjectedToolArg] = Field(
            default=None, description="Sender role id"
        )
        state: Annotated[StudioState | None, InjectedToolArg] = Field(
            default=None, description="Current studio state"
        )

    args_schema = Args

    def _run(
        self,
        envelope: Envelope,
        role_id: str,
        state: StudioState,
    ) -> dict[str, Any]:  # type: ignore[override]
        if not state:
            raise StateError("State payload is required")
        if envelope.get("sender") and envelope.get("sender") != role_id:
            raise StateError("Envelope sender must match current role")
        envelope = {**envelope, "sender": envelope.get("sender", role_id)}
        message = self._protocol.send_envelope(state, envelope)
        return {"messages": [message]}
